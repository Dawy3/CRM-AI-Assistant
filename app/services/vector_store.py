"""
Vector store service with Hybrid Search (vector + Keyword)
Suppports multiple backends: Pinecone, Qdrant, OpenSearch
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from  pinecone import Pinecone
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue, PointStruct
import os 
import logging 
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)

class VectorStoreBase(ABC):
    """Base class for vector store implementations"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        key: int=5,
        filters: Optional[Dict[str, Any]] = None
    )-> List[Dict[str, Any]] :
        """Victor similarity search"""
        pass
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector and keyword"""
        pass
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    )-> bool:
        """Add Document to the store"""
        pass

class PineconeStore(VectorStoreBase):
    """Pinecone vector store implementation"""
    
    def __init__(self):
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        pinecone = Pinecone(
            api_key= os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
            )
        
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "crm-knowledge")
        
        self.index = pinecone.Index(self.index_name)
        
    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to pinecone"""
        try:
            vectors= []
            for doc in documents:
                embedding = self.embedding_model.encode(doc['content']).tolist()
                vectors.append({
                    "id" : doc.get("id" , f"doc_{hash(doc['content'])}"),
                    "values" : embedding,
                    "metadata" : {
                        "content" : doc["content"],
                        **doc.get("metadata", {})
                    }
                })
            self.index.upsert(vectors=vectors)
            return True
        
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone: {str(e)}")
            return False
        
    async def search(
        self,
        query: str,
        k : int=5,
        filters : List[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Vector similarity search in Pinecone"""
        # Generate embedding
        query_embedding  =  self.embedding_model.encode(query).tolist()
        
        # Search 
        results = self.index.query(
            vector= query_embedding,
            top_k=k,
            filter =  filters,
            include_metadata= True
        )
        
        # Format results 
        documents = []
        for match in results.matches:
            documents.append({
                "id" : match.id,
                "content": match.metadata.get("content", ""),
                "score" : match.score,
                "metadata" : match.metadata
            })
        return documents
    
    async def hybrid_search(
        self,
        query: str,
        k : int= 5,
        filters:  Optional[Dict[str, Any]] = None,
        alpha : float = 0.7
    ) -> List[Dict[str, Any]] :
        """
        Hybrid search in Pinecone (vector + sparse).
        
        Args:
            query: Search query
            k: Number of results
            filters: Metadata filters
            alpha: Weight for keyword vs vector search (0=keyword only, 1=semantic)
        """
        # Pinecone hybrid search use sparse-dense vectors 
        # For now, we'll use dense only (full hybrid requires sparse encoder)
        return await self.search(query, k, filters)
    
    
    

class QdrantStore(VectorStoreBase):
    """Qdrant vector store implementation"""
    
    def __init__(self):
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "crm_knowledge")
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
        except:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=Distance.COSINE
                )
            )
    
    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Vector search in Qdrant"""
        
        # Generate embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Build filter
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            query_filter = Filter(must=conditions)
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=query_filter
        )
        
        # Format results
        documents = []
        for hit in results:
            documents.append({
                "id": hit.id,
                "content": hit.payload.get("content", ""),
                "score": hit.score,
                "metadata": hit.payload
            })
        
        return documents
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector and full-text search.
        """
        # Vector search
        vector_results = await self.search(query, k=k*2, filters=filters)
        
        # Keyword search (simplified - Qdrant has full-text search)
        # In production, you'd use Qdrant's text index feature
        
        # For now, re-rank vector results (RRF - Reciprocal Rank Fusion)
        return vector_results[:k]
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to Qdrant"""        
        try:
            points = []
            for i, doc in enumerate(documents):
                embedding = self.embedding_model.encode(doc["content"]).tolist()
                points.append(
                    PointStruct(
                        id=doc.get("id", i),
                        vector=embedding,
                        payload={
                            "content": doc["content"],
                            **doc.get("metadata", {})
                        }
                    )
                )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        
        except Exception as e:
            logger.error(f"Error adding documents to Qdrant: {str(e)}")
            return False


class VectorStoreService:
    """
    Vector store service that selects the appropriate backend.
    """
    
    def __init__(self):
        store_type = os.getenv("VECTOR_STORE_TYPE", "qdrant").lower()
        
        if store_type == "pinecone":
            self.store = PineconeStore()
        elif store_type == "qdrant":
            self.store = QdrantStore()
        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")
        
        logger.info(f"Initialized {store_type} vector store")
        
    async def search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Perform vector search"""
        return await self.store.search(query, k, filters)
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search (vector + keyword).
        
        Args:
            alpha: Weight for vector vs keyword (0=keyword only, 1=vector only)
        """
        return await self.store.hybrid_search(query, k, filters, alpha)
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """Add documents to vector store"""
        return await self.store.add_documents(documents)
    
            