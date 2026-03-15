import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from config.settings import settings

class QdrantDbClient:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            check_compatibility=False
        )
        self.collection_name = "pet_light_rag"

    def get_client(self) -> QdrantClient:
        return self.client
    
    def create_collection_if_not_exists(self, vector_size: int = 1536):
        from qdrant_client.http.models import Distance, VectorParams
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            
qdrant_db = QdrantDbClient()
