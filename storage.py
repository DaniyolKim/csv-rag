from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import chromadb
import numpy as np

class VectorStore:
    def __init__(self, backend="qdrant", collection_name="docs"):
        self.backend = backend
        self.collection_name = collection_name
        self.model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        self.vector_size = self.model.get_sentence_embedding_dimension()
        
        if backend == "qdrant":
            self.client = QdrantClient(url="http://localhost:6333")
            # 컬렉션이 없으면 생성
            try:
                self.client.get_collection(collection_name)
            except Exception:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
        else:
            self.client = chromadb.Client()
            self.collection = self.client.get_or_create_collection(collection_name)
    
    def get_collections(self):
        if self.backend == "qdrant":
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        else:
            return self.client.list_collections()

    def ingest(self, texts: list[str], batch_size=1000):
        """
        텍스트를 임베딩하고 벡터 저장소에 저장합니다.
        대용량 데이터는 배치 단위로 처리합니다.
        """
        total = len(texts)
        for i in range(0, total, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embs = self.model.encode(batch_texts, show_progress_bar=True)
            
            if self.backend == "qdrant":
                # 배치별로 고유한 ID 생성
                start_idx = i
                points = [{"id": start_idx + idx, "vector": vec.tolist(), "payload": {"text": txt}}
                          for idx, (vec, txt) in enumerate(zip(batch_embs, batch_texts))]
                self.client.upsert(collection_name=self.collection_name, points=points)
            else:
                # ChromaDB는 ID가 필요합니다
                ids = [str(i + idx) for idx in range(len(batch_texts))]
                self.collection.add(
                    documents=batch_texts,
                    embeddings=[vec.tolist() for vec in batch_embs],
                    ids=ids
                )
            
            print(f"처리 진행: {min(i+batch_size, total)}/{total} ({(min(i+batch_size, total)/total*100):.1f}%)")
