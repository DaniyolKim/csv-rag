from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import chromadb
import numpy as np
import kss  # Korean Sentence Splitter
from sklearn.cluster import KMeans

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

    def split_into_sentences(self, text):
        """
        텍스트를 문장 단위로 분리합니다.
        """
        try:
            sentences = kss.split_sentences(text)
            return [sent.strip() for sent in sentences if sent.strip()]
        except Exception as e:
            print(f"문장 분리 중 오류 발생: {str(e)}")
            # 오류 발생 시 원본 텍스트를 그대로 반환
            return [text]
    
    def cluster_sentences(self, embeddings, n_clusters=5):
        """
        문장 임베딩을 클러스터링합니다.
        """
        if len(embeddings) < n_clusters:
            # 문장 수가 클러스터 수보다 적으면 클러스터링 생략
            return list(range(len(embeddings)))
        
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(embeddings)
        except Exception as e:
            print(f"클러스터링 중 오류 발생: {str(e)}")
            # 오류 발생 시 모든 문장을 같은 클러스터로 할당
            return [0] * len(embeddings)
    
    def ingest(self, texts: list[str], batch_size=1000, cluster=True, n_clusters=5):
        """
        텍스트를 문장 단위로 분리하고 임베딩한 후 벡터 저장소에 저장합니다.
        대용량 데이터는 배치 단위로 처리합니다.
        
        Args:
            texts: 저장할 텍스트 리스트
            batch_size: 한 번에 처리할 배치 크기
            cluster: 클러스터링 수행 여부
            n_clusters: 클러스터 수
        """
        # 모든 텍스트를 문장으로 분리
        all_sentences = []
        doc_ids = []  # 원본 문서 ID 추적
        
        for doc_idx, text in enumerate(texts):
            sentences = self.split_into_sentences(text)
            all_sentences.extend(sentences)
            doc_ids.extend([doc_idx] * len(sentences))
        
        total = len(all_sentences)
        print(f"총 {len(texts)}개 문서에서 {total}개 문장을 추출했습니다.")
        
        # 배치 단위로 처리
        for i in range(0, total, batch_size):
            batch_sentences = all_sentences[i:i+batch_size]
            batch_doc_ids = doc_ids[i:i+batch_size]
            
            # 문장 임베딩
            batch_embs = self.model.encode(batch_sentences, show_progress_bar=True)
            
            # 클러스터링 수행 (옵션)
            if cluster and len(batch_sentences) > 1:
                clusters = self.cluster_sentences(batch_embs, n_clusters)
            else:
                clusters = [0] * len(batch_sentences)
            
            if self.backend == "qdrant":
                # 배치별로 고유한 ID 생성
                start_idx = i
                points = [
                    {
                        "id": start_idx + idx, 
                        "vector": vec.tolist(), 
                        "payload": {
                            "text": sent,
                            "doc_id": doc_id,
                            "cluster": int(clust)
                        }
                    }
                    for idx, (vec, sent, doc_id, clust) in enumerate(
                        zip(batch_embs, batch_sentences, batch_doc_ids, clusters)
                    )
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
            else:
                # ChromaDB는 ID가 필요합니다
                ids = [str(i + idx) for idx in range(len(batch_sentences))]
                metadatas = [
                    {
                        "doc_id": str(doc_id),
                        "cluster": str(clust)
                    }
                    for doc_id, clust in zip(batch_doc_ids, clusters)
                ]
                self.collection.add(
                    documents=batch_sentences,
                    embeddings=[vec.tolist() for vec in batch_embs],
                    ids=ids,
                    metadatas=metadatas
                )
            
            print(f"처리 진행: {min(i+batch_size, total)}/{total} ({(min(i+batch_size, total)/total*100):.1f}%)")
