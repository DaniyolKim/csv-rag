from langchain.prompts import PromptTemplate
from storage import VectorStore
from llm_wrapper import LLMClient

class RetrievalAgent:
    def __init__(self, collections=None):
        self.store = VectorStore()
        self.llm_client = LLMClient(provider="ollama")
        self.collections = collections or ["docs"]

    def retrieve(self, query: str, top_k=5):
        em_q = self.store.model.encode([query])[0]
        all_results = []
        all_metadata = []
        
        # Qdrant와 ChromaDB 클라이언트에 따라 다른 검색 방식 사용
        if hasattr(self.store.client, 'search'):
            # Qdrant 클라이언트
            for collection_name in self.collections:
                try:
                    hits = self.store.client.search(
                        collection_name=collection_name, query_vector=em_q.tolist(), limit=top_k
                    )
                    for hit in hits:
                        all_results.append(hit.payload["text"])
                        all_metadata.append({
                            "doc_id": hit.payload.get("doc_id", "unknown"),
                            "cluster": hit.payload.get("cluster", 0),
                            "score": hit.score
                        })
                except Exception:
                    continue
        else:
            # ChromaDB 클라이언트
            for collection_name in self.collections:
                try:
                    collection = self.store.client.get_or_create_collection(collection_name)
                    results = collection.query(query_embeddings=[em_q.tolist()], n_results=top_k)
                    all_results.extend(results["documents"][0])
                    
                    # ChromaDB 메타데이터 추출
                    if "metadatas" in results and results["metadatas"]:
                        for metadata in results["metadatas"][0]:
                            all_metadata.append({
                                "doc_id": metadata.get("doc_id", "unknown"),
                                "cluster": metadata.get("cluster", "0")
                            })
                    else:
                        # 메타데이터가 없는 경우 기본값 추가
                        all_metadata.extend([{"doc_id": "unknown", "cluster": "0"}] * len(results["documents"][0]))
                except Exception:
                    continue
        
        # 결과와 메타데이터를 함께 반환
        results_with_metadata = list(zip(all_results, all_metadata))
        
        # 클러스터별로 그룹화하여 다양한 클러스터의 결과를 포함하도록 함
        if results_with_metadata:
            # 클러스터별로 그룹화
            clusters = {}
            for text, meta in results_with_metadata:
                cluster = meta["cluster"]
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append((text, meta))
            
            # 각 클러스터에서 최상위 결과를 선택
            final_results = []
            final_metadata = []
            
            # 클러스터별로 최대 2개씩 결과 선택
            for cluster_results in clusters.values():
                for text, meta in cluster_results[:2]:  # 각 클러스터에서 최대 2개
                    if len(final_results) < top_k:  # top_k 제한
                        final_results.append(text)
                        final_metadata.append(meta)
            
            # 남은 슬롯을 채우기 위해 아직 선택되지 않은 결과 추가
            if len(final_results) < top_k:
                for text, meta in results_with_metadata:
                    if text not in final_results and len(final_results) < top_k:
                        final_results.append(text)
                        final_metadata.append(meta)
            
            return final_results[:top_k], final_metadata[:top_k]
        
        # 결과가 없거나 클러스터링이 없는 경우 원래 결과 반환
        return all_results[:top_k], all_metadata[:top_k]

    def ask(self, query: str):
        docs, metadata = self.retrieve(query)
        context = "\n\n".join(docs)
        
        messages = [
            {"role": "system", "content": "당신은 마케팅 전략 전문가입니다."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        answer = self.llm_client.chat(messages=messages)
        
        # 답변, 참고 문서, 메타데이터를 함께 반환
        sources_with_metadata = []
        for doc, meta in zip(docs, metadata):
            sources_with_metadata.append({
                "text": doc,
                "doc_id": meta.get("doc_id", "unknown"),
                "cluster": meta.get("cluster", 0),
                "score": meta.get("score", 0.0)
            })
        
        return {
            "answer": answer,
            "sources": docs,
            "sources_metadata": sources_with_metadata
        }