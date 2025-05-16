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
        
        # Qdrant와 ChromaDB 클라이언트에 따라 다른 검색 방식 사용
        if hasattr(self.store.client, 'search'):
            # Qdrant 클라이언트
            for collection_name in self.collections:
                try:
                    hits = self.store.client.search(
                        collection_name=collection_name, query_vector=em_q.tolist(), limit=top_k
                    )
                    all_results.extend([hit.payload["text"] for hit in hits])
                except Exception:
                    continue
        else:
            # ChromaDB 클라이언트
            for collection_name in self.collections:
                try:
                    collection = self.store.client.get_or_create_collection(collection_name)
                    results = collection.query(query_embeddings=[em_q.tolist()], n_results=top_k)
                    all_results.extend(results["documents"][0])
                except Exception:
                    continue
                    
        return all_results[:top_k]

    def ask(self, query: str):
        docs = self.retrieve(query)
        context = "\n\n".join(docs)
        
        messages = [
            {"role": "system", "content": "당신은 마케팅 전략 전문가입니다."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        answer = self.llm_client.chat(messages=messages)
        
        # 답변과 참고 문서를 함께 반환
        return {
            "answer": answer,
            "sources": docs
        }