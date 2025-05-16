import streamlit as st
from agent import RetrievalAgent
from storage import VectorStore
import os
import pandas as pd

# 페이지 설정
st.set_page_config(
    page_title="CSV 기반 RAG 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "vector_backend" not in st.session_state:
    st.session_state.vector_backend = "qdrant"

# 메인 페이지
st.title("CSV 기반 RAG 시스템 - 질의응답")

# 사이드바에 Database 선택 옵션 추가
st.sidebar.subheader("시스템 설정")
st.session_state.vector_backend = st.sidebar.selectbox(
    "Database", ["qdrant", "chromadb"], index=0
)

# 사용 가능한 컬렉션 목록 가져오기
store = VectorStore(backend=st.session_state.vector_backend)
collections = store.get_collections()

# 컬렉션이 없으면 기본값 추가
if not collections:
    collections = ["docs"]

# 사이드바에 컬렉션 선택 옵션 추가
selected_collections = st.sidebar.multiselect(
    "사용할 컬렉션 선택", 
    collections,
    default=collections[0] if collections else None
)

st.subheader("질문하기")
query = st.text_input("질문 입력")

if query:
    if not selected_collections:
        st.warning("최소 하나 이상의 컬렉션을 선택해주세요.")
    else:
        try:
            with st.spinner("답변 생성 중..."):
                agent = RetrievalAgent(collections=selected_collections)
                result = agent.ask(query)
                
                st.subheader("답변")
                st.write(result["answer"])
                
                # 참고 문서 표시
                with st.expander("참고 문서", expanded=False):
                    if "sources_metadata" in result:
                        # 클러스터별로 그룹화
                        clusters = {}
                        for source in result["sources_metadata"]:
                            cluster = source["cluster"]
                            if cluster not in clusters:
                                clusters[cluster] = []
                            clusters[cluster].append(source)
                        
                        # 클러스터별로 표시
                        for cluster_id, sources in clusters.items():
                            st.markdown(f"### 클러스터 {cluster_id}")
                            for i, source in enumerate(sources):
                                st.markdown(f"**문서 {i+1}** (문서 ID: {source['doc_id']})")
                                st.info(source["text"])
                                if "score" in source:
                                    st.caption(f"유사도 점수: {source['score']:.4f}")
                                st.divider()
                    else:
                        # 기존 방식으로 표시
                        for i, doc in enumerate(result["sources"]):
                            st.markdown(f"**문서 {i+1}**")
                            st.info(doc)
                            st.divider()
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            st.exception(e)