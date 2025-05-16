import streamlit as st
from preprocess import Preprocessor
from storage import VectorStore
import pandas as pd
import os

# 페이지 설정
st.set_page_config(page_title="데이터 업로드", layout="wide")

# 세션 상태 확인
if "vector_backend" not in st.session_state:
    st.session_state.vector_backend = "qdrant"

st.title("CSV 기반 RAG 시스템 - 데이터 업로드")

# 사이드바에 Database 선택 옵션 추가
st.sidebar.subheader("시스템 설정")
st.session_state.vector_backend = st.sidebar.selectbox(
    "Database", ["qdrant", "chromadb"], index=0
)
st.sidebar.subheader("전처리 옵션")
drop_na = st.sidebar.slider("최대 결측치 허용 비율 (%)", 0, 100, 50)
use_ai_filter = st.sidebar.checkbox("AI 기반 광고/홍보 필터링 활성화", True)
    
# 컬렉션 이름 입력 추가
collection_name = st.sidebar.text_input("컬렉션 이름", "docs")
if not collection_name:
    st.sidebar.warning("컬렉션 이름을 입력해주세요.")
    collection_name = "docs"

uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
if not uploaded:
    st.info("먼저 CSV 파일을 업로드하세요.")
    st.stop()

try:
    df = pd.read_csv(uploaded)
    
    # 텍스트 컬럼 확인 및 다중 선택 지원
    text_cols = st.multiselect("텍스트 컬럼 선택 (여러 개 선택 가능)", df.columns)
    if not text_cols:
        st.error("텍스트 컬럼을 하나 이상 선택해주세요.")
        st.stop()
        
    # 선택한 컬럼들만 포함하는 새 데이터프레임 생성
    text_df = pd.DataFrame()
    text_df["text"] = df[text_cols].astype(str).apply(lambda row: " ".join(row.values), axis=1)
    
    # 원본 컬럼 정보 저장
    st.session_state.selected_text_columns = text_cols
            
    st.subheader("원본 데이터 미리보기")
    st.write(f"총 {len(df)}")
    st.dataframe(df.head())
    
    # 원본 텍스트 데이터 저장
    original_text_df = text_df.copy()
    
    # 텍스트 데이터만 전처리기에 전달
    pre = Preprocessor(text_df)
    df_clean = pre.run(drop_na_ratio=drop_na/100, ai_filter=use_ai_filter)
    
    # 전처리 후 데이터 표시
    st.subheader("전처리 후 데이터")
    st.write(f"총 {len(df_clean)}건 남음 (원본: {len(original_text_df)}건, 필터링됨: {len(original_text_df) - len(df_clean)}건)")
    st.dataframe(df_clean.head())
    
    # 필터링된 데이터 계산 및 표시 (필터링된 데이터가 있을 때만)
    filtered_count = len(original_text_df) - len(df_clean)
    if filtered_count > 0:
        # 원본 인덱스와 전처리 후 인덱스를 비교하여 필터링된 행 찾기
        filtered_indices = set(original_text_df.index) - set(df_clean.index)
        filtered_df = original_text_df.loc[list(filtered_indices)]
        
        # 필터링 이유 표시
        filtering_reasons = []
        if drop_na < 100:
            filtering_reasons.append("결측치 제거")
        filtering_reasons.append("빈 텍스트 제거")
        filtering_reasons.append("URL 및 HTML 태그 제거 후 빈 텍스트")
        filtering_reasons.append("키워드 필터링 (광고, 홍보, 체험단)")
        if use_ai_filter:
            filtering_reasons.append("AI 기반 광고/홍보 필터링")
        
        # 필터링된 데이터 표시
        st.subheader("필터링된 데이터")
        st.write(f"총 {filtered_count}건 필터링됨")
        st.write("필터링 적용 항목: " + ", ".join(filtering_reasons))
        st.dataframe(filtered_df.head(10))

    # 배치 크기 설정 옵션 추가
    batch_size = st.sidebar.number_input("배치 크기 (대용량 데이터용)", min_value=100, max_value=5000, value=1000, step=100)
    
    if st.button("임베딩 및 저장"):
        with st.spinner("임베딩 및 저장 중..."):
            store = VectorStore(backend=st.session_state.vector_backend, collection_name=collection_name)
            # 전처리된 텍스트 데이터를 배치 단위로 전달
            total_texts = df_clean["text"].tolist()
            
            # 진행 상황 표시를 위한 프로그레스 바
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                store.ingest(total_texts, batch_size=batch_size)
                progress_bar.progress(100)
                status_text.text(f"처리 완료: {len(total_texts)}/{len(total_texts)} (100%)")
                st.success(f"벡터 저장 완료 (컬렉션: {collection_name})")
                st.info("이제 메인 페이지에서 질문을 할 수 있습니다.")
            except Exception as e:
                st.error(f"저장 중 오류 발생: {str(e)}")
                st.info("배치 크기를 줄여서 다시 시도해보세요.")

except Exception as e:
    st.error(f"오류가 발생했습니다: {str(e)}")
    st.exception(e)