import streamlit as st
import pandas as pd
import os
import tempfile
from llm_wrapper import LLMClient

def main():
    st.title("CSV 기반 RAG 시스템 - Ollama 버전")
    
    # 사이드바 설정
    st.sidebar.subheader("시스템 설정")
    
    # Ollama 모델 선택
    llm_client = LLMClient(provider="ollama")
    
    try:
        # Ollama 서버에서 사용 가능한 모델 목록 가져오기
        available_models = llm_client.get_available_models()
        if not available_models:
            available_models = ["exaone3.5:32b"]  # 기본 모델
            st.sidebar.warning("Ollama 서버에서 모델을 가져올 수 없습니다. 기본 모델을 사용합니다.")
        
        selected_model = st.sidebar.selectbox(
            "Ollama 모델 선택",
            options=available_models,
            index=0
        )
        
        # 온도 설정
        temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # LLM 클라이언트 업데이트
        llm_client = LLMClient(provider="ollama", model=selected_model)
        
        # CSV 파일 업로드
        uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
        if not uploaded:
            st.info("먼저 CSV 파일을 업로드하세요.")
            return

        # CSV 파일 처리
        df = pd.read_csv(uploaded)
        
        st.subheader("데이터 미리보기")
        st.dataframe(df.head())
        
        # 질문 입력
        user_question = st.text_input("질문을 입력하세요:", placeholder="이 데이터에 대해 궁금한 점을 물어보세요...")
        
        if user_question and st.button("질문하기"):
            with st.spinner("답변 생성 중..."):
                # 데이터 요약 생성
                data_summary = f"데이터 요약:\n"
                data_summary += f"- 행 수: {df.shape[0]}\n"
                data_summary += f"- 열 수: {df.shape[1]}\n"
                data_summary += f"- 열 이름: {', '.join(df.columns.tolist())}\n"
                data_summary += f"- 처음 5개 행:\n{df.head().to_string()}\n"
                
                # LLM에 질문 전송
                messages = [
                    {"role": "system", "content": f"당신은 데이터 분석 전문가입니다. 다음 데이터를 기반으로 질문에 답변해주세요:\n\n{data_summary}"},
                    {"role": "user", "content": user_question}
                ]
                
                response = llm_client.chat(messages, temperature=temperature)
                
                # 응답 표시
                st.subheader("답변")
                st.write(response)
        
    except Exception as e:
        st.error(f"오류가 발생했습니다: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()