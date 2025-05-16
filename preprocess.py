import pandas as pd
from ai_filter import AIFilter
import re

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # 텍스트 컬럼이 없으면 오류 발생
        if "text" not in self.df.columns:
            raise ValueError("데이터프레임에 'text' 컬럼이 없습니다.")

    def remove_missing(self, ratio: float):
        """결측치가 일정 비율 이상인 행 제거"""
        thresh = int((1 - ratio) * len(self.df.columns))
        self.df.dropna(thresh=thresh, axis=0, inplace=True)
        
        # text 컬럼의 결측치 처리
        self.df = self.df.dropna(subset=["text"])

    def keyword_filter(self, keywords: list[str]):
        """키워드 기반 필터링"""
        if len(keywords) > 0:
            # 정규식 패턴으로 변환 (대소문자 구분 없이)
            pattern = "|".join(keywords)
            # 문자열 컬럼에만 적용
            mask = ~self.df["text"].astype(str).str.contains(pattern, case=False, regex=True)
            self.df = self.df[mask]

    def clean_text(self):
        """텍스트 정제"""
        # HTML 태그 제거
        self.df["text"] = self.df["text"].astype(str).apply(lambda x: re.sub(r'<.*?>', '', x))
        # URL 링크 제거 (http/https)
        self.df["text"] = self.df["text"].apply(lambda x: re.sub(r'https?://\S+', '', x))
        # 중복 공백 제거
        self.df["text"] = self.df["text"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
        # 빈 텍스트 제거
        self.df = self.df[self.df["text"].str.strip() != ""]

    def ai_filter(self):
        """AI 기반 필터링"""
        try:
            ai = AIFilter()
            labels = ai.classify(self.df["text"].tolist())
            self.df = self.df[[not lab for lab in labels]]
        except Exception as e:
            print(f"AI 필터링 중 오류 발생: {str(e)}")
            # 오류 발생 시 필터링 건너뛰기

    def run(self, drop_na_ratio: float, ai_filter: bool):
        """전처리 파이프라인 실행"""
        # 원본 인덱스 보존
        self.df = self.df.copy()
        
        # 결측치 제거
        self.remove_missing(drop_na_ratio)
        
        # 텍스트 정제 (HTML 태그, URL 제거, 공백 처리)
        self.clean_text()
        
        # 키워드 기반 필터링 (광고, 홍보, 체험단)
        self.keyword_filter(["광고", "홍보", "체험단", "후기 작성", "구매했어요", "리뷰 이벤트", "할인코드", "무료배송", "체험단", "브랜드", "제품명", "이벤트 참여", "1+1", "공식몰"])
        
        # AI 기반 필터링 (옵션)
        if ai_filter:
            self.ai_filter()
        
        # 인덱스 보존하여 반환
        return self.df
