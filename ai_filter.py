class AIFilter:
    def __init__(self):
        # 실제 구현에서는 여기에 모델을 로드하거나 초기화합니다
        pass
        
    def classify(self, texts: list[str]) -> list[bool]:
        """
        텍스트가 광고/홍보인지 분류합니다.
        
        Args:
            texts: 분류할 텍스트 목록
            
        Returns:
            광고/홍보 여부를 나타내는 불리언 리스트 (True: 광고/홍보, False: 일반 텍스트)
        """
        # 간단한 키워드 기반 분류 (실제 구현에서는 ML 모델 사용)
        ad_keywords = ["할인", "특가", "이벤트", "프로모션", "무료체험"]
        
        results = []
        for text in texts:
            is_ad = any(keyword in text for keyword in ad_keywords)
            results.append(is_ad)
            
        return results