# API 모델 정의
class RecommendationRequest(BaseModel):
    user_id: int
    alpha: float = 0.5  # 하이브리드 추천의 alpha 값, 기본값 0.5