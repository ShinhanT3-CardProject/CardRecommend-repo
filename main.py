from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# 모델 및 데이터 로드
with open('svd_model.pkl', 'rb') as f:
    svd = pickle.load(f)
with open('user_card_svd_matrix.pkl', 'rb') as f:
    user_card_svd_matrix = pickle.load(f)
card_features_df = pd.read_pickle('card_features.pkl')
user_theme_matrix = pd.read_pickle('user_theme_matrix.pkl')

user_card_matrix = pd.read_pickle('user_card_matrix.pkl')

# 추천 함수 정의
def svd_recommendations(user_id, user_card_matrix, user_card_svd_matrix):
    if user_id not in user_card_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_index = user_card_matrix.index.get_loc(user_id)
    user_scores = user_card_svd_matrix[user_index]
    
    scores = np.dot(user_scores, user_card_svd_matrix.T)
    recommendations = pd.Series(scores, index=user_card_matrix.columns)
    recommendations = recommendations.sort_values(ascending=False)
    
    return recommendations

def content_based_recommendations(user_id, user_theme_matrix, card_features_df):
    if user_id not in user_theme_matrix.index:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_categories = user_theme_matrix.loc[user_id].dropna().index
    if len(user_categories) == 0:
        return pd.Series()  # 추천할 카테고리가 없는 경우
    
    recommended_cards = card_features_df[card_features_df['category_id'].isin(user_categories)]
    
    return recommended_cards['card_id']

def hybrid_recommendations(user_id, user_card_matrix, user_card_svd_matrix, user_theme_matrix, card_features_df, alpha=0.5):
    svd_recommendations_result = svd_recommendations(user_id, user_card_matrix, user_card_svd_matrix)
    content_based_recommendations_result = content_based_recommendations(user_id, user_theme_matrix, card_features_df)
    
    combined_scores = pd.Series(index=user_card_matrix.columns).fillna(0)
    
    for card, score in svd_recommendations_result.items():
        combined_scores[card] += alpha * score
    
    for card in content_based_recommendations_result:
        combined_scores[card] += (1 - alpha) * 1
    
    combined_scores = combined_scores.sort_values(ascending=False)
    return combined_scores

# API 엔드포인트 정의
@app.post("/recommendations")
def get_recommendations(request: RecommendationRequest):
    try:
        recommendations = hybrid_recommendations(request.user_id, user_card_matrix, user_card_svd_matrix, user_theme_matrix, card_features_df, alpha=request.alpha)
        return {"recommendations": recommendations.head(10).index.tolist()}  # 상위 10개 추천 카드 ID 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Uvicorn 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
