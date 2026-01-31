import os
from catboost import CatBoostClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime


def get_engine():
    db = ("postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml")
    return create_engine(db)


class PostGet(BaseModel):
    id: int
    text: str
    topic: str
    class Config:
        orm_mode = True

def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH

def load_models():
    model_path = get_model_path("C:/Users/Vadim2.0/karpov/Final_project/catboost_model")
    model = CatBoostClassifier()
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = get_engine()
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)

def load_user_features() -> pd.DataFrame:
    query = "SELECT * FROM vadim_meshkov_pbr7487_user_features_lesson_22"
    user_features = batch_load_sql(query)
    user_features = user_features.set_index("user_id")
    return user_features

def load_post_features():
    engine = get_engine()

    post_text_df = pd.read_sql(
        "SELECT * FROM public.post_text_df",
        con=engine
    )
    posts_df = post_text_df[["post_id", "text", "topic"]].copy()
    posts_df = posts_df.rename(columns={"post_id": "id"})
    posts_df = posts_df.set_index("id")
    query = "SELECT * FROM vadim_meshkov_pbr7487_post_features_lesson_22"
    post_features = batch_load_sql(query)
    post_features = post_features.set_index("post_id")
    return post_features, posts_df

model = load_models()
user_features = load_user_features()
post_features, post_df = load_post_features()
user_cols = list(user_features.columns)
post_cols = list(post_features.columns)
time_cols = ['day_of_week', 'hour']

FEATURE_COLUMNS = time_cols + user_cols + post_cols
#print(FEATURE_COLUMNS)


def make_features_user(user_id: int, time: datetime) -> pd.DataFrame:
    """
    Собираем датафрейм для отработки модели
    """

    if user_id not in user_features.index:
        raise HTTPException(status_code=404, detail="User not found")

    user_row = user_features.loc[user_id]

    features = post_features.copy()

    for col in user_cols:
        features[col] = user_row[col]

    features["hour"] = time.hour
    features["day_of_week"] = time.weekday()

    X = features[FEATURE_COLUMNS].fillna(0)

    X.index.name = "post_id"

    return X

app = FastAPI()
@app.get("/post/recommendations/", response_model=List[PostGet])
def recommendation_posts(id: int, time: datetime, limit: int = 5) -> List[PostGet]:
    """
    Эндпоинт для получения топ limit рекомендаций постов для пользователя с id
    """

    X = make_features_user(id, time)

    preds = model.predict_proba(X)[:, 1]

    df_pred = pd.DataFrame(
        {
            "post_id": X.index.values,
            "pred": preds
        }
    )

    top_posts = df_pred.sort_values("pred", ascending=False).head(limit)

    recommendations: List[PostGet] = []

    for post_id in top_posts["post_id"].values:
        row = post_df.loc[post_id]
        recommendations.append(
            PostGet(
                id=post_id,
                text=row["text"],
                topic=row["topic"]
            )
        )

    return recommendations
