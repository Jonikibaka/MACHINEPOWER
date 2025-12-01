import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# Названия числовых и категориальных признаков
NUMERIC = [
    "age", "study_hours", "attendance_percentage",
    "math_score", "science_score", "english_score", "overall_score"
]
CAT = [
    "gender", "school_type", "parent_education",
    "internet_access", "travel_time", "extra_activities", "study_method"
]

def build_preprocessor():
    #"Создаём препроцессор с StandardScaler и OneHotEncoder
    num = StandardScaler()
    cat = OneHotEncoder(handle_unknown='ignore')
    pre = ColumnTransformer([
        ("num", num, NUMERIC),
        ("cat", cat, CAT)
    ])
    return pre

def fit_and_save_preprocessor(df, path="preprocessor.joblib"):
    #Обучаем препроцессор и сохраняем в файл
    X = df.drop(columns=["final_grade", "student_id"], errors='ignore')
    pre = build_preprocessor()
    pre.fit(X)
    joblib.dump(pre, path)
    return pre

def load_preprocessor(path="preprocessor.joblib"):
    #Загружаем препроцессор из файла
    return joblib.load(path)

def transform_df(df, preprocessor):
    #Преобразуем DataFrame в numpy массив с помощью препроцессора
    X = df.drop(columns=['student_id'], errors='ignore')
    if 'final_grade' in X.columns:
        X = X.drop(columns=['final_grade'])
    return preprocessor.transform(X)
