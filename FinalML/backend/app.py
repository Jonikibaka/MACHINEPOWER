import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
from flask import Flask, render_template, request, jsonify
from model import StudentNet
from preprocess import transform_df


# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))  

MODEL_PATH = os.path.join(PROJECT_DIR, 'student_net.pt')
PREP_PATH = os.path.join(PROJECT_DIR, 'preprocessor.joblib')

TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

preprocessor = joblib.load(PREP_PATH)

# Загружаем модель
device = torch.device('cpu')
checkpoint = torch.load(MODEL_PATH, map_location=device)
input_dim = checkpoint['input_dim']
model = StudentNet(input_dim=input_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


# Главная страница
@app.route('/')
def home():
    return render_template('index.html')

# API для предсказания
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    expected_columns = preprocessor.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            if col in ['age', 'study_hours', 'attendance_percentage', 
                       'math_score', 'science_score', 'english_score', 'overall_score']:
                df[col] = 0
            else:
                df[col] = "unknown"

    df_transformed = transform_df(df, preprocessor)
    X = torch.tensor(df_transformed, dtype=torch.float32)
    # Предсказание модели
    with torch.no_grad():
        output = model(X)
        predicted_class = torch.argmax(output, dim=1).item()
    grade_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}
    predicted_grade = grade_map.get(predicted_class, "Unknown")

    return {"predicted_grade": predicted_grade}

if __name__ == '__main__':
    os.chdir(BASE_DIR)  
    app.run(debug=True)