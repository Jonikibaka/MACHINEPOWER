import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Student Grade Prediction")

age = st.number_input("Age", 10, 25)
study_hours = st.number_input("Study Hours", 0.0, 12.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)

if st.button("Predict"):
    df = pd.DataFrame([{
        "age": age,
        "study_hours": study_hours,
        "attendance_percentage": attendance
    }])
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X = scaler.transform(df)
    pred = model.predict(X)[0]
    st.success(f"Predicted grade: {pred}")
