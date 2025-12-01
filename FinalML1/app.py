from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    # Align columns
    model_cols = scaler.feature_names_in_
    df = df.reindex(columns=model_cols, fill_value=0)
    X = scaler.transform(df)
    pred = model.predict(X)[0]
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(port=5000)
