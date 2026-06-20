from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load(
    "../models/energy_prediction_model.pkl"
)

scaler = joblib.load(
    "../models/scaler.pkl"
)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    scaled = scaler.transform(df)

    prediction = model.predict(
        scaled
    )[0]

    return jsonify({
        "predicted_energy_consumption":
        round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
