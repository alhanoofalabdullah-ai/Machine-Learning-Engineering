from flask import Flask
from flask import request
from flask import jsonify

import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load(
    "../models/demand_forecasting_model.pkl"
)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    df = pd.DataFrame([data])

    prediction = model.predict(
        df
    )[0]

    return jsonify({
        "predicted_demand": round(
            float(prediction),
            2
        )
    })

if __name__ == "__main__":
    app.run(debug=True)
