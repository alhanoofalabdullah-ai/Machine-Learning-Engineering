from flask import Flask
from flask import request
from flask import jsonify
import joblib

app = Flask(__name__)

model = joblib.load(
    "../models/sentiment_model.pkl"
)

vectorizer = joblib.load(
    "../models/vectorizer.pkl"
)

@app.route("/predict", methods=["POST"])
def predict():

    text = request.json["text"]

    transformed = vectorizer.transform(
        [text]
    )

    prediction = model.predict(
        transformed
    )[0]

    return jsonify({
        "sentiment": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
