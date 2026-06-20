from flask import Flask
from flask import request
from flask import jsonify

import joblib

app = Flask(__name__)

model = joblib.load(
    "../models/document_classifier.pkl"
)

vectorizer = joblib.load(
    "../models/vectorizer.pkl"
)

@app.route("/classify", methods=["POST"])
def classify():

    document_text = request.json["text"]

    transformed = vectorizer.transform(
        [document_text]
    )

    prediction = model.predict(
        transformed
    )[0]

    return jsonify({
        "document_category": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
