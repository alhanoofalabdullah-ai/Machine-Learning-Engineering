from flask import Flask
from flask import request
from flask import jsonify
import joblib

app = Flask(__name__)

model = joblib.load(
    "../models/resume_classifier.pkl"
)

vectorizer = joblib.load(
    "../models/vectorizer.pkl"
)

@app.route("/screen", methods=["POST"])
def screen_candidate():

    resume_text = request.json["resume"]

    vectorized = vectorizer.transform(
        [resume_text]
    )

    result = model.predict(
        vectorized
    )[0]

    return jsonify({
        "recommendation": int(result)
    })

if __name__ == "__main__":
    app.run(debug=True)
