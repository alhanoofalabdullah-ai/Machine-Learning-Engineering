from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/recommend/<user_id>")
def recommend(user_id):

    recommendations = [
        "Laptop Pro",
        "Wireless Mouse",
        "Mechanical Keyboard"
    ]

    return jsonify({
        "user_id": user_id,
        "recommendations": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
