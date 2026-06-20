from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/recommend/<property_id>")
def recommend(property_id):

    recommendations = [
        "Property-102",
        "Property-215",
        "Property-330"
    ]

    return jsonify({
        "property_id": property_id,
        "recommended_properties": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
