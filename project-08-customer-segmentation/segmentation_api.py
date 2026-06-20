from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route("/segments")
def segments():

    return jsonify({
        "VIP": 120,
        "Loyal": 340,
        "Regular": 510,
        "New": 220,
        "AtRisk": 90
    })

if __name__ == "__main__":
    app.run(debug=True)
