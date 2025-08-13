from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, pickle

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load models
with open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "models", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)


# Serve front-end from templates
@app.route("/")
def home():
    return render_template("index.html")  # Flask will serve index.html from templates/


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    if not review:
        return jsonify({"error": "No review text provided"}), 400

    review_vector = vectorizer.transform([review])
    pred = model.predict(review_vector)[0]
    sentiment = (
        pred if isinstance(pred, str) else ("positive" if pred == 1 else "negative")
    )
    return jsonify({"sentiment": sentiment})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
