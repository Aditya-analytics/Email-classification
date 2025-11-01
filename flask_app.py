from flask import Flask, request, jsonify
import joblib
import re
import numpy as np
import traceback

app = Flask(__name__)

# ------------------------------
# Load Model Safely
# ------------------------------
try:
    model = joblib.load("nb_spam_ham.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None


# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers/symbols
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text


# ------------------------------
# Prediction Logic
# ------------------------------
def predict_spam_api(email_text):
    if model is None:
        return "Error", 0.0, ""

    cleaned_text = clean_text(email_text)

    # Empty text edge case
    if not cleaned_text:
        return "Ham", 1.0, cleaned_text

    try:
        prediction = model.predict([cleaned_text])[0]
        prob = model.predict_proba([cleaned_text])[0]

        # Determine confidence
        if prediction.lower() == "spam":
            confidence = float(prob[1])
        else:
            confidence = float(prob[0])

        return prediction, confidence, cleaned_text

    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        return "Error", 0.0, cleaned_text


# ------------------------------
# API Endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "email_text" not in data:
            return jsonify({"error": "Invalid input. Missing 'email_text'."}), 400

        email_text = data["email_text"]
        prediction, confidence, cleaned_text = predict_spam_api(email_text)

        if prediction == "Error":
            return jsonify({"error": "An error occurred during prediction."}), 500

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "cleaned_text": cleaned_text
        })

    except Exception as e:
        print("❌ API Exception:", e)
        traceback.print_exc()
        return jsonify({"error": "Internal server error."}), 500


# ------------------------------
# Run Flask Server
# ------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
