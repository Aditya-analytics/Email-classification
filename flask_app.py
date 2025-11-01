from flask import Flask, request, jsonify, abort
import joblib
import re
import numpy as np

app = Flask(__name__)

# ------------------------------
# Load Model (with safety)
# ------------------------------
try:
    model = joblib.load("nb_spam_ham.pkl")
    print("✅ Model loaded successfully.")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text: str) -> str:
    """Cleans input text by removing symbols, numbers, and extra spaces."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ------------------------------
# Spam Prediction Logic
# ------------------------------
def predict_spam_api(email_text: str):
    if model is None:
        return "Error", 0.0, ""

    # Clean and validate text
    cleaned_text = clean_text(email_text)
    if not cleaned_text:
        return "Ham", 1.0, cleaned_text  # Empty or invalid text defaults to Ham

    try:
        prediction = model.predict([cleaned_text])[0]
        prob = model.predict_proba([cleaned_text])[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error", 0.0, cleaned_text

    # Confidence based on predicted label
    confidence = prob[1] if prediction.lower() == "spam" else prob[0]
    return prediction, confidence, cleaned_text

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    """Health check route."""
    return jsonify({"message": "📬 Spam Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if the email is Spam or Ham."""
    data = request.get_json()
    if not data or 'email_text' not in data:
        abort(400, description="Missing 'email_text' field in request body.")

    email_text = data['email_text']
    prediction, confidence, cleaned_text = predict_spam_api(email_text)

    if prediction == "Error":
        return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({
        'prediction': prediction,
        'confidence': round(float(confidence), 3),
        'cleaned_text': cleaned_text
    })

# ------------------------------
# Run Server
# ------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
