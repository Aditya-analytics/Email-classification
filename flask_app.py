from flask import Flask, request, jsonify
import joblib
import re
import numpy as np

app = Flask(__name__)

# ------------------------------
# Load Model
# ------------------------------
model = joblib.load("nb_spam_ham.pkl")

# ------------------------------
# Text Cleaning Function
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # remove numbers/symbols
    text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
    return text

# ------------------------------
# Spam Prediction Logic
# ------------------------------
def predict_spam_api(email_text):
    if model is None:
        return "Error", 0.0, "" # Return empty string for cleaned_text on model error

    # Clean the email text before prediction
    cleaned_text = clean_text(email_text)

    # Handle empty input after cleaning
    if not cleaned_text:
        return "Ham", 1.0, cleaned_text # Default to Ham if input is empty or just symbols

    # Predict label and probability
    try:
        prediction = model.predict([cleaned_text])[0]
        prob = model.predict_proba([cleaned_text])[0]
    except Exception as e:
        print(e)
        return "Error", 0.0, cleaned_text

    # Confidence logic: prob[1] is spam, prob[0] is ham
    confidence = prob[1] if prediction.lower() == "spam" else prob[0]

    return prediction, confidence, cleaned_text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'email_text' not in data:
        return jsonify({'error': 'Invalid input. Missing email_text.'}), 400

    email_text = data['email_text']
    prediction, confidence, cleaned_text = predict_spam_api(email_text)

    if prediction == "Error":
        return jsonify({'error': 'An error occurred during prediction.'}), 500

    return jsonify({
        'prediction': prediction,
        'confidence': confidence,
        'cleaned_text': cleaned_text
    })

if __name__ == '__main__':
    app.run(debug=True)
