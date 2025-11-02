import streamlit as st
import time
import joblib
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import FunctionTransformer

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
def predict_spam(email_text):
    if model is None:
        return "Error", 0.0, "" # Return empty string for cleaned_text on model error

    time.sleep(0.5)  # simulate processing time

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
        st.error(f"An error occurred during prediction: {e}")
        return "Error", 0.0, cleaned_text

    # Confidence logic: prob[1] is spam, prob[0] is ham
    confidence = prob[1] if prediction.lower() == "spam" else prob[0]

    return prediction, confidence, cleaned_text

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------
# --- NEW CUSTOM CSS ---
# ------------------------------
st.markdown("""
<style>
/* --- Main App Background --- */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f5f7fa 0%, #e0e8f0 100%);
}

/* --- Sidebar Styling --- */
[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    /* Replaced border with a softer shadow */
    box-shadow: 2px 0px 15px rgba(0, 0, 0, 0.05);
}
[data-testid="stSidebar"] [data-testid="stHeader"] {
    font-weight: 600;
    color: #1E3A8A;
}

/* --- Main Title --- */
[data-testid="stAppViewContainer"] [data-testid="stHeading"] h1 {
    font-family: 'Arial', sans-serif;
    font-weight: 700;
    color: #1E3A8A; /* Dark Blue */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.05);
}

/* --- Text Area --- */
[data-testid="stTextArea"] textarea {
    border: 1px solid #B0BEC5;
    border-radius: 10px;
    font-size: 1.05rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    transition: border-color 0.3s, box-shadow 0.3s;
}
/* --- NEW: Text Area Hover --- */
[data-testid="stTextArea"] textarea:hover {
    border-color: #2563EB;
    box-shadow: 0 4px 8px rgba(37, 99, 235, 0.1);
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #1E3A8A; /* Dark Blue on focus */
    box-shadow: 0 0 10px rgba(30,58,138,0.2);
    outline: none;
}

/* --- Classify Button --- */
[data-testid="stButton"] button {
    font-weight: 600;
    border-radius: 10px;
    border: none;
    padding: 10px 24px;
    color: white;
    background: linear-gradient(45deg, #2563EB 0%, #1E3A8A 100%);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
    transition: all 0.3s ease;
    transform: scale(1);
}
[data-testid="stButton"] button:hover {
    background: linear-gradient(45deg, #1E3A8A 0%, #2563EB 100%);
    box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
    transform: scale(1.03);
}
[data-testid="stButton"] button:active {
    transform: scale(0.98);
    box-shadow: 0 2px 10px rgba(30, 58, 138, 0.3);
}

/* --- Result Cards --- */
.result-card-spam {
    border: 1px solid #FFCDD2; /* Lighter red border */
    border-radius: 12px;
    padding: 20px 20px 10px 20px;
    background-color: #FFFDFD;
    margin-top: 20px;
    box-shadow: 0 10px 25px rgba(211, 47, 47, 0.1); /* Red shadow */
    transition: all 0.3s ease-in-out; /* --- NEW: Transition --- */
}
/* --- NEW: Card Hover Effect --- */
.result-card-spam:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 30px rgba(211, 47, 47, 0.15);
}

.result-card-ham {
    border: 1px solid #C8E6C9; /* Lighter green border */
    border-radius: 12px;
    padding: 20px 20px 10px 20px;
    background-color: #FDFFFD;
    margin-top: 20px;
    box-shadow: 0 10px 25px rgba(0, 121, 107, 0.1); /* Green shadow */
    transition: all 0.3s ease-in-out; /* --- NEW: Transition --- */
}
/* --- NEW: Card Hover Effect --- */
.result-card-ham:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 30px rgba(0, 121, 107, 0.15);
}


/* --- Result Icon Animation (Tweaked) --- */
@keyframes bounce {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-8px); }
}
.result-icon {
    font-size: 3rem;
    padding: 10px 0;
    animation: bounce 1.2s ease-in-out 1;
}

/* --- NEW: Expander Styling --- */
[data-testid="stExpander"] {
    border: none;
    background-color: rgba(0,0,0,0.02);
    border-radius: 8px;
    transition: background-color 0.3s ease;
}
[data-testid="stExpander"] summary {
    font-weight: 500;
    padding: 5px 0;
    border-radius: 8px;
    transition: background-color 0.3s ease;
}
[data-testid="stExpander"] summary:hover {
    background-color: rgba(0,0,0,0.04);
}

/* --- Footer --- */
.footer {
    text-align: center;
    color: #455A64;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    
    st.header("About This App")
    st.info(
        "This application uses a Naive Bayes model "
        "to classify emails as 'Spam' or 'Ham' (Not Spam). "
        "Paste an email into the text box to get a prediction."
    )
    st.markdown("---")
    st.subheader("Model Details")
    st.markdown(
        """
        - **Model:** Multinomial Naive Bayes
        - **Features:** TF-IDF Vectorization
        - **Dataset:** SMS Spam Collection
        """
    )
    st.markdown("---")
    st.markdown("Created with [Streamlit](https://streamlit.io)")

# ------------------------------
# Main Application
# ------------------------------
st.title("📧 Classy Email Spam Detector")
st.markdown("Paste the full content of an email below to check if it's spam or not.")

email_input = st.text_area(
    "Enter Email Text:",
    height=250,
    placeholder="Dear User, you have won a prize worth $1000...",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1.2, 1, 1.2])

# --- Button Column ---
# We ONLY define the button here
with col2:
    classify_button = st.button("Classify Email", use_container_width=True, type="primary")

# --- Result Display Logic ---
# This is now OUTSIDE col2, so it will use the main container width
if classify_button:
    if not email_input:
        # Show the warning in col2, under the button
        with col2:
            st.warning("Please paste an email into the text box above.", icon="⚠️")
    elif model is None:
         # Show the model error in col2, under the button
         with col2:
             st.error("Model is not loaded. Cannot classify.")
    else:
        # The spinner and results will appear in the main container
        with st.spinner("🔍 Scanning for threats..."):
            prediction, confidence, cleaned_text = predict_spam(email_input)

        # --- NEW Improved Result Display (Full Width) ---
        if prediction.lower() == "spam":
            with st.container():
                st.markdown('<div class="result-card-spam">', unsafe_allow_html=True)
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown("<div class='result-icon'>🚨</div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown("<h3 style='color: #D32F2F; margin-top: 0; margin-bottom: 0.5rem;'>Classification: SPAM</h3>", unsafe_allow_html=True)
                    st.metric(label="Spam Confidence", value=f"{confidence * 100:.2f}%")
                
                st.markdown("---")
                with st.expander("See Analysis Details"):
                    st.markdown(
                        "**Analysis:** The model is highly confident this is spam based on its learned word patterns."
                    )
                    st.markdown("**Cleaned Text for Analysis:**")
                    st.code(cleaned_text, language="text")
                st.markdown('</div>', unsafe_allow_html=True)

        elif prediction.lower() == "ham":
            with st.container():
                st.markdown('<div class="result-card-ham">', unsafe_allow_html=True)
                cols = st.columns([1, 4])
                with cols[0]:
                    st.markdown("<div class='result-icon'>🛡️</div>", unsafe_allow_html=True)
                with cols[1]:
                    st.markdown("<h3 style='color: #00796B; margin-top: 0; margin-bottom: 0.5rem;'>Classification: HAM (Not Spam)</h3>", unsafe_allow_html=True)
                    st.metric(label="Ham Confidence", value=f"{confidence * 100:.2f}%")
                
                st.markdown("---")
                with st.expander("See Analysis Details"):
                    st.markdown(
                        "**Analysis:** The message appears to be legitimate with a low spam probability score."
                    )
                    st.markdown("**Cleaned Text for Analysis:**")
                    st.code(cleaned_text, language="text")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Handle the "Error" case from predict_spam
            st.error("An error occurred during classification. Please check the logs.")


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2025 Spam Detector Inc.</div>",
    unsafe_allow_html=True
)


