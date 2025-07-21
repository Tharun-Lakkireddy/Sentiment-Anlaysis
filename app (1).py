import streamlit as st
import joblib
import re
import base64

# ------------------------
# 1. Load model and vectorizer
# ------------------------
# These should match your training setup paths and file names
model = joblib.load(r"C:\Users\Tharun\Sentiment analysis\project\logistic_regression_model.pkl")
vectorizer = joblib.load(r"C:\Users\Tharun\Sentiment analysis\project\tfidf_vectorizer.pkl")

# ------------------------
# 2. Clean user input text
# ------------------------
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

# ------------------------
# 3. Set background image using base64 encoding
# ------------------------
def set_bg(image_path):
    # Open the image and convert it to base64 format
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    # Inject custom CSS to set it as Streamlit background
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/avif;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        font-family: 'Arial', sans-serif;
    }}
/* Entire app text set to black */
    .stApp, .stApp * {{
        color: #000000 !important;
    }}
    h1 {{
        text-align: center;
        font-size: 48px;
        
    }}

    p {{
        text-align: center;
        font-size: 20px;
       
    }}

    .stTextArea {{
        margin-left: auto;
        margin-right: auto;
        width: 80% !important;
    }}

    .stTextArea label {{
        font-weight: bold;
        font-size: 18px;
    }}

    .stTextArea textarea {{
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #000080 !important;
        font-size: 16px !important;
        padding: 10px;
    }}

    div.stButton > button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s ease;
    }}

    div.stButton > button:hover {{
        background-color: #45a049;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
# Call the function with your local image
set_bg(r"C:\Users\Tharun\Sentiment analysis\project\image.avif")

# ------------------------
# 4. Display app title and instructions
# ------------------------
st.markdown("<h1>Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter a text to predict its sentiment (Positive / Negative / Neutral)</p>", unsafe_allow_html=True)

# ------------------------
# 5. User text input
# ------------------------
user_input = st.text_area("Enter your text here:", height=150)

# ------------------------
# 6. Predict and display sentiment
# ------------------------
if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"🎯 Predicted Sentiment: **{prediction.capitalize()}**")
