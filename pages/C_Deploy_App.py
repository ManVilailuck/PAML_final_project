import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text

# UI
st.title("📬 Live Sentiment Prediction")
st.write("This will take a few moments, please be patient! Thank you :)")

# Custom stopwords (optional, used during vectorization)
all_stopwords = list(text.ENGLISH_STOP_WORDS.union({
    "br", "product", "amazon", "one", "use", "buy", "item", "would", "get", "like"
}))

# Make sure models are trained
if 'nb_model' not in st.session_state or 'logreg_model' not in st.session_state or 'vectorizer' not in st.session_state:
    st.error("⚠️ Please train the models in the 'Train/Test' page before making predictions.")
    st.stop()

# Load trained components from session state
nb_model = st.session_state.nb_model
logreg_model = st.session_state.logreg_model
vectorizer = st.session_state.vectorizer

# Text input
review_text = st.text_area("Enter a product review below:")

# Model choice
model_choice = st.selectbox("Choose a model", ["Naive Bayes", "Logistic Regression"])

# Predict button
if st.button("Predict Sentiment"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        X_input = vectorizer.transform([review_text])

        if model_choice == "Naive Bayes":
            prediction = nb_model.predict(X_input)[0]
        else:
            prediction = logreg_model.predict(X_input)[0]

        label = "Predicted as Positive 😊" if prediction == 1 else "Predicted as Negative 😞"
        st.success(f"**Predicted Sentiment:** {label}")
