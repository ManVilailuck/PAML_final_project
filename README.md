# Practical Applications in Machine Learning Final Project

# Digesting Text: Predicting Amazon Review Sentiment with Naive Bayes and Logistic

This Streamlit web app presents our final project for INFO 5368: Practical Applications in Machine Learning. We built an end-to-end text classification system that analyzes Amazon product reviews to predict whether the sentiment is positive or negative.

## MUST download; file too big to upload onto Git
Dataset
This project uses the Amazon Fine Food Reviews dataset from Kaggle:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
We filtered out neutral (3-star) reviews and labeled scores of 1-2 as negative and 4-5 as positive.

We implemented and compared Naive Bayes and Logistic Regression classifiers (from scratch) to classify the sentiment of Amazon food product reviews.

The goal was to explore the strengths of simple machine learning models in text classification, visualize key patterns in the data, and deploy an interactive system that lets users test live predictions.

App Sections
A: Explore & Preprocess Data
• View sentiment class distribution
• Plot review lengths
• Generate word clouds (original and cleaned)

B: Train & Test Models
• Train Naive Bayes and Logistic Regression
• Evaluate models using accuracy, precision, and recall
• Store trained models in session for use in prediction

C: Deploy App
• Input custom review text
• Choose a model
• Predict and display sentiment result

Run the application:
```
streamlit run streamlit_app.py
```
*** Make sure all necessary libraries are installed ***

You can install the dependencies with:

pip install -r requirements.txt

Or install them manually:

streamlit

pandas

numpy

matplotlib

seaborn

scikit-learn

wordcloud
