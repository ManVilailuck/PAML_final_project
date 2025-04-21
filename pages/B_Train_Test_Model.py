
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

st.write("This will take a few moments, please be patient! Thank you :)")

# Stopwords list
all_stopwords = list(text.ENGLISH_STOP_WORDS.union({"br", "product", "amazon", "one", "use", "buy", "item", "would", "get", "like"}))

# Naive Bayes
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_log_prior = {}
        self.feature_log_prob = {}
        for c in self.classes:
            X_c = X[y == c]
            self.class_log_prior[c] = np.log(X_c.shape[0] / X.shape[0])
            word_counts = X_c.sum(axis=0) + 1
            total_words = word_counts.sum()
            self.feature_log_prob[c] = np.log(word_counts / total_words)
    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            row = X[i]
            log_probs = {}
            for c in self.classes:
                log_prob = self.class_log_prior[c] + row.dot(self.feature_log_prob[c].T).item()
                log_probs[c] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

# Logistic Regression
class LogisticRegressionScratch:
    def __init__(self, lr=0.1, epochs=1):
        self.lr = lr
        self.epochs = epochs
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        X = X.toarray()
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            dw = np.dot(X.T, (y_pred - y)) / y.size
            db = np.sum(y_pred - y) / y.size
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):
        X = X.toarray()
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

@st.cache_resource
# Train function
def train_models():
    df = pd.read_csv("Reviews.csv")
    df = df[['Text', 'Score']].dropna()
    df = df[df['Score'] != 3]
    df['Label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

    vectorizer = CountVectorizer(stop_words=all_stopwords, max_features=5000)
    X = vectorizer.fit_transform(df['Text'])
    y = df['Label'].values

    nb_model = NaiveBayesClassifier()
    nb_model.fit(X, y)

    logreg_model = LogisticRegressionScratch()
    logreg_model.fit(X, y)

    return nb_model, logreg_model, vectorizer, X, y


st.title("📊 Evaluate Model Performance")

nb_model, logreg_model, vectorizer, X, y = train_models()

# Split data
@st.cache_data
def evaluate_models(_nb_model, _logreg_model, _vectorizer, _X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    nb_pred = _nb_model.predict(X_test)
    logreg_pred = _logreg_model.predict(X_test)

    from sklearn.metrics import accuracy_score, precision_score, recall_score

    return pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression"],
        "Accuracy": [
            accuracy_score(y_test, nb_pred),
            accuracy_score(y_test, logreg_pred)
        ],
        "Precision": [
            precision_score(y_test, nb_pred),
            precision_score(y_test, logreg_pred)
        ],
        "Recall": [
            recall_score(y_test, nb_pred),
            recall_score(y_test, logreg_pred)
        ]
    })

results = evaluate_models(nb_model, logreg_model, vectorizer, X, y)

st.dataframe(results.style.format({
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}"
}))

nb_model, logreg_model, vectorizer, X, y = train_models()

# Store in session
st.session_state.nb_model = nb_model
st.session_state.logreg_model = logreg_model
st.session_state.vectorizer = vectorizer