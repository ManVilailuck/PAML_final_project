import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text  # for default stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
# Downloaded from Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

# Don't forget to use unique path
df = pd.read_csv('Reviews.csv')
# Drop unnecessary columns
df = df[['Text', 'Score']]

# Drop nulls + duplicates
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Filter only 1,2,4,5 star reviews, drop 3s
df = df[df['Score'] != 3]
df['Label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)  
# 1 = positive, 0 = negative

df = df[['Text', 'Label']]
df.reset_index(drop=True, inplace=True)

df.head()

# Limit to top 5000 words for speed
vectorizer = CountVectorizer(stop_words='english', max_features=5000)

# Transform text to BoW features
X = vectorizer.fit_transform(df['Text'])
y = df['Label'].values

X.shape # Should be something like (n_samples, 5000)

sns.countplot(x='Label', data=df)
plt.title("Sentiment Class Distribution")
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

df['length'] = df['Text'].apply(lambda x: len(x.split()))

sns.histplot(df['length'], bins=50, kde=True)
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

print(df.columns)
print(df.head())

sns.boxplot(x='Label', y='length', data=df)
plt.title("Review Length by Sentiment Class")
plt.xticks([0, 1], ['Negative (0)', 'Positive (1)'])
plt.xlabel("Sentiment")
plt.ylabel("Review Length (# words)")
plt.show()

pos_text = ' '.join(df[df['Label'] == 1]['Text'])
neg_text = ' '.join(df[df['Label'] == 0]['Text'])

wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
wordcloud_neg = WordCloud(width=800, height=400, background_color='black').generate(neg_text)

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title("Positive Reviews")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title("Negative Reviews")
plt.axis('off')
plt.show()

custom_stopwords = set([
    "br", "product", "amazon", "one", "use", "buy", "item", "would", "get", "like"
])

default_stopwords = text.ENGLISH_STOP_WORDS
all_stopwords = default_stopwords.union(custom_stopwords)

vectorizer = CountVectorizer(stop_words=all_stopwords, max_features=5000)

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_log_prior = {}
        self.feature_log_prob = {}

        # Convert to dense matrix if sparse
        X = X.toarray()

        for c in self.classes:
            X_c = X[y == c]
            # Prior probability log(P(class))
            self.class_log_prior[c] = np.log(X_c.shape[0] / X.shape[0])

            # Word counts for class c + Laplace smoothing
            word_counts = X_c.sum(axis=0) + 1
            total_words = word_counts.sum()
            self.feature_log_prob[c] = np.log(word_counts / total_words)

    def predict(self, X):
        X = X.toarray()
        predictions = []
        for x in X:
            log_probs = {}
            for c in self.classes:
                log_prob = self.class_log_prior[c] + np.sum(x * self.feature_log_prob[c])
                log_probs[c] = log_prob
            predictions.append(max(log_probs, key=log_probs.get))
        return np.array(predictions)

nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=20):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = X.toarray()  # Assuming sparse input
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

epochs_list = [10, 30, 60]

for ep in epochs_list:
    print(f"\n--- Training Logistic Regression with {ep} epochs ---")
    
    nb = LogisticRegression(lr=0.1, epochs=ep)
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")

model_results = {
    "Naive Bayes": {
        "Accuracy": 0.8994,
        "Precision": 0.9458,
        "Recall": 0.9341
    },
    "Logistic Regression (60 epochs)": {
        "Accuracy": 0.8422,
        "Precision": 0.8422,
        "Recall": 1.0000
    }
}

results_df = pd.DataFrame(model_results).T  # Transpose so models are rows
results_df = results_df.round(4)
results_df

import pickle

# Save Naive Bayes
with open("nb_model.pkl", "wb") as f:
    pickle.dump(nb, f)

# Save LogReg
with open("logreg_model.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

# Save Vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)