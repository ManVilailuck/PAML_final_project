import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.title("🧪 Exploratory Data Analysis and Preprocessing")
st.write("This will take a few moments, please be patient! Thank you :)")

# Unified preprocessing and WordCloud generation
@st.cache_data
def get_eda_outputs():
    df = pd.read_csv("Reviews.csv")
    df = df[['Text', 'Score']].dropna()
    df = df[df['Score'] != 3]
    df['Label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    df['length'] = df['Text'].apply(lambda x: len(x.split()))

    # WordCloud text
    raw_pos = " ".join(df[df["Label"] == 1]["Text"])
    raw_neg = " ".join(df[df["Label"] == 0]["Text"])

    # WordClouds (uncleaned)
    wc_raw_pos = WordCloud(background_color='white').generate(raw_pos)
    wc_raw_neg = WordCloud(background_color='black').generate(raw_neg)

    # Cleaned WordClouds
    custom_stopwords = ENGLISH_STOP_WORDS.union({
        "br", "product", "amazon", "one", "use", "buy", "item", "would", "get", "like"
    })

    wc_clean_pos = WordCloud(stopwords=custom_stopwords, background_color='white').generate(raw_pos)
    wc_clean_neg = WordCloud(stopwords=custom_stopwords, background_color='black').generate(raw_neg)

    return df, wc_raw_pos, wc_raw_neg, wc_clean_pos, wc_clean_neg

# Load everything (cached)
df, wc_raw_pos, wc_raw_neg, wc_clean_pos, wc_clean_neg = get_eda_outputs()

# -----------------------------
# Visualizations

st.subheader("Sentiment Class Distribution")
fig1, ax1 = plt.subplots()
sns.countplot(x='Label', data=df, ax=ax1)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Negative (0)', 'Positive (1)'])
ax1.set_title("Class Distribution")
st.pyplot(fig1)

st.subheader("Review Length Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['length'], bins=50, kde=True, ax=ax2)
ax2.set_title("Review Lengths")
st.pyplot(fig2)

st.subheader("Review Length by Sentiment")
fig3, ax3 = plt.subplots()
sns.boxplot(x='Label', y='length', data=df, ax=ax3)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Negative (0)', 'Positive (1)'])
ax3.set_title("Length by Sentiment")
st.pyplot(fig3)

# -----------------------------
# WordClouds

st.subheader("Original WordClouds (No Stopwords Removed)")
st.image(wc_raw_pos.to_array(), caption="Raw Positive Reviews WordCloud")
st.image(wc_raw_neg.to_array(), caption="Raw Negative Reviews WordCloud")

st.subheader("Cleaned WordClouds (After Removing Stopwords)")
st.image(wc_clean_pos.to_array(), caption="Cleaned Positive Reviews WordCloud")
st.image(wc_clean_neg.to_array(), caption="Cleaned Negative Reviews WordCloud")
