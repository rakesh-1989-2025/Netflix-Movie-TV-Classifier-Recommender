import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("netflix_titles.csv")

df['text'] = df['title'].astype(str) + ' ' + df['listed_in'].astype(str) + ' ' + df['description'].astype(str)

model = joblib.load("model.pkl")

st.title("🎬 Netflix Movie/TV Classifier & Recommender")

input_text = st.text_input("Enter show title, genres, or description: ")

if st.button("Predict Type"):
    pred = model.predict([input_text])[0]
    st.write("Prediction:", "Movie" if pred==1 else "TV Show")

st.header("Content-Based Recommendations")
title_pick = st.selectbox("Pick a title:", df['title'].dropna().unique())

if st.button("Recommend"):
    tfidf = TfidfVectorizer(max_features=8000)
    matrix = tfidf.fit_transform(df['text'])
    idx = df[df['title']==title_pick].index[0]
    sim = cosine_similarity(matrix[idx], matrix).flatten()
    top = sim.argsort()[-4:-1][::-1]
    st.write(df.iloc[top][['title','type','listed_in']])
