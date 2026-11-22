import re
import html
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

# models folder is next to the app folder
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
EMB_MODEL = "all-MiniLM-L6-v2"

st.set_page_config(page_title="News Topic Classifier", page_icon="📰", layout="centered")


# Same cleaning used in the notebooks
def clean_text(text):
    text = re.sub(r"\s?#39;", "'", str(text)).replace("quot;", '"')
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\\", " ")
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Load the two trained models once
@st.cache_resource
def load_models():
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf_lr.joblib")
    emb_clf = joblib.load(f"{MODEL_DIR}/emb_mlp.joblib")
    encoder = SentenceTransformer(EMB_MODEL)
    return tfidf, emb_clf, encoder


# Get the probability of each topic from both models
def predict(text, tfidf, emb_clf, encoder):
    text = clean_text(text)
    p1 = tfidf.predict_proba([text])[0]
    emb = encoder.encode([text], normalize_embeddings=True)
    p2 = emb_clf.predict_proba(emb)[0]
    return pd.DataFrame({
        "TF-IDF + LR": pd.Series(p1, index=tfidf.classes_),
        "Embeddings + MLP": pd.Series(p2, index=emb_clf.classes_),
    })


st.title("📰 News Topic Classifier")
st.write("Type a short news headline and the models will guess the topic: "
         "**World, Sports, Business** or **Sci/Tech**.")

try:
    tfidf, emb_clf, encoder = load_models()
except FileNotFoundError:
    st.error("Models not found. Run the notebooks in the `notebooks` folder first.")
    st.stop()

text = st.text_input("Headline", placeholder="e.g. Stocks fall as inflation rises")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please write a headline first.")
    else:
        probs = predict(text, tfidf, emb_clf, encoder)
        c1, c2 = st.columns(2)
        for col, name in zip([c1, c2], probs.columns):
            col.metric(name, probs[name].idxmax(), f"{probs[name].max() * 100:.1f}% confidence")

        st.subheader("Probabilities")
        st.bar_chart(probs, stack=False)

        if probs["TF-IDF + LR"].idxmax() != probs["Embeddings + MLP"].idxmax():
            st.info("The two models do not agree on this headline.")
