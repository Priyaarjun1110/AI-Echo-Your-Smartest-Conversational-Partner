# simple_app.py — AI Echo: Sentiment Analysis (Simplified)

import re, string
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- Ensure NLTK data ---
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)

def label_sentiment(rating):
    if rating <= 2: return "Negative"
    elif rating == 3: return "Neutral"
    else: return "Positive"

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Echo (Simple)", layout="wide")
st.title("🗣️ AI Echo: Sentiment Analysis (Simplified)")

uploaded = st.file_uploader("Upload CSV dataset", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} rows!")

    # Prepare
    if "rating" not in df.columns or "review" not in df.columns:
        st.error("Dataset must contain 'review' and 'rating' columns.")
        st.stop()

    df["cleaned_review"] = df["review"].apply(clean_text)
    df["sentiment"] = df["rating"].apply(label_sentiment)

    tab1, tab2, tab3 = st.tabs(["EDA", "Models", "Predict"])

    # --- EDA ---

    with tab1:
        st.subheader("Exploratory Data Analysis (10 Key Questions)")

        # 1. Overall sentiment proportions
        st.markdown("**1️⃣ Overall Sentiment of Reviews**")
        fig, ax = plt.subplots()
        sns.countplot(x="sentiment", data=df, order=["Negative","Neutral","Positive"], ax=ax)
        st.pyplot(fig)
        st.write(df["sentiment"].value_counts(normalize=True) * 100)

        # 2. Sentiment vs Rating
        st.markdown("**2️⃣ Sentiment vs Rating**")
        cross = pd.crosstab(df["rating"], df["sentiment"], normalize="index") * 100
        st.dataframe(cross.style.background_gradient(cmap="Blues"))

        # 3. Keywords/phrases per sentiment
        st.markdown("**3️⃣ Keywords per Sentiment (WordClouds)**")
        from wordcloud import WordCloud
        colA, colB, colC = st.columns(3)
        for sentiment, col in zip(["Negative", "Neutral", "Positive"], [colA, colB, colC]):
            text = " ".join(df.loc[df["sentiment"]==sentiment, "cleaned_review"])
            if text.strip():
                wc = WordCloud(width=400, height=200, background_color="white").generate(text)
                with col:
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
                    ax.set_title(sentiment)
                    st.pyplot(fig)

        # 4. Sentiment over time
        st.markdown("**4️⃣ Sentiment Trends Over Time**")
        if "date" in df.columns:
            ts = df.dropna(subset=["date"]).assign(
                month=lambda d: pd.to_datetime(d["date"]).dt.to_period("M").dt.to_timestamp()
            )
            trend = ts.groupby(["month", "sentiment"]).size().reset_index(name="count")

            fig, ax = plt.subplots()
            sns.lineplot(data=trend, x="month", y="count", hue="sentiment", ax=ax)
            ax.set_title("Sentiment Trend Over Time")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # 5. Verified vs Non-verified sentiment
        st.markdown("**5️⃣ Verified vs Non-verified**")
        if "verified_purchase" in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x="verified_purchase", hue="sentiment", ax=ax)
            ax.set_title("Sentiment by Verified Purchase")
            st.pyplot(fig)

        # 6. Review length vs sentiment
        st.markdown("**6️⃣ Review Length vs Sentiment**")
        if "review_length" not in df.columns:
            df["review_length"] = df["review"].astype(str).str.len()
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="sentiment", y="review_length", ax=ax)
        ax.set_title("Review Length by Sentiment")
        st.pyplot(fig)

        # 7. Location-based sentiment
        st.markdown("**7️⃣ Sentiment by Location**")
        if "location" in df.columns:
            loc = df.groupby("location")["rating"].mean().sort_values(ascending=False).head(10)
            fig, ax = plt.subplots()
            loc.plot(kind="barh", ax=ax)
            ax.set_title("Top 10 Locations by Avg Rating")
            st.pyplot(fig)

        # 8. Sentiment across platforms
        st.markdown("**8️⃣ Sentiment by Platform**")
        if "platform" in df.columns:
            plat = df.groupby("platform")["rating"].mean().reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=plat, x="platform", y="rating", ax=ax)
            ax.set_title("Avg Rating by Platform")
            st.pyplot(fig)

        # 9. Sentiment by ChatGPT version
        st.markdown("**9️⃣ Sentiment by Version**")
        if "version" in df.columns:
            ver = df.groupby("version")["rating"].mean().reset_index()
            fig, ax = plt.subplots()
            sns.barplot(data=ver, x="version", y="rating", ax=ax)
            ax.set_title("Average Rating by Version")
            ax.tick_params(axis="x", rotation=45)
            st.pyplot(fig)

        # 10. Common negative feedback themes
        st.markdown("**🔟 Common Negative Feedback Themes**")
        neg = df[df["sentiment"]=="Negative"]
        words = " ".join(neg["cleaned_review"]).split()
        from collections import Counter
        common = Counter(words).most_common(20)
        if common:
            w, c = zip(*common)
            fig, ax = plt.subplots()
            sns.barplot(x=list(c), y=list(w), ax=ax)
            ax.set_title("Top Words in Negative Reviews")
            st.pyplot(fig)
        else:
            st.info("No negative reviews available.")

    # --- Models ---
    with tab2:
        st.subheader("Train Logistic Regression Model")
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_review"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
        )
        vec = TfidfVectorizer()
        Xtr, Xte = vec.fit_transform(X_train), vec.transform(X_test)
        clf = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred, labels=["Negative","Neutral","Positive"])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Neg","Neu","Pos"], yticklabels=["Neg","Neu","Pos"], ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    # --- Predict ---
    with tab3:
        st.subheader("Predict Sentiment")
        user_text = st.text_area("Enter a review:")
        if st.button("Predict"):
            cleaned = clean_text(user_text)
            pred = clf.predict(vec.transform([cleaned]))[0]
            st.success(f"Predicted Sentiment: **{pred}**")

else:
    st.info("Please upload a CSV file with at least 'review' and 'rating' columns.")
