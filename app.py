import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="AI Echo Sentiment Dashboard",
    layout="wide"
)

# Professional Seaborn Styling
sns.set_theme(style="darkgrid")

st.title("🤖 AI Echo - Sentiment Analysis Dashboard")

# -----------------------------------
# Download NLTK Resources
# -----------------------------------
@st.cache_resource
def download_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)

download_resources()

# -----------------------------------
# Load Dataset (Cached)
# -----------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("chatgpt_style_reviews_dataset.xlsx - Sheet.csv")
    return df

df = load_data()

# -----------------------------------
# Load Models (Cached)
# -----------------------------------
@st.cache_resource
def load_models():
    # Detect the winner from the training script
    model_type = joblib.load("model_type.pkl")
    processor = joblib.load("processor.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    if model_type == "ML":
        model = joblib.load("best_model.pkl")
    else:
        # For Deep Learning
        model = tf.keras.models.load_model("best_model.h5")

    return model, processor, label_encoder, model_type

model, processor, label_encoder, model_type = load_models()

# -----------------------------------
# Text Cleaning Function
# -----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_words)

# -----------------------------------
# Sentiment Prediction Function
# -----------------------------------
def predict_sentiment(text):
    cleaned = clean_text(text)
    if model_type == "ML":
        vector = processor.transform([cleaned])
        prediction = model.predict(vector)
    else:
        seq = processor.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=50)
        prediction = np.argmax(model.predict(padded), axis=1)
    
    sentiment = label_encoder.inverse_transform(prediction)
    return sentiment[0]

# -----------------------------------
# Tabs Layout
# -----------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 EDA Dashboard",
    "💡 Sentiment Insights",
    "🔮 Sentiment Prediction"
])

# ===================================
# TAB 1 : EDA DASHBOARD
# ===================================
with tab1:
    st.header("📊 Exploratory Data Analysis Dashboard")
    st.markdown("---")

    # --- ROW 1: Questions 1 & 2 ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 1. Distribution of Review Ratings")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x="rating", palette="viridis", ax=ax1)
        ax1.set_title("Total Count per Star Rating")
        st.pyplot(fig1)
        st.info("**Insight:** Understand overall sentiment — are users mostly happy (4-5) or frustrated (1-2)?")

    with col2:
        st.subheader("👍👎 2. Helpful Reviews (>10 Votes)")
        if "helpful_votes" in df.columns:
            helpful_threshold = 10
            helpful_count = len(df[df["helpful_votes"] > helpful_threshold])
            other_count = len(df) - helpful_count
            
            fig2, ax2 = plt.subplots()
            ax2.pie([helpful_count, other_count], labels=["Helpful", "Other"], autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=140)
            st.pyplot(fig2)
            st.info(f"**Insight:** {helpful_count} reviews have more than 10 helpful votes, showing high community value.")

    st.markdown("---")

    # --- ROW 2: Question 3 (Word Clouds) ---
    st.subheader("🧭 3. Most Common Keywords (Positive vs. Negative)")
    col3, col4 = st.columns(2)

    with col3:
        st.write("✅ **Positive (4–5 Stars)**")
        pos_text = " ".join(df[df["rating"] >= 4]["review"].astype(str))
        wc_pos = WordCloud(width=800, height=400, background_color="white", colormap="summer").generate(pos_text)
        fig3, ax3 = plt.subplots()
        ax3.imshow(wc_pos, interpolation="bilinear")
        ax3.axis("off")
        st.pyplot(fig3)

    with col4:
        st.write("😡 **Negative (1–2 Stars)**")
        neg_text = " ".join(df[df["rating"] <= 2]["review"].astype(str))
        wc_neg = WordCloud(width=800, height=400, background_color="white", colormap="autumn").generate(neg_text)
        fig4, ax4 = plt.subplots()
        ax4.imshow(wc_neg, interpolation="bilinear")
        ax4.axis("off")
        st.pyplot(fig4)
    st.info("**Insight:** Discover what users love (e.g., interface, quality) vs what they complain about (e.g., bugs, crashes).")

    st.markdown("---")

    # --- ROW 3: Questions 4 & 5 ---
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("📆 4. Average Rating Over Time")
        if "date" in df.columns:
            df_time = df.copy()
            df_time["date"] = pd.to_datetime(df_time["date"], errors="coerce")
            df_time = df_time.dropna(subset=["date"])
            trend = df_time.groupby(df_time["date"].dt.to_period("M"))["rating"].mean()
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            trend.plot(kind="line", marker='o', color='teal', ax=ax5)
            plt.xticks(rotation=45)
            st.pyplot(fig5)
            st.info("**Insight:** Track user satisfaction changes over weeks or months.")

    with col6:
        st.subheader("🌍 5. Ratings by User Location")
        if "location" in df.columns:
            loc_avg = df.groupby("location")["rating"].mean().sort_values(ascending=False).head(10)
            fig6, ax6 = plt.subplots(figsize=(8, 5))
            sns.barplot(x=loc_avg.values, y=loc_avg.index, palette="magma", ax=ax6)
            st.pyplot(fig6)
            st.info("**Insight:** Identify regional differences in satisfaction or experience.")

    st.markdown("---")

    # --- ROW 4: Questions 6 & 7 ---
    col7, col8 = st.columns(2)

    with col7:
        st.subheader("🧑‍💻 6. Platform Comparison (Web vs Mobile)")
        if "platform" in df.columns:
            plat_avg = df.groupby("platform")["rating"].mean().reset_index()
            fig7, ax7 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=plat_avg, x="platform", y="rating", palette="coolwarm", ax=ax7)
            st.pyplot(fig7)
            st.info("**Insight:** Helps product teams focus improvements on specific platforms.")

    with col8:
        st.subheader("✅❌ 7. Verified vs Non-Verified Satisfaction")
        if "verified_purchase" in df.columns:
            ver_avg = df.groupby("verified_purchase")["rating"].mean().reset_index()
            fig8, ax8 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=ver_avg, x="verified_purchase", y="rating", palette="Set2", ax=ax8)
            st.pyplot(fig8)
            st.info("**Insight:** Indicates whether loyal/paying users are happier with the product.")

    st.markdown("---")

    # --- ROW 5: Questions 8 & 9 ---
    col9, col10 = st.columns(2)

    with col9:
        st.subheader("🔠 8. Average Length of Reviews per Rating")
        df["review_len"] = df["review"].astype(str).apply(lambda x: len(x.split()))
        fig9, ax9 = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="rating", y="review_len", palette="pastel", ax=ax9)
        st.pyplot(fig9)
        st.info("**Insight:** Do users write longer reviews when they are unhappy (1-star) or very happy (5-star)?")

    with col10:
        st.subheader("💬 9. Most Mentioned Words in 1-Star Reviews")
        one_star_text = " ".join(df[df["rating"] == 1]["review"].astype(str))
        if one_star_text.strip():
            wc_one = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(one_star_text)
            fig10, ax10 = plt.subplots()
            ax10.imshow(wc_one, interpolation="bilinear")
            ax10.axis("off")
            st.pyplot(fig10)
        st.info("**Insight:** Spot recurring issues, fatal bugs, or complaints unique to 1-star ratings.")

    st.markdown("---")

    # --- ROW 6: Question 10 ---
    st.subheader("📱🧪 10. ChatGPT Version vs Average Rating")
    if "version" in df.columns:
        ver_rating = df.groupby("version")["rating"].mean().sort_values(ascending=False).reset_index()
        fig11, ax11 = plt.subplots(figsize=(12, 5))
        sns.barplot(data=ver_rating, x="version", y="rating", palette="Spectral", ax=ax11)
        st.pyplot(fig11)
        st.info("**Insight:** Evaluate if new updates (e.g., 5.0.3) improved satisfaction or caused regressions.")
# ===================================
# TAB 2 : SENTIMENT INSIGHTS
# ===================================
with tab2:
    st.header("💡 Deep Sentiment Analysis Insights")
    st.markdown("---")

    # --- STEP 1: PRE-CALCULATE SENTIMENTS FOR DATASET ---
    # We do this once to power all charts in this tab
    @st.cache_data
    def get_batch_sentiments(_df):
        temp_df = _df.copy()
        # Apply the prediction function to every row
        temp_df['predicted_sentiment'] = temp_df['review'].apply(predict_sentiment)
        # Calculate length for Q6
        temp_df['review_len'] = temp_df['review'].astype(str).apply(lambda x: len(x.split()))
        return temp_df

    with st.spinner("Analyzing dataset sentiment..."):
        sdf = get_batch_sentiments(df)

    # --- Q1: Overall Sentiment Proportions ---
    st.subheader("1. What is the overall sentiment of user reviews?")
    col_q1a, col_q1b = st.columns([1, 2])
    with col_q1a:
        sentiment_counts = sdf['predicted_sentiment'].value_counts()
        st.write(sentiment_counts)
    with col_q1b:
        fig_q1 = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                        color=sentiment_counts.index,
                        color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q1, width="stretch")
    st.info("→ **Insight:** This shows the model's classification breakdown. Proportions help identify the dominant user mood.")

    st.markdown("---")

    # --- Q2: Sentiment vs Rating ---
    st.subheader("2. How does sentiment vary by rating?")
    fig_q2 = px.density_heatmap(sdf, x="rating", y="predicted_sentiment", text_auto=True, 
                                 labels={'predicted_sentiment':'Model Prediction', 'rating':'User Rating'},
                                 color_continuous_scale="Viridis")
    st.plotly_chart(fig_q2, width="stretch")
    st.info("→ **Insight:** Do 1-star reviews always contain negative sentiment? This heatmap spots 'mismatches' (e.g., a 5-star review that the model thinks is negative).")

    st.markdown("---")

    # --- Q3: Keywords per Sentiment ---
    st.subheader("3. Keywords associated with each Sentiment Class")
    sent_choice = st.selectbox("Select Sentiment to view WordCloud:", ["Positive", "Neutral", "Negative"])
    sent_text = " ".join(sdf[sdf["predicted_sentiment"] == sent_choice]["review"].astype(str))
    
    # Use standard sizing and appropriate colors
    wc_color = {"Positive": "summer", "Neutral": "Greys", "Negative": "autumn"}[sent_choice]
    wc_q3 = WordCloud(width=800, height=400, background_color="white", colormap=wc_color).generate(sent_text)
    fig_q3, ax_q3 = plt.subplots()
    ax_q3.imshow(wc_q3, interpolation="bilinear")
    ax_q3.axis("off")
    st.pyplot(fig_q3)

    st.markdown("---")

    # --- Q4: Sentiment Over Time ---
    st.subheader("4. How has sentiment changed over time?")
    sdf['date'] = pd.to_datetime(sdf['date'], errors='coerce')
    sdf_time = sdf.dropna(subset=['date']).copy()
    sdf_time['month'] = sdf_time['date'].dt.to_period('M').astype(str)
    timeline = sdf_time.groupby(['month', 'predicted_sentiment']).size().reset_index(name='count')
    fig_q4 = px.line(timeline, x='month', y='count', color='predicted_sentiment', markers=True,
                     color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
    st.plotly_chart(fig_q4, width="stretch")
    st.info("→ **Insight:** Analyze trends by month to spot peaks in satisfaction (updates) or dissatisfaction (outages).")

    st.markdown("---")

    # --- Q5 & Q6 ---
    col_q5, col_q6 = st.columns(2)
    with col_q5:
        st.subheader("5. Verified Users Sentiment")
        fig_q5 = px.histogram(sdf, x="verified_purchase", color="predicted_sentiment", barmode="group",
                              color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q5, width="stretch")
    with col_q6:
        st.subheader("6. Review Length vs Sentiment")
        fig_q6 = px.box(sdf, x="predicted_sentiment", y="review_len", color="predicted_sentiment",
                        color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q6, width="stretch")
    st.info("→ **Insight:** Q5 checks if paying users are happier. Q6 checks if people write longer reviews when they are angry.")

    st.markdown("---")

    # --- Q7 & Q8 ---
    col_q7, col_q8 = st.columns(2)
    with col_q7:
        st.subheader("7. Sentiment by Location")
        loc_sent = sdf.groupby(['location', 'predicted_sentiment']).size().reset_index(name='count')
        # Show top 10 locations
        top_locs = sdf['location'].value_counts().nlargest(10).index
        loc_sent = loc_sent[loc_sent['location'].isin(top_locs)]
        fig_q7 = px.bar(loc_sent, x='location', y='count', color='predicted_sentiment',
                        color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q7, width="stretch")
    with col_q8:
        st.subheader("8. Sentiment Across Platforms")
        fig_q8 = px.histogram(sdf, x="platform", color="predicted_sentiment", barmode="group",
                              color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q8, width="stretch")

    st.markdown("---")

    # --- Q9 & Q10 ---
    col_q9, col_q10 = st.columns(2)
    with col_q9:
        st.subheader("9. Sentiment by Version")
        ver_sent = sdf.groupby(['version', 'predicted_sentiment']).size().reset_index(name='count')
        fig_q9 = px.bar(ver_sent, x='version', y='count', color='predicted_sentiment',
                        color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c'})
        st.plotly_chart(fig_q9, width="stretch")
    with col_q10:
        st.subheader("10. Common Negative Themes")
        neg_text_only = " ".join(sdf[sdf["predicted_sentiment"] == "Negative"]["review"].astype(str))
        if neg_text_only.strip():
            wc_q10 = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(neg_text_only)
            fig_q10, ax_q10 = plt.subplots()
            ax_q10.imshow(wc_q10, interpolation="bilinear")
            ax_q10.axis("off")
            st.pyplot(fig_q10)
        else:
            st.write("Not enough negative data.")
    st.info("→ **Insight:** Identify recurring pain points and issues that caused version regressions.")

# ===================================
# TAB 3 : SENTIMENT PREDICTION
# ===================================
with tab3:
    st.header("🔮 AI Sentiment Prediction")
    st.info(f"The system is currently utilizing the **{model_type}** model for analysis.")

    review_text = st.text_area("Type a user review here to test the model:", height=150)

    if st.button("Analyze Sentiment"):
        if review_text.strip() != "":
            with st.spinner('Analyzing...'):
                sentiment = predict_sentiment(review_text)
                
                if sentiment == "Positive":
                    st.balloons()
                    st.success(f"### Predicted Sentiment: {sentiment} 😊")
                elif sentiment == "Negative":
                    st.error(f"### Predicted Sentiment: {sentiment} 😡")
                else:
                    st.warning(f"### Predicted Sentiment: {sentiment} 😐")
        else:
            st.warning("Please enter text before clicking Predict.")