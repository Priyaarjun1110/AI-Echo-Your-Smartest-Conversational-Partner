# 🤖 AI Echo: Sentiment Analysis

## 📌 Project Overview
AI Echo is a **Natural Language Processing (NLP) based sentiment analysis project** that analyzes user reviews of a ChatGPT-style application.  
The project classifies reviews into **Positive, Neutral, or Negative sentiments**, providing valuable insights into customer satisfaction, common issues, and opportunities for improvement.

---

## 🚀 Objectives
- Classify customer reviews into **Positive, Neutral, Negative**.
- Extract **keywords and patterns** influencing sentiment.
- Track **sentiment trends** over time and across platforms.
- Provide actionable insights for **product improvement**.

---

## 🛠️ Tech Stack
- **Programming Language**: Python  
- **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, NLTK, Scikit-learn  
- **Models**: Naïve Bayes, Logistic Regression, Random Forest, LSTMs, Transformers  
- **Deployment**: Streamlit (Dashboard), AWS (Optional)

---

## 📂 Dataset
Dataset: [ChatGPT-Style Reviews Dataset](https://docs.google.com/spreadsheets/d/1eyPDJj8ttd8t-o6JVT4txCbvJ9DtcF-U/edit?gid=1201624046#gid=1201624046)

### Dataset Features:
- **date** → When the review was submitted  
- **title** → Short headline of the review  
- **review** → Full user feedback  
- **rating** → Numerical score (1–5 stars)  
- **username** → Reviewer identity  
- **helpful_votes** → Number of helpful votes  
- **review_length** → Length of review text  
- **platform** → Web / Mobile  
- **language** → Language of review  
- **location** → Country of reviewer  
- **version** → App version reviewed  
- **verified_purchase** → Whether reviewer is a verified subscriber  

---

## 🔎 Approach
1. **Data Preprocessing**  
   - Cleaning, tokenization, lemmatization  
   - Stopword & punctuation removal  
   - Handling missing values  
   - Normalization (lowercasing, stemming)  

2. **Exploratory Data Analysis (EDA)**  
   - Sentiment distribution  
   - Word clouds, histograms  
   - Trends by location, platform, rating  

3. **Modeling**  
   - Text vectorization (TF-IDF, embeddings, BERT)  
   - Classification using ML/DL models  
   - Performance evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC  

4. **Deployment & Visualization**  
   - Streamlit dashboard for sentiment insights  
   - Visualizations: word clouds, line charts, bar plots, heatmaps  

---

## 📊 Key Insights & Results
- **Sentiment Distribution** → Breakdown of Positive, Neutral, Negative reviews  
- **Feature Importance** → Words influencing sentiment classification  
- **Model Performance** → Accuracy, F1-score, ROC-AUC comparisons  
- **Recommendations** → Product improvements from analysis  

---

## 📈 Evaluation Metrics
- Accuracy  
- Precision & Recall  
- F1-Score  
- Confusion Matrix  
- ROC-AUC Curve  

---

## 🎯 Business Use Cases
- Customer Feedback Analysis  
- Brand Reputation Monitoring  
- Feature Enhancement & Prioritization  
- Automated Customer Support  
- Marketing Strategy Optimization  

---

## 📌 Deliverables 
- ✅ EDA Report with Visualizations  
- ✅ Trained ML/DL Sentiment Classification Model  
- ✅ Streamlit Dashboard for insights  
- ✅ Model Performance Report  


---

## 📌 Project Structure
```
AI-Echo-Sentiment-Analysis/
│── app.py                 # Streamlit Dashboard
│── train_model.py         # Training Script
│── data_prep.py           # Data Preprocessing
│── AI_Echo_Sentiment_Project.ipynb  # Jupyter Notebook (EDA + Training)
│── artifacts/             # Trained Models & Outputs
│── configs/               # Config files
│── README.md              # Project Documentation
│── requirements.txt       # Dependencies
```


