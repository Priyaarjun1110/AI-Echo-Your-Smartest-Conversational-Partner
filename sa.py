import pandas as pd
import numpy as np
import re
import nltk
import joblib
import tensorflow as tf

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# ===============================
# Download NLTK Resources
# ===============================
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ===============================
# Set Random Seed
# ===============================
tf.random.set_seed(42)

# ===============================
# Text Cleaning Function
# ===============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    cleaned_words = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(cleaned_words)

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("chatgpt_style_reviews_dataset.xlsx - Sheet.csv")
df.dropna(subset=['review', 'rating'], inplace=True)

# Create Sentiment Labels
df['sentiment'] = df['rating'].apply(
    lambda x: 'Positive' if x >= 4 else ('Neutral' if x == 3 else 'Negative')
)

# Text Preprocessing
df['cleaned'] = df['review'].apply(clean_text)

# ===============================
# Encode Labels
# ===============================
le = LabelEncoder()
y_encoded = le.fit_transform(df['sentiment'])
class_names = le.classes_ # ['Negative', 'Neutral', 'Positive']

# ===============================
# Train Test Split
# ===============================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['cleaned'], y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ==========================================
# 5. Project Evaluation Metrics Function
# ==========================================
def display_evaluation_metrics(y_true, y_pred, y_probs, model_name, file_prefix):
    print(f"\n{'='*10} {model_name} Evaluation Metrics {'='*10}")
    
    # 1. Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # 2, 3, 4. Precision, Recall, F1-Score
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 5. Confusion Matrix Visual
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f"{file_prefix}_cm.png") # Saves the image
    plt.close()

    # 6. AUC-ROC Curve Plotting
    # Binarize labels for multiclass plotting
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    n_classes = 3
    
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'orange', 'green']
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'AUC-ROC Curve: {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f"{file_prefix}_roc.png") # Saves the image
    plt.close()
    
    print(f"Images saved: {file_prefix}_cm.png and {file_prefix}_roc.png")

# --- Inside your sa.py script, update the calls to the function ---

# For ML:
display_evaluation_metrics(y_test, ml_preds, ml_probs, "Logistic Regression", "ml")

# For DL:
display_evaluation_metrics(y_test, dl_preds, dl_probs, "LSTM", "dl")

# ==========================================
# MODEL 1 : Machine Learning (Logistic Regression)
# ==========================================
tfidf = TfidfVectorizer(max_features=2000)
X_train_ml = tfidf.fit_transform(X_train_text)
X_test_ml = tfidf.transform(X_test_text)

ml_model = LogisticRegression(max_iter=1000, C=0.5)
ml_model.fit(X_train_ml, y_train)

# Predictions and Probabilities
ml_preds = ml_model.predict(X_test_ml)
ml_probs = ml_model.predict_proba(X_test_ml)

# Display ML Metrics
display_evaluation_metrics(y_test, ml_preds, ml_probs, "Machine Learning (Logistic Regression)")

# ==========================================
# MODEL 2 : Deep Learning (LSTM)
# ==========================================
max_words = 2000
max_len = 50

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_text)

X_train_dl = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=max_len)
X_test_dl = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=max_len)

dl_model = Sequential([
    Embedding(input_dim=max_words, output_dim=64),
    LSTM(64, dropout=0.2),
    Dense(3, activation='softmax')
])

dl_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
dl_model.fit(X_train_dl, y_train, epochs=10, batch_size=32, verbose=0) # verbose=0 to keep logs clean

# Predictions and Probabilities
dl_probs = dl_model.predict(X_test_dl)
dl_preds = np.argmax(dl_probs, axis=1)

# Display DL Metrics
display_evaluation_metrics(y_test, dl_preds, dl_probs, "Deep Learning (LSTM)")

# ==========================================
# Model Comparison & Saving
# ==========================================
ml_acc = accuracy_score(y_test, ml_preds)
_, dl_acc = dl_model.evaluate(X_test_dl, y_test, verbose=0)

if ml_acc >= dl_acc:
    print("\nWINNER: Machine Learning")
    joblib.dump(ml_model, "best_model.pkl")
    joblib.dump(tfidf, "processor.pkl")
    joblib.dump("ML", "model_type.pkl")
else:
    print("\nWINNER: Deep Learning")
    dl_model.save("best_model.h5")
    joblib.dump(tokenizer, "processor.pkl")
    joblib.dump("DL", "model_type.pkl")

joblib.dump(le, "label_encoder.pkl")
print("Models saved successfully.")