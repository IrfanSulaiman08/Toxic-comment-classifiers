# ====================================================
# train_and_predict.py
# Toxic Comment Classification (NLP + Linear SVM)
# Backend-only project (No Frontend / No Streamlit)
# ====================================================

import pandas as pd
import pickle
import streamlit as st

import os

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------
# STEP 1: Load Dataset
# ----------------------------------------------------
df = pd.read_csv(r"C:\Users\Irfan Sulaiman\OneDrive\Desktop\project ML\Data\toxic_comment_dataset_1000.csv.csv")

# ----------------------------------------------------
# STEP 2: Encode Labels (Toxic → 1, Non-Toxic → 0)
# ----------------------------------------------------
le = LabelEncoder()
df["label_encoded"] = le.fit_transform(df["label"])

# Features & Target
X = df["comment"]
y = df["label_encoded"]

# ----------------------------------------------------
# STEP 3: Train-Test Split
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# STEP 4: TF-IDF Vectorization
# ----------------------------------------------------
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.8
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ----------------------------------------------------
# STEP 5: Train Linear SVM Model
# ----------------------------------------------------
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# ----------------------------------------------------
# STEP 6: Evaluate Model
# ----------------------------------------------------
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------
# STEP 7: Save Model Files
# ----------------------------------------------------
os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/toxic_model.pkl", "wb"))
pickle.dump(tfidf, open("model/vectorizer.pkl", "wb"))
pickle.dump(le, open("model/label_encoder.pkl", "wb"))

print("\nModel saved successfully!")

# ----------------------------------------------------
# STEP 8: PREDICT NEW INPUT (USER INPUT)
# ----------------------------------------------------
while True:
    text = input("\nEnter a comment (or type 'exit'): ")

    if text.lower() == "exit":
        break

    text_vec = tfidf.transform([text])
    prediction = model.predict(text_vec)[0]
    label = le.inverse_transform([prediction])[0]

    print("Prediction:", label)
