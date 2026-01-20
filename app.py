import streamlit as st
import pickle

# Load model and tools
with open("model/toxic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# ---------- UI PART ----------
st.set_page_config(page_title="Toxic Comment Classifier")
st.title("🛑 Toxic Comment Detector")

user_text = st.text_area("Enter a comment")

if st.button("Predict"):
    if user_text.strip() == "":
        st.warning("Please enter some text")
    else:
        # Vectorize
        text_vec = tfidf.transform([user_text])

        # Predict
        pred = model.predict(text_vec)

        # Decode label
        label = le.inverse_transform(pred)[0]

        # Show result
        if label.lower() == "toxic":
            st.error("🚨 Toxic Comment")
        else:
            st.success("✅ Non-Toxic Comment")