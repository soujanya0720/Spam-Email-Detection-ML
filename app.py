from pathlib import Path
import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "spam_model.joblib"

st.set_page_config(page_title="Spam Email Detector", page_icon="📧")
st.title("📧 Spam Email Detection")
st.write("Enter an email message and the machine-learning model will predict whether it is spam or ham (not spam).")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.warning("The model has not been trained yet. Run `python train_model.py` in your terminal, then refresh this page.")
else:
    email_text = st.text_area("Paste your email message here:", height=180,
                              placeholder="Example: Congratulations! You have won a prize...")
    if st.button("Check Email", type="primary"):
        if not email_text.strip():
            st.error("Please enter an email message first.")
        else:
            prediction = model.predict([email_text])[0]
            probabilities = model.predict_proba([email_text])[0]
            classes = model.classes_
            confidence = float(probabilities[list(classes).index(prediction)])

            if prediction == "spam":
                st.error("⚠️ Prediction: SPAM")
                st.write("Be careful. Do not click suspicious links or share passwords, PINs, or OTPs.")
            else:
                st.success("✅ Prediction: HAM (Not Spam)")
                st.write("The model did not detect strong spam patterns in this message.")

            st.caption(f"Model confidence: {confidence:.1%}")

st.divider()
st.caption("Learning project: this is a demonstration model trained on a small sample dataset. It can make mistakes.")
