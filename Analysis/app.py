import streamlit as st
from inference import load_model, predict
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

tokenizer, model = load_model()
st.set_page_config(page_title="Cyberbullying Detection", layout="centered")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")


st.title("Cyberbullying Detection System")
st.markdown("""
Detect and categorize cyberbullying messages into:
- **Ethnicity/Race**
- **Not Cyberbullying**
- **Gender/Sexual**
- **Religion**
""")

user_input = st.text_area("Enter a sentence to analyze:")


if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        category = predict(user_input, tokenizer, model)
        st.success(f"Predicted Category: {category}")
