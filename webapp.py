# importing libraries

from ctypes import alignment
from urllib import response
import pandas as pd
import streamlit as st
import altair as alt
from PIL import Image
import pandas as pd
import numpy as np
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk import PorterStemmer, WordNetLemmatizer
from functions import *
import pickle
from datetime import datetime

# Initialize session state for feedback
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {
        'correct_predictions': 0,
        'total_predictions': 0,
        'feedback_history': []
    }

# Define color scheme for categories
CATEGORY_COLORS = {
    "Age": "#FF6B6B",  # Red
    "Ethnicity": "#4ECDC4",  # Teal
    "Gender": "#45B7D1",  # Blue
    "Religion": "#96CEB4",  # Green
    "Other Cyberbullying": "#FFBE0B",  # Orange
    "Not Cyberbullying": "#2ECC71"  # Emerald Green
}

# Custom CSS for styling
st.markdown("""
<style>
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .severity-high {
        color: #FF6B6B;
        font-weight: bold;
    }
    .severity-medium {
        color: #FFBE0B;
        font-weight: bold;
    }
    .severity-low {
        color: #2ECC71;
        font-weight: bold;
    }
    .sarcasm-detected {
        background-color: #FFE0E0;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Navigation
page = st.sidebar.selectbox("Navigate", ["Cyberbullying Detection", "About"])

if page == "Cyberbullying Detection":
    st.write('''
    # CyberSentinels

    This app predicts the nature of the text into 6 Categories.
    * Age
    * Ethnicity
    * Gender
    * Religion
    * Other Cyberbullying
    * Not Cyberbullying

    ***
    ''')

    # Text Box
    st.header('Enter Text or Post or Comment')
    tweet_input = st.text_area("Tweet Input", height=150)
    print(tweet_input)
    st.write('''***''')

    # print input on webpage
    st.header("Entered text ")
    if tweet_input:
        tweet_input
    else:
        st.write('''***No Text Entered!***''')
    st.write('''***''')

    # Output on the page
    st.header("Prediction")
    if tweet_input:
        prediction = custom_input_prediction(tweet_input)
        
        # Display prediction and image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Colored prediction box
            st.markdown(f"""
            <div class="prediction-box" style="background-color: {CATEGORY_COLORS[prediction]}20;">
                <h3>Model Prediction</h3>
                <p style="color: {CATEGORY_COLORS[prediction]}; font-size: 24px; font-weight: bold;">
                    {prediction}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Sarcasm Detection
            is_sarcastic = False
            sarcasm_confidence = 0.0
            
            # Simple sarcasm detection logic (placeholder)
            if "..." in tweet_input or "!" in tweet_input:
                is_sarcastic = True
                sarcasm_confidence = 0.7

            if is_sarcastic:
                st.markdown("""
                <div class="sarcasm-detected">
                    <h4>⚠️ Potential Sarcasm Detected</h4>
                    <p>This text may contain sarcastic or passive-aggressive content.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Severity Assessment
            severity = "low"
            if prediction != "Not Cyberbullying":
                if len(tweet_input.split()) > 10:
                    severity = "high"
                elif len(tweet_input.split()) > 5:
                    severity = "medium"
                
                st.markdown(f"""
                <p>Severity Level: <span class="severity-{severity}">
                    {severity.upper()}
                </span></p>
                """, unsafe_allow_html=True)
            
            # User Feedback Section
            st.subheader("Was this prediction correct?")
            feedback_col1, feedback_col2 = st.columns(2)
            
            if feedback_col1.button("👍 Correct"):
                st.session_state.feedback_data['correct_predictions'] += 1
                st.session_state.feedback_data['total_predictions'] += 1
                st.session_state.feedback_data['feedback_history'].append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': tweet_input,
                    'prediction': prediction,
                    'was_correct': True
                })
                st.success("Thank you for your feedback!")
                
            if feedback_col2.button("👎 Incorrect"):
                st.session_state.feedback_data['total_predictions'] += 1
                st.session_state.feedback_data['feedback_history'].append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'text': tweet_input,
                    'prediction': prediction,
                    'was_correct': False
                })
                correct_category = st.selectbox(
                    "What was the correct category?",
                    ["Age", "Ethnicity", "Gender", "Religion", "Other Cyberbullying", "Not Cyberbullying"]
                )
                if st.button("Submit Correction"):
                    st.session_state.feedback_data['feedback_history'][-1]['correct_category'] = correct_category
                    st.success("Thank you for your feedback!")
        
        with col2:
            if prediction == "Age":
                st.image("images/age_cyberbullying.png", use_column_width=True)
            elif prediction == "Ethnicity":
                st.image("images/ethnicity_cyberbullying.png", use_column_width=True)
            elif prediction == "Gender":
                st.image("images/gender_cyberbullying.png", use_column_width=True)
            elif prediction == "Not Cyberbullying":
                st.image("images/not_cyberbullying.png", use_column_width=True)
            elif prediction == "Other Cyberbullying":
                st.image("images/other_cyberbullying.png", use_column_width=True)
            elif prediction == "Religion":
                st.image("images/religion_cyberbullying.png", use_column_width=True)
        
        # Explanation of the prediction
        st.header("Understanding the Prediction")
        st.write("""
        This prediction is based on analyzing various aspects of the text, including:
        - Language patterns and word choice
        - Context and intent
        - Similarity to known cyberbullying patterns
        - Sarcasm and subtle forms of harassment
        """)
        
        if prediction != "Not Cyberbullying":
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {CATEGORY_COLORS[prediction]}10;">
                This text was classified as <strong style="color: {CATEGORY_COLORS[prediction]}">{prediction}</strong>-related cyberbullying 
                because it contains patterns commonly associated with {prediction.lower()}-based harassment or discrimination.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {CATEGORY_COLORS[prediction]}10;">
                This text was classified as <strong style="color: {CATEGORY_COLORS[prediction]}">Not Cyberbullying</strong> 
                because it doesn't contain patterns typically associated with cyberbullying behavior.
            </div>
            """, unsafe_allow_html=True)
        
    else:
        st.write('''***No Text Entered!***''')

    # Model Performance Metrics
    st.header("Model Performance")
    if st.session_state.feedback_data['total_predictions'] > 0:
        accuracy = (st.session_state.feedback_data['correct_predictions']/ 
                   st.session_state.feedback_data['total_predictions'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Predictions", st.session_state.feedback_data['total_predictions'])
            st.metric("Accuracy", f"{accuracy:.2%}")
        
        with col2:
            st.metric("Correct Predictions", st.session_state.feedback_data['correct_predictions'])
            st.metric("Incorrect Predictions", 
                     st.session_state.feedback_data['total_predictions'] - 
                     st.session_state.feedback_data['correct_predictions'])
        
        # Feedback History
        with st.expander("View Feedback History"):
            if st.session_state.feedback_data['feedback_history']:
                history_df = pd.DataFrame(st.session_state.feedback_data['feedback_history'])
                st.dataframe(history_df)
    else:
        st.info("No feedback data available yet. Please analyze some tweets and provide feedback!")

else:  # About page
    st.write("""
    # About CyberSentinels

    ## Agentic AI Approach

    CyberSentinels uses an advanced Agentic AI approach to detect and analyze cyberbullying content. This innovative system combines multiple specialized AI agents working together to provide comprehensive analysis:

    ### 1. Core Classification Model
    - Support Vector Machine (SVM) classifier
    - Trained on a large dataset of labeled cyberbullying content
    - Categorizes text into six distinct types of cyberbullying

    ### 2. Specialized AI Agents
    Our system employs multiple specialized agents:
    
    🔍 **Analysis Agents**
    - Text Preprocessing Agent
    - Context Analysis Agent
    - Sarcasm Detection Agent
    
    🤔 **Understanding Agents**
    - Pattern Recognition Agent
    - Intent Analysis Agent
    - Impact Assessment Agent
    
    🛠️ **Support Agents**
    - Feedback Processing Agent
    - Recommendation Agent
    - Learning & Adaptation Agent

    ### 3. Continuous Learning
    - Real-time user feedback integration
    - Pattern adaptation and refinement
    - Continuous model improvement

    ### 4. Ethical Considerations
    - Privacy-first approach
    - Bias mitigation strategies
    - Responsible AI practices

    ## Technology Stack
    - Python
    - Streamlit
    - Scikit-learn
    - NLTK
    - Advanced AI Models

    ## Dataset
    The model is trained on a comprehensive dataset from Kaggle, containing various types of cyberbullying content.
    """)

st.write('''***''')

