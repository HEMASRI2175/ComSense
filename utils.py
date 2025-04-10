import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

# Load Gemini API key
gemini_api_key = st.secrets.get('GEMINI_API_KEY', os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=gemini_api_key)

def get_summary(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(f"Summarize the following text:\n{text}")
        return response.text
    except Exception as e:
        st.error(f"Failed to generate summary: {e}")
        return None
