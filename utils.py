import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = st.secrets['GEMINI_API_KEY']

def get_summary(text):
    text_splitter = TokenTextSplitter(
        chunk_size=1000, 
        chunk_overlap=10
    )
    chunks = text_splitter.create_documents([text])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=gemini_api_key
    )
    
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce"
    )

    response = chain.run(chunks)

    return response
