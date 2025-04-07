import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Web + PDF Chatbot")

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY is not set")
else:
    genai.configure(api_key=api_key)

# Functions
def fetch_webpage_text(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            return text
        else:
            st.warning(f"Failed to fetch webpage. Status code: {response.status_code}")
            return ""
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are a helpful assistant. Answer the question using the provided context.
    If the answer is not present,     Extract the following details from SHL website: 

    - Assessment name and URL from SHL website - https://www.shl.com/solutions/products/product-catalog/
    - Remote Testing Support -(Yes/No)
    - Adaptive/IRT Support -(Yes/No)
    - Duration
    - Test type


    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])

# Main App
st.title("SHL's product catalog ")

user_question = st.text_input("Ask a question from the provided documents or webpage:")

if user_question:
    user_input(user_question)

from pathlib import Path

with st.sidebar:
    st.header("SHL's product catalog")
    
    default_url = "https://www.shl.com/solutions/products/product-catalog/"
    url = st.text_input("Enter SHL Catalog URL", value=default_url)


    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):

            full_text = ""

            # If URL is provided, fetch its content
            if url:
                web_text = fetch_webpage_text(url)
                full_text += web_text

            if pdf_docs:
                pdf_text = get_pdf_text(pdf_docs)
                full_text += pdf_text
            else:
                st.info("use default")
                default_pdf_path = Path("Skills-Assessment-Catalog.pdf")
                if default_pdf_path.exists():
                    with open(default_pdf_path, "rb") as f:
                        default_pdf_text = get_pdf_text([f])
                        full_text += default_pdf_text
                else:
                    st.error("Default PDF not found at 'data/default.pdf'.")

            if full_text.strip():
                text_chunks = get_text_chunks(full_text)
                get_vector_store(text_chunks)
                st.success("Vector store created successfully!")
            else:
                st.warning("No text found to process.")
