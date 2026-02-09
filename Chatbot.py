import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Cache the embedding model to avoid reloading it (500MB+) on every interaction
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = load_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Store in session state for instant access
    st.session_state.vector_store = vector_store
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    # Retrieve from session state if available, otherwise load from disk once
    if "vector_store" not in st.session_state:
        if os.path.exists("faiss_index"):
            embeddings = load_embeddings()
            st.session_state.vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        else:
            st.error("Please upload and process a PDF first.")
            return

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chatbot")
    st.header(" CHATBOT")

    # Sidebar for API Key and PDF upload
    with st.sidebar:
        st.title("Menu:")
        api_key = st.text_input("Enter your Groq API Key", type="password")
        if not api_key:
            st.warning("Please enter your Groq API Key to proceed.")
        
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if not api_key:
                st.error("API Key is required to process PDFs.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Vector Store created and cached!")

    # User input field
    user_question = st.text_input("Ask a question about your documents...")

    if user_question:
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        else:
            user_input(user_question, api_key)

if __name__ == "__main__":
    main()