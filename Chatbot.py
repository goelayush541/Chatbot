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
import docx
import pandas as pd
from PIL import Image
import pytesseract

load_dotenv()

# Configure Tesseract path - confirmed by user in Step 172
tesseract_cmd_path = r'C:\Program Files\Sejda PDF Desktop\resources\vendor\tesseract-windows-x64\tesseract.exe'
tessdata_path = r'C:\Program Files\Sejda PDF Desktop\resources\vendor\tessdata'

if os.path.exists(tesseract_cmd_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
    # Set TESSDATA_PREFIX - normalize slashes and ensure no quotes
    os.environ['TESSDATA_PREFIX'] = tessdata_path.replace('\\', '/')
    tesseract_found = True
else:
    tesseract_found = False

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

def get_docx_text(docx_file):
    doc = docx.Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def get_excel_text(excel_file):
    df = pd.read_excel(excel_file)
    return df.to_string()

def get_image_text(image_file):
    if not tesseract_found:
        return "[OCR Error: Tesseract not detected]"
    try:
        img = Image.open(image_file)
        # Rely on TESSDATA_PREFIX environment variable set above
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"[OCR Error: {e}]"

def get_txt_text(txt_file):
    try:
        return txt_file.read().decode("utf-8")
    except Exception:
        try:
            return txt_file.read().decode("latin-1")
        except:
            return ""

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
            st.error("Please upload and process documents first.")
            return

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain(api_key)

    response = chain.invoke(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chatbot")
    st.header(" CHATBOT")

    # Sidebar for API Key and document upload
    with st.sidebar:
        st.title("Menu:")
        api_key = st.text_input("Enter your Groq API Key", type="password")
        if not api_key:
            st.warning("Please enter your Groq API Key to proceed.")
        
        st.subheader("Your documents")
        uploaded_files = st.file_uploader("Upload your documents here and click on 'Process'", 
                                         accept_multiple_files=True,
                                         type=["pdf", "docx", "xlsx", "png", "jpg", "jpeg", "txt"])
        if st.button("Process"):
            if not api_key:
                st.error("API Key is required to process documents.")
            elif not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                with st.spinner("Processing..."):
                    raw_text = ""
                    process_log = []
                    for file in uploaded_files:
                        file_name = file.name.lower()
                        extracted_text = ""
                        try:
                            if file_name.endswith(".pdf"):
                                extracted_text = get_pdf_text([file])
                            elif file_name.endswith(".docx"):
                                extracted_text = get_docx_text(file)
                            elif file_name.endswith(".xlsx"):
                                extracted_text = get_excel_text(file)
                            elif file_name.endswith((".png", ".jpg", ".jpeg")):
                                st.sidebar.image(file, caption=file.name, use_container_width=True)
                                extracted_text = get_image_text(file)
                            elif file_name.endswith(".txt"):
                                extracted_text = get_txt_text(file)
                            
                            if extracted_text and extracted_text.strip():
                                raw_text += extracted_text + "\n"
                                process_log.append(f"✅ {file.name}: Extracted {len(extracted_text)} characters")
                            else:
                                process_log.append(f"⚠️ {file.name}: No text found or extraction failed")
                        except Exception as e:
                            process_log.append(f"❌ {file.name}: Error - {str(e)}")
                    
                    for log in process_log:
                        st.sidebar.write(log)

                    if raw_text.strip():
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Vector Store created and cached!")
                    else:
                        st.error("No text could be extracted from any of the uploaded files.")
                        if not tesseract_found:
                            st.info(f"Tesseract OCR not detected. Tried: {tesseract_cmd_path}")

    # User input field
    user_question = st.text_input("Ask a question about your documents...")

    if user_question:
        if not api_key:
            st.error("Please enter your Groq API Key in the sidebar.")
        else:
            user_input(user_question, api_key)

if __name__ == "__main__":
    main()
