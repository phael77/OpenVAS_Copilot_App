import os
import base64
import gc
import tempfile
import uuid
import time

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore, FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if api_key is None:
    raise ValueError("API Key not found. Please set the OPENAI_API_KEY environment variable")
else:
    print("API Key loaded successfully")

if "id" is not st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    
session_id = st.session_state.id

def reset_chat():
    st.session_state.messages = []
    gc.collect()
    

def display_file_preview(file, file_type):
    """Exibe a prévia do arquivo na interface Streamlit."""
    st.markdown(f"### {file_type.upper()} File Preview")
    if file_type == "pdf":
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f"""
            <iframe src="data:application/pdf;base64,{base64_pdf}" 
                    width="100%" height="600px" 
                    type="application/pdf"></iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    elif file_type == "csv":
        file.seek(0)  # Reinicia o cursor do arquivo
        df = pd.read_csv(file)
        st.dataframe(df.head())
    elif file_type == "xml":
        file.seek(0)  # Reinicia o cursor do arquivo
        tree = ET.parse(file)
        root = tree.getroot()
        st.code(ET.tostring(root, encoding="unicode", method="xml")[:500])
        
@st.cache_resource
def load_embeddings():
    """Carrega o modelo de embeddings OpenAI."""
    return OpenAIEmbeddings()

@st.cache_resource
def load_llm():
    """Carrega o modelo de linguagem GPT."""
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_completion_tokens=1000)

def process_csv(file_path):
    """Processa arquivos CSV e converte em documentos LangChain."""
    df = pd.read_csv(file_path)
    documents = [Document(page_content=row.to_string()) for _, row in df.iterrows()]
    return documents

def process_xml(file_path):
    """Processa arquivos XML e converte em documentos LangChain."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    documents = [Document(page_content=ET.tostring(child, encoding="unicode", method="xml")) for child in root]
    return documents

def process_pdf(file_path):
    """Processa arquivos PDF e converte em documentos LangChain."""
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    return loader.load()

# Sidebar para carregar arquivos
with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf`, `.csv`, or `.xml` file", type=["pdf", "csv", "xml"])

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get("file_cache", {}):
                    # Processa o arquivo conforme o tipo
                    if uploaded_file.name.endswith(".pdf"):
                        documents = process_pdf(file_path)
                        file_type = "pdf"
                    elif uploaded_file.name.endswith(".csv"):
                        documents = process_csv(file_path)
                        file_type = "csv"
                    elif uploaded_file.name.endswith(".xml"):
                        documents = process_xml(file_path)
                        file_type = "xml"
                    else:
                        st.error("Unsupported file type.")
                        st.stop()

                    # Configuração de embeddings e vetor
                    embeddings = load_embeddings()
                    vectorstore = FAISS.from_documents(documents, embeddings)

                    # Configuração do LLM e RetrievalQA
                    llm = load_llm()
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=vectorstore.as_retriever(),
                        chain_type_kwargs={
                            "prompt": PromptTemplate(
                                input_variables=["context", "question"],
                                template="""
                                Use o seguinte contexto para responder à pergunta.
                                Contexto:
                                {context}
                                Pergunta:
                                {question}
                                Resposta:
                                """,
                            )
                        },
                    )

                    st.session_state.file_cache[file_key] = qa_chain
                else:
                    qa_chain = st.session_state.file_cache[file_key]

                # Exibe o arquivo carregado
                st.success("Ready to Chat!")
                display_file_preview(uploaded_file, file_type)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

# Interface do Chat
col1, col2 = st.columns([6, 1])

with col1:
    st.header("OpenVAS Copilot")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    reset_chat()

# Exibe mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Campo de entrada do chat
if prompt := st.chat_input("What's up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Processa a resposta do LLM
        qa_chain = st.session_state.file_cache.get(file_key)
        if qa_chain:
            response = qa_chain.run(prompt)
            full_response = response

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})