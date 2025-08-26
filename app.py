import streamlit as st
import os
import base64
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# --- CONFIGURATION ---
# Define paths
PDF_DIR = "data/pdfs"
VECTORSTORE_DIR = "chroma_db"
LOGO_PATH = "assets/bits.png"

# --- HELPER FUNCTIONS ---

@st.cache_data
def get_base64_of_bin_file(bin_file):
    """ Reads a binary file and returns its base64 encoded string. """
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.warning(f"Logo file not found at {LOGO_PATH}. Please make sure it exists.")
        return None

def set_background_logo(png_file):
    """ Sets a background logo for the Streamlit app. """
    bin_str = get_base64_of_bin_file(png_file)
    if bin_str:
        page_bg_img = f'''
        <style>
        .stApp {{
            background: none;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background-image: url("data:image/png;base64,{bin_str}");
            background-position: center;
            background-repeat: no-repeat;
            background-size: 50vw;
            opacity: 0.07;
            z-index: -1;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource
def load_and_process_pdfs():
    """
    Loads PDFs from the specified directory, splits them into chunks,
    and creates a Chroma vector store with Ollama embeddings.
    This function is cached to avoid reprocessing on every run.
    """
    if not os.path.exists(PDF_DIR) or not any(f.endswith('.pdf') for f in os.listdir(PDF_DIR)):
        st.error(f"No PDF files found in the '{PDF_DIR}' directory.")
        st.info("Please create the 'data/pdfs' directory and add your PDF files to it.")
        return None

    all_chunks = []
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

    with st.spinner(f"Processing {len(pdf_files)} PDF(s)... This may take a moment."):
        for filename in pdf_files:
            path = os.path.join(PDF_DIR, filename)
            try:
                loader = PyPDFLoader(path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_documents(documents)
                all_chunks.extend(chunks)
            except Exception as e:
                st.warning(f"Could not read {filename}: {e}")

        if not all_chunks:
            st.error("Failed to process any documents. The vector store cannot be created.")
            return None
        
        # Use a robust embedding model from Ollama
        embeddings = OllamaEmbeddings(model="llama3")
        
        # Create and save the Chroma vector store
        # The persist_directory parameter saves the vector store to disk
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        
        # We need to explicitly persist the database
        vectorstore.persist()

    return vectorstore

@st.cache_resource
def get_qa_chain():
    """
    Initializes and returns the QA chain. It loads the vector store from disk if it exists,
    otherwise it creates it by processing the PDFs.
    """
    # Use Ollama for both embeddings and the language model
    try:
        llm = Ollama(model="llama3")  # Ensure this model is pulled in your local Ollama setup
    except Exception as e:
        st.error(f"Could not connect to Ollama. Make sure Ollama is running and the model is pulled. Error: {e}")
        st.stop()
    
    embeddings = OllamaEmbeddings(model="llama3")
    
    if os.path.exists(VECTORSTORE_DIR):
        # Load existing vector store
        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
    else:
        # Create it if it doesn't exist
        with st.spinner("First-time setup: Building vector store from PDFs..."):
            vectorstore = load_and_process_pdfs()
        if vectorstore is None:
            st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    You are a helpful assistant for answering questions about the provided documents.
    Use the following context to answer the question.
    If the answer cannot be found in the context, state that you don't know. Do not make up an answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain


# --- LOGIN LOGIC ---
def login_page():
    """ Displays the login page and handles authentication. """
    st.header("Login")
    st.markdown("Please enter your credentials to access the assistant.")

    # Using a set for efficient lookup
    allowed_emails = {
        "viv1989kumar@gmail.com",
        "admin@gmail.com",
        "user@gmail.com",
    }

    with st.form("login_form"):
        email = st.text_input("Email").lower()
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Simple authentication: check if email is in the list and password is not empty
            if email in allowed_emails and password:
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid email or password. Access denied.")

# --- MAIN CHAT INTERFACE ---
def chat_interface():
    """ The main chat interface of the application. """
    st.title("📚 AI Document Assistant By Vivek Kumar")
    st.write("Ask me any question related to subjects in the MBA in AI 1st Sem....")
    st.caption(f"Logged in as: {st.session_state.user_email}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is your question?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    qa_chain = get_qa_chain()
                    result = qa_chain.invoke({"query": prompt})
                    response = result.get("result", "Sorry, I couldn't generate a response.")
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})

    # Add a button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    # Add a logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_email = ""
        st.session_state.messages = []
        st.rerun()


# --- APP ENTRY POINT ---
def main():
    st.set_page_config(page_title="AI Document Assistant", page_icon="📚")
    set_background_logo(LOGO_PATH)

    # Initialize session state for login
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    # Show login page or chat interface based on login state
    if st.session_state.logged_in:
        chat_interface()
    else:
        login_page()

if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    main()