Streamlit RAG AI Document Assistant
This is a complete Streamlit application that provides a chat interface to ask questions about a collection of PDF documents. It uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain and Hugging Face.

Project Structure
Before you begin, make sure your project has the following structure. You will need to create the data/pdfs and assets directories.

your-project-folder/
│
├── .streamlit/
│   └── secrets.toml         # (You will create this for local testing or on Streamlit Cloud)
│
├── data/
│   └── pdfs/
│       └── your_document_1.pdf  # <-- Add your PDF files here
│       └── your_document_2.pdf
│
├── assets/
│   └── bits.png             # <-- Add your logo here
│
├── app.py                   # (The main application code)
├── requirements.txt         # (The list of Python packages)
└── README.md                # (This file)

How to Run and Deploy
Step 1: Get a Hugging Face API Token
The application uses models hosted on the Hugging Face Hub. You will need a free API token to access them.

Go to the Hugging Face website and create an account if you don't have one.

Navigate to your profile, then go to Settings -> Access Tokens.

Create a new token with read permissions. Copy this token.

Step 2: Local Setup (Optional, but Recommended)
Windows Prerequisites
IMPORTANT: If you are on Windows, you must install the following prerequisites before installing the Python packages.

1. Microsoft C++ Build Tools

This is required to compile packages like faiss-cpu.

Go to the Visual Studio Downloads page.

Scroll down to Tools for Visual Studio and find Build Tools for Visual Studio. Click Download.

Run the installer. In the "Workloads" tab, check the box for Desktop development with C++.

Click Install. After the installation is complete, restart your computer.

2. SWIG (Simplified Wrapper and Interface Generator)

This is another tool needed to build faiss-cpu.

Go to the SWIG for Windows download page.

Download the latest version (e.g., swigwin-4.2.1).

Extract the ZIP file to a permanent location on your computer, for example, C:\swigwin.

Add SWIG to your system's PATH:

Press the Windows key, type env, and select "Edit the system environment variables".

In the System Properties window, click the "Environment Variables..." button.

In the "System variables" section, find and select the Path variable, then click "Edit...".

Click "New" and add the path to the directory where you extracted SWIG (e.g., C:\swigwin).

Click OK on all windows to save the changes.

Verify the installation: Open a new Command Prompt or PowerShell window and type swig -version. You should see the SWIG version information. If you don't, restart your computer.

Setup Steps
Clone the repository and navigate into the project directory.

Create a virtual environment:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Add your PDFs: Place all the PDF files you want the AI to learn from inside the data/pdfs/ directory.

Add your logo: Place your logo file (e.g., bits.png) inside the assets/ directory.

Set up secrets: Create a file at .streamlit/secrets.toml and add your Hugging Face token:

HUGGINGFACEHUB_API_TOKEN = "hf_YOUR_TOKEN_HERE"

Run the app:

streamlit run app.py

The first time you run it, the app will process your PDFs and build the vector database. This might take a few minutes.

Step 3: Deploy to Streamlit Cloud
Create a GitHub Repository: Upload all the project files (app.py, requirements.txt, README.md) and directories (data/pdfs, assets) to a new repository on your GitHub account.

Sign up for Streamlit Cloud: Go to share.streamlit.io and sign up using your GitHub account.

Deploy the App:

Click "New app" and select your new GitHub repository.

The branch and main file path (app.py) should be detected automatically.

Go to the "Advanced settings..." section.

In the "Secrets" section, paste your Hugging Face API token:

HUGGINGFACEHUB_API_TOKEN = "hf_YOUR_TOKEN_HERE"

Click "Deploy!".

Your app will now be deployed. The initial startup will be a bit slow as it needs to install the dependencies and build the FAISS vector store from your PDFs for the first time. Subsequent loads will be much faster.