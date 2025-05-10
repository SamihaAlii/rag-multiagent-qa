RAG Local Agent Demo

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using local language models and LangChain. It retrieves relevant information from local text documents and answers user questions via a basic agentic workflow.

---

## 📌 Features

- ✅ Document ingestion from `.txt` files
- ✅ Vector store creation using FAISS
- ✅ Keyword-based agent routing to tools
- ✅ Streamlit-based interactive UI
- ✅ Completely offline (no OpenAI or cloud dependencies)
- ✅ Lightweight models run locally via Hugging Face

---

## 🗂️ Project Structure

rag_local_agent_demo/
├── docs/ # Contains source .txt documents
│ ├── support_info.txt
│ ├── product_specs.txt
│ └── return_policy.txt
├── rag_pipeline_demo.py # Main Streamlit app
├── requirements.txt # Python dependencies
└── README.md # Project overview and usage

yaml
Copy
Edit

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag_local_agent_demo.git
cd rag_local_agent_demo
2. Create virtual environment (optional but recommended)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚀 Running the Streamlit App
bash
Copy
Edit
streamlit run rag_pipeline_demo.py
The app will:

Load and embed your documents from the docs/ folder.

Launch a simple UI for asking questions.

Use a keyword-based agent to route queries to the right context.

📄 Example Query
Input
What are your customer support hours?

Output
Our support team is available Monday through Friday, 9 AM to 5 PM IST.

