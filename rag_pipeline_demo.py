import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Set Streamlit config
st.set_page_config(page_title="Local RAG Assistant", layout="wide")

# Path to docs
DOCS_FOLDER = "docs"

# Load the LLM
@st.cache_resource
def load_llm():
    generator = pipeline("text-generation", model="sshleifer/tiny-gpt2", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=generator)
    return llm

# Load and split documents
@st.cache_resource
def load_documents():
    documents = []
    for filename in os.listdir(DOCS_FOLDER):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(DOCS_FOLDER, filename))
            documents.extend(loader.load())
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

# Create vector store
@st.cache_resource
def create_vector_store(_docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_docs, embeddings)

# Tool routing
def is_tool_query(query: str):
    keywords = ["calculate", "compute", "define", "convert"]
    return any(word in query.lower() for word in keywords)

# Example tool function
def define_tool(query: str):
    return "This is a tool-based response. Custom logic goes here."

# Load components
llm = load_llm()
docs = load_documents()
db = create_vector_store(docs)
retriever = db.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define tool
tool = Tool(
    name="Definition Tool",
    func=define_tool,
    description="Useful for queries involving calculations or definitions."
)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# UI
st.title("📚 Local RAG-based Knowledge Assistant")

query = st.text_input("Enter your question:")

if query:
    st.markdown("---")
    if is_tool_query(query):
        st.subheader("🔧 Tool Invoked")
        response = agent.run(query)
    else:
        st.subheader("🧠 Retrieved Snippets")
        relevant_docs = retriever.get_relevant_documents(query)
        for i, doc in enumerate(relevant_docs, 1):
            st.markdown(f"**Snippet {i}:** {doc.page_content[:300]}...")
        response = qa_chain.run(query)

    st.markdown("### 💬 Final Answer:")
    st.success(response)
