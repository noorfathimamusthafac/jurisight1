import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def start_ingest():
    print("🚀 Starting legal document ingestion...")
    
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDF files found in data/ directory!")
        return

    all_docs = []
    for pdf in pdf_files:
        print(f"📄 Processing {pdf}...")
        loader = PyPDFLoader(os.path.join(data_dir, pdf))
        all_docs.extend(loader.load())
    
    print(f"✂️ Splitting {len(all_docs)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    texts = splitter.split_documents(all_docs)
    
    print(f"🧠 Generating embeddings for {len(texts)} chunks using MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("📁 Creating FAISS Vector Database...")
    db = FAISS.from_documents(texts, embeddings)
    
    print("💾 Saving to ipc_vector_db...")
    db.save_local("ipc_vector_db")
    print("✅ Ingestion successfully completed!")

if __name__ == "__main__":
    start_ingest()
