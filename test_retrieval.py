import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def test_retrieval(query):
    print(f"\n🔍 Testing retrieval for: '{query}'")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
    
    docs = db.similarity_search(query, k=5)
    print(f"✅ Found {len(docs)} relevant chunks.")
    
    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}) ---")
        print(doc.page_content[:400] + "...")

if __name__ == "__main__":
    queries = ["What is Section 302 IPC?", "Section 309 IPC suicide", "Section 495 IPC", "Murder punishment"]
    for q in queries:
        test_retrieval(q)
