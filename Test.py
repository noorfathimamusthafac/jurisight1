import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import re

# Load environment variables
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Smart AI-Base Legal Advice",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
    <style>
    /* ... (keep previous styles) ... */
    .legal-response {
        padding: 1rem;
        border-left: 3px solid #ff6262;
        margin: 1rem 0;
        background-color: #fff5f5;
        border-radius: 8px;
    }
    .section-title {
        color: #cc0000;
        margin-top: 1rem;
    }
    .ipc-section {
        background-color: #ffe6e6;
        padding: 0.5rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)
# Sanitize model output to hide any analysis or hidden tags
def sanitize_model_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"(?is)<think[\s\S]*?(</think>|$)", "", text)
    cleaned = re.sub(r"(?is)<analysis[\s\S]*?(</analysis>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<thought[\s\S]*?(</thought>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<system[\s\S]*?(</system>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<hidden[\s\S]*?(</hidden>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)</?(think|analysis|thought|system|hidden)[^>]*>", "", cleaned)
    return cleaned.strip()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True
    )
if "page" not in st.session_state:
    st.session_state.page = "main"

# Modified main chat interface with improved formatting
def main_chat_interface():
    # Header section
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        st.image(
            "https://img.freepik.com/free-photo/close-up-lawyer-ai-robot_23-2151015291.jpg?t=st=1738522250~exp=1738525850~hmac=6efaa30c3e59a6f8cf520941f727d0f1c4920aba5b32145be64a802c8791884f&w=1060",
            use_column_width=True
        )
        st.markdown("### Indian Legal Assistant")
        st.caption("Ask questions about the Indian Penal Code (IPC)")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.markdown(message["content"], unsafe_allow_html=True)
                st.button(
                    "🧹 Clear Conversation",
                    on_click=reset_conversation,
                    key=f"clear_{message['timestamp']}",
                    type="primary"
                )
            else:
                st.markdown(message["content"])

    # User input handling
    if prompt := st.chat_input("Ask your legal question..."):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": time.time()
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            status_container = st.status("Analyzing query...", expanded=True)
            
            with status_container:
                try:
                    result = qa_chain.invoke({"question": prompt})
                    answer = sanitize_model_output(result.get("answer", ""))
                    
                    # Extract IPC sections (avoid backslash in f-string)
                    newline = '\n'
                    ipc_sections = "".join([f"<div class='ipc-section'>{s.strip()}</div>" 
                                           for s in answer.split(newline) if 'Section' in s])
                    
                    # Format response with enhanced structure
                    formatted_response = f"""
                    <div class='legal-response'>
                        <div class='section-title'>📌 Key Points:</div>
                        {answer.split('.')[0]}.
                        
                        <div class='section-title'>⚖️ Relevant IPC Sections:</div>
                        {ipc_sections}
                        
                        <div class='section-title'>💡 Important Notes:</div>
                        <ul>
                            <li>This analysis is based on current IPC provisions</li>
                            <li>Actual legal outcomes may vary case-by-case</li>
                            <li>Consult a lawyer for specific advice</li>
                        </ul>
                        
                        <hr style='border: 1px solid #ffd0d0; margin: 1rem 0;'>
                        <div style='color: #666; font-size: 0.9rem;'>
                            ⚠️ AI-generated response. Verify with official sources.
                        </div>
                    </div>
                    """
                    
                    # Stream response in meaningful chunks
                    full_response = ""
                    chunks = formatted_response.split("<div")
                    for chunk in chunks:
                        part = f"<div{chunk}" if chunk != chunks[0] else chunk
                        full_response += part
                        time.sleep(0.15)
                        response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)
                    
                    response_placeholder.markdown(full_response, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    formatted_response = "Sorry, I couldn't generate a response due to an error. Please try again."

        # Add assistant response to history only if available
        if 'formatted_response' in locals() and formatted_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": formatted_response,
                "timestamp": time.time()
            })

# Helper functions
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.rerun()

# Initialize components with caching
@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={
            "trust_remote_code": True,
            "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"
        }
    )

@st.cache_resource(show_spinner=False)
def load_vector_store(_embeddings):
    if not os.path.exists("ipc_vector_db"):
        st.error("Vector database not found! Run ingest.py first.")
        st.stop()
    return FAISS.load_local("ipc_vector_db", _embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=False)
def initialize_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        st.error("Missing OPENAI_API_KEY or OPENAI_BASE_URL in environment!")
        st.stop()

    return ChatOpenAI(
        model="zai-org/GLM-4.6",
        temperature=0.5,
        max_tokens=1024,
        api_key=api_key,
        base_url=base_url,
    )

# System prompt template
PROMPT_TEMPLATE = """<s>[INST]
As a legal AI assistant specializing in Indian Penal Code (IPC), follow these guidelines:
1. Provide accurate, concise information based on context
2. Cite relevant IPC sections when applicable
3. If unsure, state that clearly
4. Maintain professional tone
5. Never provide legal advice

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer: [/INST]"""

# Initialize components
embeddings = load_embeddings()
db = load_vector_store(embeddings)
llm = initialize_llm()

# Create conversation chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    memory=st.session_state.memory,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question", "chat_history"]
        )
    }
)

# Run main chat interface
main_chat_interface()