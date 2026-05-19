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
    div.stButton > button:first-child {
        background-color: #ffd0d0;
        color: #333333;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: 1px solid #ff6262;
    }
    div.stButton > button:hover {
        background-color: #ff6262 !important;
        color: white !important;
    }
    .stChatInput {
        bottom: 20px;
        position: fixed;
    }
    .stAlert {
        border-radius: 12px;
    }
    [data-testid="stStatusWidget"] {
        display: none;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .lawyer-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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

# Lawyer consultation page
def show_consultation_page():
    st.title("📞 Connect with Legal Experts")
    st.markdown("### Verified Lawyers Specializing in IPC Cases")
    
    # Lawyer cards
    lawyers = [
        {
            "name": "Adv. Rajesh Khanna",
            "exp": "12 years",
            "specialization": "Criminal Law, Cyber Crime",
            "rating": "⭐ 4.9/5 (380 reviews)",
            "photo": "https://www.advocateinchandigarh.com/wp-content/uploads/2018/04/Advocate-Prashant-1.jpg"
        },
        {
            "name": "Adv. Kumar Singh",
            "exp": "8 years",
            "specialization": "Domestic Violence, Property Disputes",
            "rating": "⭐ 4.8/5 (290 reviews)",
            "photo": "https://superlawyer.in/wp-content/uploads/2023/07/AD2.png"
        },
        {
            "name": "Adv. Amit Sharma",
            "exp": "15 years",
            "specialization": "White Collar Crimes, Bail Hearings",
            "rating": "⭐ 4.7/5 (420 reviews)",
            "photo": "https://content.jdmagicbox.com/v2/comp/adilabad/n6/9999p8732.8732.140320230103.f5n6/catalogue/kema-srikanth-advocate-adilabad-lawyers-for-accident-claims-mgro5nl4ws-250.jpg"
        }
    ]
    
    for lawyer in lawyers:
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(lawyer["photo"], width=120)
            with col2:
                st.markdown(f"""
                **{lawyer['name']}**  
                📅 {lawyer['exp']} experience  
                🎯 {lawyer['specialization']}  
                {lawyer['rating']}
                """)
                
                if st.button(f"Book Consultation - {lawyer['name']}"):
                    st.session_state.selected_lawyer = lawyer
                    st.session_state.page = "booking"
            
            st.markdown("---")
    
    if st.button("🔙 Back to Chat"):
        st.session_state.page = "main"
        st.experimental_rerun()

# Booking form
def show_booking_page():
    lawyer = st.session_state.selected_lawyer
    st.title("📅 Book Appointment")
    st.markdown(f"### Consultation with {lawyer['name']}")
    
    with st.form("booking_form"):
        st.write("**Available Time Slots**")
        time_slot = st.selectbox("Select Time", 
                               ["Tomorrow 10:00 AM", "Tomorrow 2:00 PM", 
                                "Tomorrow 4:00 PM", "Next Day 11:00 AM"])
        name = st.text_input("Your Name")
        contact = st.text_input("Contact Number")
        email = st.text_input("Email Address")
        case_details = st.text_area("Brief Case Details")
        
        if st.form_submit_button("Confirm Booking"):
            st.success("🎉 Booking Confirmed! You'll receive confirmation details via email.")
            time.sleep(2)
            st.session_state.page = "main"
            st.experimental_rerun()
    
    if st.button("🔙 Back to Lawyers List"):
        st.session_state.page = "consultation"
        st.experimental_rerun()

# Main chat interface
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
        
        if st.button("👨⚖️ Consult a Lawyer Now"):
            st.session_state.page = "consultation"
            st.experimental_rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input handling
    if prompt := st.chat_input("Ask your legal question..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
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
                    
                    # Format response (clean Markdown block for proper spacing)
                    formatted_response = (
                        f"### Answer\n\n{answer}\n\n"
                        f"---\n"
                        f"⚠️ This is general legal information, not legal advice."
                    )

                    # Render once to preserve spacing/formatting
                    response_placeholder.markdown(formatted_response, unsafe_allow_html=False)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    formatted_response = "Sorry, I couldn't generate a response due to an error. Please try again."
            
            # Add reset button
            st.button("🧹 Clear Conversation", on_click=reset_conversation)

        # Add assistant response to history only if available
        if 'formatted_response' in locals() and formatted_response:
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.experimental_rerun()

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

# Page routing
if st.session_state.page == "main":
    main_chat_interface()
elif st.session_state.page == "consultation":
    show_consultation_page()
elif st.session_state.page == "booking":
    show_booking_page()