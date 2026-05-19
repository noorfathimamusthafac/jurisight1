import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate

try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    try:
        from langchain_classic.memory import ConversationBufferWindowMemory
    except ImportError:
        from langchain.memory import ConversationBufferWindowMemory

try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    try:
        from langchain_classic.chains import ConversationalRetrievalChain
    except ImportError:
        from langchain.chains import ConversationalRetrievalChain

from langchain_openai import ChatOpenAI
import re
import logging
import warnings
import csv
from langchain_community.llms import HuggingFaceHub




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
    </style>
""", unsafe_allow_html=True)

# Quiet noisy library logs/warnings (benign torch.classes and HF logging)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Examining the path of torch\.classes")
# Sanitize model output to hide any analysis or hidden tags
def sanitize_model_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Remove <think>...</think> blocks, including when closing tag is missing
    cleaned = re.sub(r"(?is)<think[\s\S]*?(</think>|$)", "", text)
    # Remove other hidden analysis blocks if present (with or without closing tag)
    cleaned = re.sub(r"(?is)<analysis[\s\S]*?(</analysis>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<thought[\s\S]*?(</thought>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<system[\s\S]*?(</system>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<hidden[\s\S]*?(</hidden>|$)", "", cleaned)
    # Also strip any remaining opening/closing tags of those names
    cleaned = re.sub(r"(?is)</?(think|analysis|thought|system|hidden)[^>]*>", "", cleaned)
    return cleaned.strip()

# Refuse harmful or illegal requests with a clear, user-facing message
def is_harmful_request(user_question: str) -> bool:
    if not user_question:
        return False
    q = user_question.lower()
    red_flags = [
        "get away with murder",
        "how to murder",
        "kill someone",
        "hide a body",
        "evade police",
        "commit a crime",
    ]
    return any(flag in q for flag in red_flags)

def refusal_response() -> str:
    return (
        "### Safety Notice\n\n"
        "I can’t assist with harming others, evading law enforcement, or committing crimes.\n\n"
        "Under the IPC, unlawful killing is a serious offense (Sections 299–304). If you need lawful information, I can explain the legal framework (e.g., self‑defense provisions, Section 80 accidents) or suggest safe, legal alternatives."
    )

# Optional: load IPC→BNS mapping from CSV if present
@st.cache_resource(show_spinner=False)
def load_bns_mapping(path: str = "data/bns_mapping.csv"):
    """Load mapping into forward (IPC->BNS) and reverse (BNS->IPC) indexes, plus raw rows."""
    ipc_to_bns: dict[str, list[dict]] = {}
    bns_to_ipc: dict[str, list[dict]] = {}
    raw_rows: list[dict] = []
    try:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ipc_key = str(row.get('ipc_section', '')).strip().upper()
                bns_key = str(row.get('bns_section', '')).strip()
                title = str(row.get('title', '')).strip()
                notes = str(row.get('notes', '')).strip()
                entry = {'ipc_section': ipc_key, 'bns_section': bns_key, 'title': title, 'notes': notes}
                raw_rows.append(entry)
                if ipc_key and ipc_key.isdigit():  # exact numeric key only
                    ipc_to_bns.setdefault(ipc_key, []).append({'bns_section': bns_key, 'title': title, 'notes': notes})
                if bns_key:
                    bns_to_ipc.setdefault(bns_key, []).append({'ipc_section': ipc_key, 'title': title, 'notes': notes})
    except FileNotFoundError:
        return {}, {}, []
    except Exception:
        return {}, {}, []
    return ipc_to_bns, bns_to_ipc, raw_rows

def extract_ipc_sections(text: str):
    if not text:
        return []
    candidates = set()
    # Patterns like: Section 299, Sec. 300, s. 304A, IPC 376, etc.
    for m in re.finditer(r"(?i)(?:section|sec\.|s\.)\s*(\d+[A-Z]?)", text):
        candidates.add(m.group(1).upper())
    for m in re.finditer(r"(?i)\bIPC\s*(\d+[A-Z]?)", text):
        candidates.add(m.group(1).upper())
    return sorted(candidates)

def extract_bns_sections(text: str):
    if not text:
        return []
    # BNS often like 2.01, 3.2.4 etc.
    return sorted({m.group(0) for m in re.finditer(r"\b\d+(?:\.\d+){1,3}\b", text)})

def _ipc_expr_matches(expr: str, sec: str) -> bool:
    """Check if an IPC section number (e.g., '299') matches an expression like '299-309/321', '376 series', '121-140 series', '375, 376 series, 354 series, 509'."""
    if not expr or not sec or not sec.isdigit():
        return False
    s = sec
    expr_l = expr.lower()
    # Direct token match
    tokens = [t.strip() for t in re.split(r"[,/]|\band\b|\bor\b", expr_l) if t.strip()]
    for t in tokens:
        # range like '299-309'
        m = re.match(r"^(\d+)\s*-\s*(\d+)", t)
        if m:
            a, b = m.group(1), m.group(2)
            try:
                if int(a) <= int(s) <= int(b):
                    return True
            except ValueError:
                pass
        # series like '376 series' or '121-140 series'
        m2 = re.match(r"^(\d+)(?:\s*-\s*(\d+))?\s*series$", t)
        if m2:
            a, b = m2.group(1), m2.group(2)
            if b:
                try:
                    if int(a) <= int(s) <= int(b):
                        return True
                except ValueError:
                    pass
            else:
                if s.startswith(a):
                    return True
        # plain number
        if t.isdigit() and t == s:
            return True
    # fallback: direct substring ' 299 ' presence
    return re.search(rf"(?<!\d){re.escape(s)}(?!\d)", expr_l) is not None

def augment_with_bns(answer: str, question: str) -> str:
    ipc_to_bns, bns_to_ipc, raw_rows = load_bns_mapping()
    if not ipc_to_bns and not bns_to_ipc:
        return answer
    ipc_mentioned = set(extract_ipc_sections(question)) | set(extract_ipc_sections(answer))
    bns_mentioned = set(extract_bns_sections(question)) | set(extract_bns_sections(answer))
    rows = []
    # From IPC -> BNS
    for sec in sorted(ipc_mentioned):
        for entry in ipc_to_bns.get(sec, [])[:5]:
            bns = entry.get('bns_section')
            title = entry.get('title')
            if bns:
                line = f"- IPC {sec} → BNS {bns}"
                if title:
                    line += f" — {title}"
                rows.append(line)
        # Range/list/series matches from raw rows
        if raw_rows:
            for rr in raw_rows:
                expr = rr.get('ipc_section', '')
                bns = rr.get('bns_section', '')
                title = rr.get('title', '')
                if not bns:
                    continue
                if expr and (not expr.isdigit()) and _ipc_expr_matches(expr, sec):
                    line = f"- IPC {sec} ↔ BNS {bns}"
                    if title:
                        line += f" — {title}"
                    rows.append(line)
    # From BNS -> IPC (if user mentions only BNS)
    for bns in sorted(bns_mentioned):
        for entry in bns_to_ipc.get(bns, [])[:5]:
            ipc = entry.get('ipc_section', '').upper()
            title = entry.get('title')
            if ipc:
                line = f"- BNS {bns} → IPC {ipc}"
                if title:
                    line += f" — {title}"
                rows.append(line)
    if not rows:
        return answer
    # Deduplicate and limit
    dedup = []
    seen = set()
    for r in rows:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    mapping_block = "\n\n### IPC ↔ BNS Mapping\n" + "\n".join(dedup[:10])
    return answer + mapping_block

def mapping_only_block(question: str) -> str | None:
    """Return a mapping block even if the model fails, so users still get IPC↔BNS links."""
    try:
        ipc_to_bns, bns_to_ipc = load_bns_mapping()
    except Exception:
        return None
    if not ipc_to_bns and not bns_to_ipc:
        return None
    ipc_mentioned = set(extract_ipc_sections(question))
    bns_mentioned = set(extract_bns_sections(question))
    rows: list[str] = []
    for sec in sorted(ipc_mentioned):
        for entry in ipc_to_bns.get(sec, [])[:10]:
            bns = entry.get('bns_section')
            title = entry.get('title')
            if bns:
                line = f"- IPC {sec} → BNS {bns}"
                if title:
                    line += f" — {title}"
                rows.append(line)
    for bns in sorted(bns_mentioned):
        for entry in bns_to_ipc.get(bns, [])[:10]:
            ipc = entry.get('ipc_section', '').upper()
            title = entry.get('title')
            if ipc:
                line = f"- BNS {bns} → IPC {ipc}"
                if title:
                    line += f" — {title}"
                rows.append(line)
    if not rows:
        return None
    dedup = []
    seen = set()
    for r in rows:
        if r not in seen:
            seen.add(r)
            dedup.append(r)
    return "### IPC ↔ BNS Mapping (from dataset)\n" + "\n".join(dedup[:10])

# Optional: deterministic overrides for specific questions
def get_canned_response(user_question: str) -> str | None:
    if not user_question:
        return None
    # Normalize and perform keyword-based matching
    q = re.sub(r"\s+", " ", user_question.strip().lower())
    q_plain = re.sub(r"[^a-z\s]", "", q)
    if ("legal" in q_plain and ("kill" in q_plain or "killing" in q_plain or "murder" in q_plain or "homicide" in q_plain)):
        return (
            "### Legality of killing a person under the Indian Penal Code (IPC)\n\n"
            "- Killing a person is generally unlawful. The IPC treats unlawful killing as a serious offense, but it also recognizes limited circumstances where causing death may be lawful or justified.\n\n"
            "### Lawful (or Justifiable) Homicide\n"
            "- Self-defense: Causing death while exercising the right of private defense against an imminent threat, within legal limits (Sections 100–106 IPC).\n"
            "- Accident or misfortune: Death caused by accident while doing a lawful act in a lawful manner with proper care and caution (Section 80 IPC).\n"
            "- Acts bound or justified by law:\n"
            "  - Act of a person bound by law or who, in good faith, believes they are bound by law (Section 76 IPC).\n"
            "  - Act of a Judge acting judicially (Section 77 IPC).\n"
            "  - Act done pursuant to a Court’s judgment or order (Section 78 IPC).\n"
            "  - Act justified by law or by good-faith mistake of fact (Section 79 IPC).\n"
            "- With consent/for benefit (strictly limited):\n"
            "  - Acts done with consent and for a person’s benefit (Sections 87, 88 IPC).\n"
            "  - Acts done without consent but for a person’s benefit in emergencies (Section 92 IPC).\n"
            "  - Note: These provisions are narrowly construed and do not generally legalize intentional killing; applicability depends on facts.\n\n"
            "### Unlawful Homicide\n"
            "- Culpable homicide (Section 299 IPC): Causing death with intent to cause death, intent to cause bodily injury likely to cause death, or with knowledge that the act is likely to cause death.\n"
            "- Murder (Section 300 IPC): Aggravated culpable homicide—e.g., where there is clear intent to cause death or bodily injury sufficient in the ordinary course of nature to cause death; subject to specified exceptions.\n"
            "- Causing death by negligence (Section 304A IPC): Rash or negligent act causing death (without intention).\n"
            "- Abetment of suicide (Sections 305, 306 IPC): Punishes abetment, including where the person abetted is a child or insane person (305), and general abetment of suicide (306).\n\n"
            "### Key takeaway\n"
            "- Killing is overwhelmingly illegal under the IPC. It may be lawful only in narrowly defined situations (e.g., self-defense, accident, acts justified by law), which are strictly evaluated based on evidence and context.\n\n"
            "—\nThis is general legal information, not legal advice. For any real situation, consult a qualified advocate."
        )
    if "bail" in q_plain:
        return (
            "### Bail under Indian law (CrPC)\n\n"
            "- Types: Regular (Sections 437–439 CrPC), Anticipatory (Section 438 CrPC), Default/statutory (Section 167(2) CrPC).\n"
            "- Bailable vs non-bailable: Section 436 (bailable) — bail is a right; Sections 437–439 (non-bailable) — court discretion.\n"
            "- Usual steps:\n"
            "  1. Engage counsel; gather FIR/complaint and case materials.\n"
            "  2. File bail/anticipatory bail with grounds (no flight risk, cooperation, roots in community).\n"
            "  3. Court may impose conditions (e.g., surety, passport deposit, appearances).\n\n"
            "- Key considerations: gravity of offence, antecedents, risk of tampering/absconding, parity with co-accused.\n\n"
            "—\nThis is general legal information, not legal advice."
        )
    if "help" in q_plain and len(q_plain.split()) <= 10:
        return (
            "### How I can help\n\n"
            "Ask specific legal questions (e.g., ‘Process to seek anticipatory bail under Section 438 CrPC?’).\n"
            "I can summarize IPC/CrPC provisions and explain general procedures.\n\n"
            "—\nThis is general legal information, not legal advice."
        )
    return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True
    )

# Header section
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image(
        "https://img.freepik.com/free-photo/close-up-lawyer-ai-robot_23-2151015291.jpg?t=st=1738522250~exp=1738525850~hmac=6efaa30c3e59a6f8cf520941f727d0f1c4920aba5b32145be64a802c8791884f&w=1060",
        use_column_width=True
    )
    st.markdown("### Indian Legal Assistant")
    st.caption("Ask questions about the Indian Penal Code (IPC)")

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
        model=os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
        temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2048")),
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

# Chat interface functions
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()
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
                if is_harmful_request(prompt):
                    answer = refusal_response()
                else:
                    # Minimal retry around model call and surface first-line error if any
                    error_text = ""
                    answer = ""
                    for _ in range(2):
                        try:
                            result = qa_chain.invoke({"question": prompt})
                            answer = sanitize_model_output(result.get("answer", ""))
                            break
                        except Exception as e_inner:
                            error_text = str(e_inner).split("\n", 1)[0]
                            time.sleep(0.3)
                    # Auto-continue if truncated
                    if answer and len(answer.split()) > 0 and answer.rstrip().endswith("…") or answer.rstrip().endswith("..."):
                        try:
                            cont = qa_chain.invoke({"question": "Continue."})
                            more = sanitize_model_output(cont.get("answer", ""))
                            if more:
                                answer = answer.rstrip("….") + " " + more
                        except Exception:
                            pass
                    # Best-effort BNS augmentation; never let it break answers
                    try:
                        if answer and os.getenv("DISABLE_BNS_MAPPING", "0") != "1":
                            answer = augment_with_bns(answer, prompt)
                    except Exception:
                        pass
                    if not answer:
                        mapping_block = mapping_only_block(prompt)
                        suffix = f"\n\nDetails: {error_text}" if error_text else ""
                        answer = (
                            (mapping_block + "\n\n" if mapping_block else "") +
                            "I couldn't generate a response just now. Please try again, or ask a focused legal question (e.g., ‘Grounds for bail under Sections 437–439 CrPC?’)." + suffix
                        )
                
                # Format response (clean Markdown block for proper spacing)
                formatted_response = (
                    f"{answer}\n\n"
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