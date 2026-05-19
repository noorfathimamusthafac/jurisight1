import os
import sys

# Force UTF-8 output to avoid UnicodeEncodeError on Windows cp1252 terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# --- PROTOBUF GHOST PATCH (Fixes 'runtime_version' hell) ---
try:
    from google.protobuf import runtime_version
except ImportError:
    import google.protobuf
    from types import ModuleType
    # Create the missing module if it doesn't exist
    gh = ModuleType('runtime_version')
    gh.OSS = 0 # Dummy values to satisfy transformers
    gh.__file__ = __file__
    google.protobuf.runtime_version = gh
    sys.modules['google.protobuf.runtime_version'] = gh
    print("🏛️ JuriSight Patch: Protobuf 'runtime_version' successfully virtualized.")
# ----------------------------------------------------------
import time
from flask import Flask, request, jsonify, render_template, session, stream_with_context, Response
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
try:
    from langchain.memory import ConversationBufferWindowMemory
except ImportError:
    from langchain_classic.memory import ConversationBufferWindowMemory
try:
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_classic.chains import ConversationalRetrievalChain
import re
import logging
import warnings
import csv
import uuid
from werkzeug.utils import secure_filename
import pdfplumber
from docx import Document
import json

# Load environment variables
load_dotenv()

# Quiet noisy library logs
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=r"Examining the path of torch\.classes")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-in-production")
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# Store conversation memories per session
conversation_memories = {}

# Store document analyses
document_analyses = {}

# Global Cache for BNS Mapping
BNS_MAPPING_CACHE = {
    'ipc_to_bns': {},
    'bns_to_ipc': {},
    'raw_rows': [],
    'pdf_mappings_loaded': False
}

# ============================================
# HELPER FUNCTIONS (from app.py)
# ============================================

def sanitize_model_output(text: str) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = re.sub(r"(?is)<think[\s\S]*?(|$)", "", text)
    cleaned = re.sub(r"(?is)<analysis[\s\S]*?(</analysis>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<thought[\s\S]*?(</thought>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<system[\s\S]*?(</system>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)<hidden[\s\S]*?(</hidden>|$)", "", cleaned)
    cleaned = re.sub(r"(?is)</?(think|analysis|thought|system|hidden)[^>]*>", "", cleaned)
    return cleaned.strip()

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
        "I can't assist with harming others, evading law enforcement, or committing crimes.\n\n"
        "Under the IPC, unlawful killing is a serious offense (Sections 299–304). If you need lawful information, I can explain the legal framework (e.g., self‑defense provisions, Section 80 accidents) or suggest safe, legal alternatives."
    )

def load_bns_mapping(path: str = "data/bns_mapping.csv"):
    """Load mapping into forward (IPC->BNS) and reverse (BNS->IPC) indexes, plus raw rows."""
    global BNS_MAPPING_CACHE
    if BNS_MAPPING_CACHE['ipc_to_bns'] or BNS_MAPPING_CACHE['bns_to_ipc']:
        return BNS_MAPPING_CACHE['ipc_to_bns'], BNS_MAPPING_CACHE['bns_to_ipc'], BNS_MAPPING_CACHE['raw_rows']

    ipc_to_bns: dict[str, list[dict]] = {}
    bns_to_ipc: dict[str, list[dict]] = {}
    raw_rows: list[dict] = []
    try:
        if not os.path.exists(path):
            print(f"⚠️ Mapping file not found at {path}")
            return {}, {}, []

        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                col1 = str(row.get('ipc_section', row.get('bns_section', ''))).strip()
                col2 = str(row.get('bns_section', row.get('ipc_section', ''))).strip()
                title = str(row.get('title', '')).strip()
                notes = str(row.get('notes', '')).strip()
                
                # Create a consolidated text for searching
                all_text = f" {col1} | {col2} | {title} | {notes} ".replace("\n", " ")
                
                ipc_key, bns_key = "", ""
                
                # Priority 1: Explicit labels
                i_match = re.search(r"(?i)\bIPC\s*(?:Section\s*)?(\d+[A-Z]?)", all_text)
                b_match = re.search(r"(?i)\bBNS\s*(?:Section\s*)?(\d+[A-Z]?)", all_text)
                
                if i_match: ipc_key = i_match.group(1).upper()
                if b_match: bns_key = b_match.group(1).upper()
                
                # Priority 2: Hardcoded high-importance mappings
                if not ipc_key or not bns_key:
                    u_text = all_text.upper()
                    if "MURDER" in u_text and ("302" in u_text or "103" in u_text): 
                        ipc_key, bns_key = ipc_key or "302", bns_key or "103"
                    elif "RAPE" in u_text and ("376" in u_text or "64" in u_text):
                        ipc_key, bns_key = ipc_key or "376", bns_key or "64"
                    elif "THEFT" in u_text and ("378" in u_text or "303" in u_text):
                        ipc_key, bns_key = ipc_key or "378", bns_key or "303"

                # Priority 3: Fallback extraction logic
                if not ipc_key or not bns_key:
                    # Look for numbers in col1/col2 that look like primary sections
                    possible_nums = re.findall(r"\b(\d{1,3}[A-Z]?)\b", all_text)
                    # Filter out small numbers that are likely sub-sections (1, 2, 3) 
                    primary_nums = [n for n in possible_nums if int(re.sub(r"\D", "", n)) > 10]
                    if not primary_nums: primary_nums = possible_nums
                    
                    if not ipc_key and primary_nums:
                        others = [n for n in primary_nums if n != col1]
                        if others: ipc_key = others[0].upper()
                    
                    if not bns_key:
                        bns_key = col1 if col1.isdigit() else (possible_nums[0] if possible_nums else "")

                if not ipc_key or not bns_key: continue
                
                # Final check to avoid the "IPC 2" bug - don't let 1 or 2 be IPC keys if they come from sub-sections
                if ipc_key in ["1", "2"] and not re.search(rf"(?i)IPC\s*{ipc_key}", all_text):
                    continue

                entry = {'ipc_section': ipc_key, 'bns_section': bns_key, 'title': title or "Legal Provision", 'notes': notes}
                raw_rows.append(entry)
                ipc_to_bns.setdefault(ipc_key, []).append(entry)
                bns_to_ipc.setdefault(bns_key, []).append(entry)

        BNS_MAPPING_CACHE.update({'ipc_to_bns': ipc_to_bns, 'bns_to_ipc': bns_to_ipc, 'raw_rows': raw_rows})
        print(f"✅ Loaded {len(ipc_to_bns)} IPC mappings across {len(raw_rows)} rows.")

    except Exception as e:
        print(f"❌ Error loading BNS mapping: {e}")
        return {}, {}, []

    return ipc_to_bns, bns_to_ipc, raw_rows


def load_pdf_mappings():
    """Load additional BNS/IPC mappings from the PDF comparison document."""
    global BNS_MAPPING_CACHE

    if BNS_MAPPING_CACHE.get('pdf_mappings_loaded'):
        return

    import pdfplumber
    import re

    pdf_path = "data/COMPARISON SUMMARY BNS to IPC .pdf"
    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF mapping file not found at {pdf_path}")
        return

    try:
        print("📖 Loading additional IPC↔BNS mappings from PDF...")
        with pdfplumber.open(pdf_path) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

        # Find all IPC mentions with their positions
        ipc_pattern = re.compile(r'(?i)\b(?:IPC|Indian Penal Code)\s*:?\s*(\d+[A-Z]?)')
        bns_pattern = re.compile(r'(?i)\b(?:BNS|Bharatiya Nyaya Sanhita)\s*:?\s*(\d+[A-Z]?)')

        ipc_mentions = []
        for m in ipc_pattern.finditer(all_text):
            ipc_mentions.append((m.group(1).upper(), m.start()))

        bns_mentions = []
        for m in bns_pattern.finditer(all_text):
            bns_mentions.append((m.group(1).upper(), m.start()))

        # Also find "Section XXX" patterns that might be IPC or BNS based on context
        section_pattern = re.compile(r'(?i)\b(?:Section|sec\.?)\s*(\d+[A-Z]?)\b')
        for m in section_pattern.finditer(all_text):
            context_after = all_text[m.end():m.end()+200].lower()
            context_before = all_text[max(0, m.start()-100):m.start()].lower()

            # Determine if this is IPC or BNS based on surrounding context
            if 'ipc' in context_before or 'ipc' in context_after[:50]:
                ipc_mentions.append((m.group(1).upper(), m.start()))
            elif 'bns' in context_before or 'bns' in context_after[:50]:
                bns_mentions.append((m.group(1).upper(), m.start()))

        # Find pairs: when IPC and BNS sections appear within 400 chars of each other
        pairs_found = set()
        for ipc_sec, ipc_pos in ipc_mentions:
            for bns_sec, bns_pos in bns_mentions:
                if abs(ipc_pos - bns_pos) < 400 and ipc_sec != bns_sec:
                    pairs_found.add((ipc_sec, bns_sec))

        # Also look for patterns like "302 ... 103" in same line (table-like)
        line_based = re.findall(r'(?im)^.*?\b(\d+)\b.*?\b(\d+)\b.*$', all_text)
        for match in line_based:
            num1, num2 = match
            if num1 != num2 and 50 <= int(num1) <= 500 and 50 <= int(num2) <= 500:
                # Likely IPC and BNS numbers on same line
                pairs_found.add((num1, num2))

        # Add to cache
        for ipc_sec, bns_sec in pairs_found:
            entry = {
                'ipc_section': ipc_sec,
                'bns_section': bns_sec,
                'title': 'From PDF comparison',
                'notes': 'Mapped from COMPARISON SUMMARY BNS to IPC .pdf'
            }
            BNS_MAPPING_CACHE['raw_rows'].append(entry)
            BNS_MAPPING_CACHE['ipc_to_bns'].setdefault(ipc_sec, []).append(entry)
            BNS_MAPPING_CACHE['bns_to_ipc'].setdefault(bns_sec, []).append(entry)

        # Add to cache
        for ipc_sec, bns_sec in pairs_found:
            entry = {
                'ipc_section': ipc_sec,
                'bns_section': bns_sec,
                'title': 'From PDF comparison',
                'notes': 'Mapped from COMPARISON SUMMARY BNS to IPC .pdf'
            }
            BNS_MAPPING_CACHE['raw_rows'].append(entry)
            BNS_MAPPING_CACHE['ipc_to_bns'].setdefault(ipc_sec, []).append(entry)
            BNS_MAPPING_CACHE['bns_to_ipc'].setdefault(bns_sec, []).append(entry)

        print(f"✅ Loaded {len(pairs_found)} additional IPC↔BNS mappings from PDF")
        BNS_MAPPING_CACHE['pdf_mappings_loaded'] = True

    except Exception as e:
        print(f"❌ Error loading PDF mappings: {e}")
        return

def extract_ipc_sections(text: str):
    if not text:
        return []
    candidates = set()
    # English prefixes - require at least 2 digits or explicit IPC to avoid "2" from "302"
    for m in re.finditer(r"(?i)(?:section|sec\.|s\.)\s*(\d{2,}[A-Z]?|\d[A-Z]?)", text):
        candidates.add(m.group(1).upper())
    # Malayalam prefix: സെക്ഷൻ
    for m in re.finditer(r"സെക്ഷൻ\s*(\d+[A-Z]?)", text):
        candidates.add(m.group(1).upper())
    # Generic IPC suffix/prefix (handles IPC/ഐപിസി)
    for m in re.finditer(r"(?i)\b(?:IPC|ഐപിസി|आईपीसी)\s*(\d+[A-Z]?)", text):
        candidates.add(m.group(1).upper())
    for m in re.finditer(r"(\d+[A-Z]?)\s*(?:IPC|ഐപിസി|आईपीसी)", text):
        candidates.add(m.group(1).upper())
    
    # Clean up single digits unless they are explicitly IPC 1, 2, etc.
    final_candidates = set()
    for c in candidates:
        if len(c) > 1 or re.search(rf"(?i)IPC\s*{c}", text):
            final_candidates.add(c)
            
    print(f"Extracted IPC sections: {final_candidates}")
    return sorted(final_candidates)


def extract_bns_sections(text: str):
    if not text:
        return []
    return sorted({m.group(0) for m in re.finditer(r"\b\d+(?:\.\d+){1,3}\b", text)})

def _ipc_expr_matches(expr: str, sec: str) -> bool:
    if not expr or not sec or not sec.isdigit():
        return False
    s = sec
    expr_l = expr.lower()
    tokens = [t.strip() for t in re.split(r"[,/]|\band\b|\bor\b", expr_l) if t.strip()]
    for t in tokens:
        m = re.match(r"^(\d+)\s*-\s*(\d+)", t)
        if m:
            a, b = m.group(1), m.group(2)
            try:
                if int(a) <= int(s) <= int(b):
                    return True
            except ValueError:
                pass
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
        if t.isdigit() and t == s:
            return True
    return re.search(rf"(?<!\d){re.escape(s)}(?!\d)", expr_l) is not None

def augment_with_bns(answer: str, question: str, use_table: bool = False, language: str = "English") -> str:
    """Augment the answer with IPC ↔ BNS mapping, localized for the target language."""
    ipc_to_bns, bns_to_ipc, raw_rows = load_bns_mapping()
    if not ipc_to_bns and not bns_to_ipc:
        return answer

    # Simple localization dictionary
    loc = {
        "Malayalam": {"head": "IPC ↔ BNS താരതമ്യം", "col1": "IPC വകുപ്പ്", "col2": "BNS വകുപ്പ്", "desc": "വിവരണം"},
        "Hindi": {"head": "IPC ↔ BNS तुलना", "col1": "IPC धारा", "col2": "BNS धारा", "desc": "विवरण"},
        "Tamil": {"head": "IPC ↔ BNS ஒப்பீடு", "col1": "IPC பிரிவு", "col2": "BNS பிரிவு", "desc": "விளக்கம்"},
        "English": {"head": "IPC ↔ BNS Comparison", "col1": "IPC Section", "col2": "BNS Section", "desc": "Description"}
    }
    l = loc.get(language, loc["English"])

    # Only look at sections mentioned in the user's question
    ipc_mentioned = set(extract_ipc_sections(question))
    bns_mentioned = set(extract_bns_sections(question))

    print(f"🌍 MAPPING SIGNAL [{language}]: IPC {ipc_mentioned}, BNS {bns_mentioned}")

    mapping_data = [] # List of tuples: (source, target, title)

    # Only use direct dictionary matches - no comprehensive search
    for sec in sorted(ipc_mentioned):
        for entry in ipc_to_bns.get(sec, [])[:2]:  # Limit to 2 matches per section
            bns = entry.get('bns_section')
            title = entry.get('title', '').strip()
            notes = entry.get('notes', '').strip()
            # Skip low-quality PDF entries
            if title == 'From PDF comparison' or notes == 'Mapped from COMPARISON SUMMARY BNS to IPC .pdf':
                if title == 'From PDF comparison':
                    continue  # Skip pure PDF entries, only use CSV entries
            if bns and bns != 'REMOVED':
                mapping_data.append((f"IPC {sec}", f"BNS {bns}", title or notes or ""))

    for bns in sorted(bns_mentioned):
        for entry in bns_to_ipc.get(bns, [])[:2]:
            ipc = entry.get('ipc_section', '').upper()
            title = entry.get('title', '').strip()
            notes = entry.get('notes', '').strip()
            # Skip low-quality PDF entries
            if title == 'From PDF comparison' or notes == 'Mapped from COMPARISON SUMMARY BNS to IPC .pdf':
                continue
            if ipc:
                mapping_data.append((f"BNS {bns}", f"IPC {ipc}", title or notes or ""))

    if not mapping_data:
        return answer

    # Deduplicate
    seen = set()
    dedup = []
    for item in mapping_data:
        if item not in seen:
            seen.add(item)
            dedup.append(item)

    if use_table:
        # Format as Markdown Table with improved spacing
        table_lines = [
            f"\n\n### {l['head']}",
            f"| {l['col1']} | {l['col2']} | {l['desc']} |",
            "| :--- | :--- | :--- |"
        ]
        for src, tgt, title in dedup[:4]:
            # Clean up content for table format
            src_clean = src.replace("\n", " ").strip()
            tgt_clean = tgt.replace("\n", " ").strip()
            # Truncate long titles for better display
            title_clean = title.replace("\n", " ").strip()
            if len(title_clean) > 80:
                title_clean = title_clean[:77] + "..."

            # Ensure IPC is in the first column for consistency
            if "IPC" in src_clean:
                table_lines.append(f"| {src_clean} | {tgt_clean} | {title_clean} |")
            else:
                table_lines.append(f"| {tgt_clean} | {src_clean} | {title_clean} |")
        mapping_block = "\n".join(table_lines) + "\n"
    else:
        # Format as Bullet Points
        lines = [f"\n\n### {l['head']}"]
        for src, tgt, title in dedup[:10]:
            line = f"- {src} ↔ {tgt}"
            if title:
                line += f" — {title}"
            lines.append(line)
        mapping_block = "\n".join(lines)

    return answer + mapping_block

# ============================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from DOCX: {str(e)}")

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise Exception(f"Error reading TXT file: {str(e)}")

def extract_text_from_document(file_path, filename):
    """Extract text from document based on file type"""
    ext = filename.rsplit('.', 1)[1].lower()

    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    else:
        raise Exception(f"Unsupported file type: {ext}")

def chunk_text(text, chunk_size=3000, overlap=200):
    """Split text into chunks for processing"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def analyze_document_with_ai(text, filename, language='English'):
    """Analyze document using AI"""
    # Chunk text if too long
    if len(text) > 4000:
        chunks = chunk_text(text, chunk_size=4000)
        # Analyze first chunk for summary
        text_to_analyze = chunks[0] + "\n\n[Document continues...]"
    else:
        text_to_analyze = text

    language_instruction = f"IMPORTANT: provide your entire analysis in the {language} language." if language != "English" else ""

    analysis_prompt = f"""Analyze the following legal document and provide a comprehensive analysis.
{language_instruction}

Document: {filename}

Content:
{text_to_analyze}

Please provide:
1. **Summary**: A concise 2-3 sentence overview of the document
2. **Key Points**: List 3-5 main legal points, clauses, or provisions
3. **Legal Issues**: Identify any potential legal concerns or issues (if any)
4. **IPC/BNS References**: List any relevant IPC or BNS sections mentioned or applicable
5. **Suggestions**: Provide 2-3 recommendations or improvements
6. **Risk Assessment**: Assess the legal risk level (Low/Medium/High) with brief justification

Format your response in clear sections with markdown headings."""

    try:
        # Use the LLM to analyze
        _llm = get_llm()
        if not _llm: return {"error": "LLM not initialized"}
        response = _llm.invoke(analysis_prompt)
        analysis_text = response.content if hasattr(response, 'content') else str(response)

        # Clean the response
        analysis_text = sanitize_model_output(analysis_text)

        # Extract IPC/BNS references from the analysis
        ipc_refs = extract_ipc_sections(analysis_text)
        bns_refs = extract_bns_sections(analysis_text)

        # Augment with BNS mapping if IPC sections found
        if ipc_refs and os.getenv("DISABLE_BNS_MAPPING", "0") != "1":
            analysis_text = augment_with_bns(analysis_text, analysis_text, use_table=False)

        return {
            "document_name": filename,
            "analysis": analysis_text,
            "ipc_references": ipc_refs,
            "bns_references": bns_refs,
            "word_count": len(text.split()),
            "char_count": len(text)
        }
    except Exception as e:
        raise Exception(f"Error analyzing document: {str(e)}")

# ============================================
# INITIALIZE AI COMPONENTS
# ============================================

# Global AI State
embeddings = None
db = None
llm = None

def get_llm():
    global llm
    if llm: return llm
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if api_key and base_url:
            llm = ChatOpenAI(
                model=os.getenv("OPENAI_MODEL", "llama-3.3-70b-versatile"),
                temperature=0.3,
                api_key=api_key,
                base_url=base_url,
            )
            return llm
    except: pass
    return None

def get_vector_db():
    global embeddings, db
    if db: return db
    try:
        if not embeddings:
            print("🏛️ JuriSight Supreme Registry: Initializing Transformer Models (Warm-up Phase)...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
        if os.path.exists("ipc_vector_db"):
            print("🏛️ JuriSight Supreme Registry: Loading FAISS Vector Infrastructure...")
            db = FAISS.load_local("ipc_vector_db", embeddings, allow_dangerous_deserialization=True)
            return db
    except Exception as e:
        print(f"❌ Core Error during DB load: {e}")
    return None

# Trigger mapping load (lightweight)
load_bns_mapping()
load_pdf_mappings()  # Load additional mappings from PDF
print("🏛️ Judicial Server v3.1 Online (Lazy Loading Active)")

# Language-specific labels for response structure
RESPONSE_LABELS = {
    "English": {
        "direct_answer": "Direct Answer",
        "section": "Section",
        "description": "Description",
        "punishment": "Punishment",
        "cognizability": "Cognizability",
        "bailable": "Bailable",
        "compoundable": "Compoundable",
        "related_sections": "Related Sections",
        "practical_guidance": "Practical Guidance"
    },
    "Malayalam": {
        "direct_answer": "നേരിട്ട് ഉത്തരം",
        "section": "വകുപ്പ്",
        "description": "വിവരണം",
        "punishment": "ശിക്ഷ",
        "cognizability": "കുറ്റം രജിസ്റ്റർ ചെയ്യാവുന്നത്",
        "bailable": "ജാമ്യം ലഭിക്കുമോ",
        "compoundable": "ഒത്തു തീര്‍പ്പു കാണാമോ",
        "related_sections": "ബന്ധപ്പെട്ട വകുപ്പുകള്‍",
        "practical_guidance": "പ്രായോഗിക നിര്‍ദ്ദേശം"
    },
    "Hindi": {
        "direct_answer": "सीधा उत्तर",
        "section": "धारा",
        "description": "विवरण",
        "punishment": "सजा",
        "cognizability": "संज्ञेयता",
        "bailable": "जमानत योग्य",
        "compoundable": "समझौता योग्य",
        "related_sections": "संबंधित धाराएँ",
        "practical_guidance": "व्यावहारिक मार्गदर्शन"
    },
    "Tamil": {
        "direct_answer": "நேரடி பதில்",
        "section": "பிரிவு",
        "description": "விளக்கம்",
        "punishment": "தண்டனை",
        "cognizability": "குற்றம் பதிவு",
        "bailable": "சிறைக்கைதி",
        "compoundable": "இலக்கு",
        "related_sections": "தொடர்புடைய பிரிவுகள்",
        "practical_guidance": "நடைமுறை வழிகாட்டல்"
    },
    "Kannada": {
        "direct_answer": "ನೇರ ಉತ್ತರ",
        "section": "ವಿಭಾಗ",
        "description": "ವಿವರಣೆ",
        "punishment": "ಶಿಕ್ಷೆ",
        "cognizability": "ಅಪರಾಧ ದಾಖಲೆ",
        "bailable": "ಜಾಮೀನು",
        "compoundable": "ಒಪ್ಪಂದ",
        "related_sections": "ಸಂಬಂಧಿತ ವಿಭಾಗಗಳು",
        "practical_guidance": "ಪ್ರಾಯೋಗಿಕ ಮಾರ್ಗದರ್ಶನ"
    },
    "Bengali": {
        "direct_answer": "সরাসরি উত্তর",
        "section": "ধারা",
        "description": "বিবরণ",
        "punishment": "শাস্তি",
        "cognizability": "সাজাযোগ্যতা",
        "bailable": "জামিনযোগ্য",
        "compoundable": "আপোষযোগ্য",
        "related_sections": "সম্পর্কিত ধারাগুলি",
        "practical_guidance": "বাস্তব নির্দেশনা"
    },
    "Gujarati": {
        "direct_answer": "સીધો જવાબ",
        "section": "ધારો",
        "description": "વર્ણન",
        "punishment": "સજા",
        "cognizability": "ગુનો નોંધવા યોગ્ય",
        "bailable": "જામીનપાત્ર",
        "compoundable": "સમાધાનપાત્ર",
        "related_sections": "સંબંધિત ધારાઓ",
        "practical_guidance": "વ્યવહારિક માર્ಗદર્શન"
    },
    "Marathi": {
        "direct_answer": "थेट उत्तर",
        "section": "कलम",
        "description": "वर्णन",
        "punishment": "शिक्षा",
        "cognizability": "गुन्हा नोंद",
        "bailable": "जामीनपात्र",
        "compoundable": "समेटनीय",
        "related_sections": "संबंधित कलमे",
        "practical_guidance": "व्यावहारिक मार्गदर्शन"
    },
    "Telugu": {
        "direct_answer": "నేరుగా సమాధానం",
        "section": "విభాగం",
        "description": "వివరణ",
        "punishment": "శిక్ష",
        "cognizability": "నేరం నమోదు",
        "bailable": "జామీన్",
        "compoundable": "ఒప్పందం",
        "related_sections": "సంబంధిత విభాగాలు",
        "practical_guidance": "ఆచరణాత్మక మార్గదర్శనం"
    },
    "Punjabi": {
        "direct_answer": "ਸਿੱਧਾ ਜਵਾਬ",
        "section": "ਧਾਰਾ",
        "description": "ਵੇਰਵਾ",
        "punishment": "ਸਜ਼ਾ",
        "cognizability": "ਗੁਨ੍ਹ ਦਰਜ",
        "bailable": "ਜ਼ਮਾਨਤ",
        "compoundable": "ਸਮਝੌਤਾ",
        "related_sections": "ਸੰਬੰਧਿਤ ਧਾਰਾਵਾਂ",
        "practical_guidance": "ਅਮਲੀ ਰਾਹਦਾਰੀ"
    }
}

PROMPT_TEMPLATE = """You are a legal expert assistant. Answer in **{language}** only.

## Response Structure (provide ALL fields):
1. **{label_direct_answer}**: [Clear answer to the user's question. If the user asks about a general crime (e.g., 'theft', 'murder') without a section number, identify the primary IPC/BNS section(s) and answer based on them.]
2. **{label_section}**: [Primary IPC/BNS section number(s) applicable. Use 'IPC XXX (BNS YYY)' format if possible.]

3. **{label_description}**: [Detailed explanation of the offense including: (a) key elements that constitute the offense, (b) who can file a complaint, (c) who can be prosecuted, (d) any exceptions or conditions, (e) distinction from similar offenses - use plain {language} that legal experts and common citizens both understand]
4. **{label_punishment}**: [State the complete punishment including: (a) minimum sentence, (b) maximum sentence, (c) fine amount if any, (d) whether it's bailable or not, (e) whether it's compoundable or not]
5. **Cognizability**: [Whether police can arrest without warrant - Cognizable/Non-cognizable]
6. **Bailable**: [Whether the accused can get bail - Yes/No with explanation]
7. **Compoundable**: [Whether the case can be settled out of court - Yes/No and by whom (complainant only or both parties)]
8. **Related Sections**: [Mention 2-3 closely related IPC/BNS sections that judges often consider together]
9. **Practical Guidance**: [3-4 bullet points on what a common person should do if they encounter this situation - include which police station to approach, what documents to keep, what NOT to do]

## CRITICAL RULES:
- If a specific section is asked, focus on it. If a general crime is asked, use the most common sections (e.g., for theft use IPC 378/379).
- NEVER say 'N/A' for fields 1-9 if information is available in your training or context.
- The mapping section should show the primary section(s) mentioned in your answer.
- Do NOT mix English in non-English responses (except citations).
- For serious offenses, add a note about reporting responsibly.


### Context from Legal Registry:
{context}

### Conversation History:
{chat_history}

### User Inquiry:
{question}

### Response in **{language}**:"""

print("Server initialization complete!")

# ============================================
# ROUTES
# ============================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "JuriSight API is running"})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages with Quantum Streaming v4.0"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        language = data.get('language', 'English')

        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        # Enforce language sanity
        if not language or not language.strip():
            language = 'English'

        # Detect language shift and clear memory if necessary
        session_id = session.get('session_id', str(uuid.uuid4()))
        session['session_id'] = session_id

        prev_lang = session.get('last_language', 'English')
        if prev_lang != language:
            if session_id in conversation_memories:
                del conversation_memories[session_id]
            session['last_language'] = language

        if session_id not in conversation_memories:
            conversation_memories[session_id] = ConversationBufferWindowMemory(
                k=5, memory_key="chat_history", return_messages=True
            )
        memory = conversation_memories[session_id]

        _llm = get_llm()
        _db = get_vector_db()

        def generate():
            nonlocal user_message
            print(f"📡 {session_id[:8]} -> Stream Generator Initialized.")

            if not _llm:
                yield json.dumps({"error": "LLM not configured"}) + "\n"
                return

            if is_harmful_request(user_message):
                yield json.dumps({"chunk": refusal_response(), "done": True}) + "\n"
                return

            # Pre-prompt injection for language
            orig_msg = user_message
            if language == "Malayalam":
                user_message = "നിങ്ങളുടെ മറുപടി മലയാളത്തിൽ മാത്രം എഴുതുക. ഒരു ഇംഗ്ലീഷ് വാക്കും ഉപയോഗിക്കരുത്. നിയമ പദങ്ങൾ മലയാളത്തിൽ തന്നെ എഴുതുക. ചോദ്യം: " + orig_msg
            elif language == "Hindi":
                user_message = "कृपया केवल हिंदी में उत्तर दें - English का प्रयोग न करें। कानूनी शब्द हिंदी में ही लिखें। प्रश्न: " + orig_msg
            elif language == "Tamil":
                user_message = "தயவுசெய்து தமிழில் மட்டும் பதிலளிக்கவும் - English பயன்படுத்த வேண்டாம்। சட்டம் தமிழில் எழுதவும்। கேள்வி: " + orig_msg
            elif language != "English":
                user_message = f"PLEASE RESPOND ENTIRELY IN {language}. DO NOT USE ENGLISH. QUERY: {orig_msg}"

            # 1. Faster Manual Retrieval
            context = ""
            if _db:
                docs = _db.similarity_search(orig_msg, k=4)
                context = "\n---\n".join([d.page_content for d in docs])

            # 2. Manual Memory Load
            history_objs = memory.load_memory_variables({})['chat_history']
            history_text = "\n".join([f"{'User' if 'Human' in str(type(m)) else 'AI'}: {m.content}" for m in history_objs])

            # 3. Final Prompt
            labels = RESPONSE_LABELS.get(language, RESPONSE_LABELS["English"])
            final_p = PROMPT_TEMPLATE.replace("{language}", language) \
                                     .replace("{context}", context) \
                                     .replace("{chat_history}", history_text) \
                                     .replace("{question}", user_message) \
                                     .replace("{label_direct_answer}", labels["direct_answer"]) \
                                     .replace("{label_section}", labels["section"]) \
                                     .replace("{label_description}", labels["description"]) \
                                     .replace("{label_punishment}", labels["punishment"])

            # 4. Stream LLM
            full_answer = ""
            print(f"📡 {session_id[:8]} -> Streaming Response Begin...")
            try:
                for chunk in _llm.stream(final_p):
                    content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                    if content:
                        print(content, end="", flush=True)
                    full_answer += content
                    yield json.dumps({"chunk": content}) + "\n"
            except Exception as e:
                print(f"\n❌ Streaming Error: {e}")
                yield json.dumps({"error": str(e)}) + "\n"
                return
            print(f"\n📡 {session_id[:8]} -> Response Complete ({len(full_answer)} chars)")

            # 5. Post-Process & Finalize
            full_answer_clean = sanitize_model_output(full_answer)
            memory.save_context({"input": orig_msg}, {"output": full_answer_clean})

            # Augment with BNS
            bns_table = ""
            try:
                if os.getenv("DISABLE_BNS_MAPPING", "0") != "1":
                    # Extract sections from both user message and AI answer to be thorough
                    aug_answer = augment_with_bns("", orig_msg + " " + full_answer_clean, use_table=True, language=language)
                    if aug_answer.strip():
                        bns_table = aug_answer
            except: pass

            yield json.dumps({"chunk": bns_table, "done": True}) + "\n"


        return Response(stream_with_context(generate()), mimetype='application/x-ndjson')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in conversation_memories:
            conversation_memories[session_id].clear()
        return jsonify({"message": "Conversation cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analyze-document', methods=['POST'])
def analyze_document():
    """Upload and analyze a legal document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        language = request.form.get('language', 'English')

        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Please upload PDF, DOCX, or TXT files."}), 400

        filename = secure_filename(file.filename)
        analysis_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{analysis_id}_{filename}")
        file.save(file_path)

        try:
            text = extract_text_from_document(file_path, filename)

            if not text or len(text.strip()) < 50:
                os.remove(file_path)
                return jsonify({"error": "Document appears to be empty or too short to analyze"}), 400

            analysis_result = analyze_document_with_ai(text, filename, language)

            document_analyses[analysis_id] = {
                **analysis_result,
                "analysis_id": analysis_id,
                "timestamp": time.time()
            }

            os.remove(file_path)

            return jsonify({
                "analysis_id": analysis_id,
                "document_name": filename,
                "analysis": analysis_result["analysis"],
                "ipc_references": analysis_result["ipc_references"],
                "bns_references": analysis_result["bns_references"],
                "word_count": analysis_result["word_count"],
                "char_count": analysis_result["char_count"]
            })

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Retrieve a previous analysis"""
    try:
        if analysis_id not in document_analyses:
            return jsonify({"error": "Analysis not found"}), 404

        return jsonify(document_analyses[analysis_id])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# E-FILING ASSISTANT ENDPOINTS
# ============================================

from efiling_assistant import get_efiling_assistant

@app.route('/api/efiling/case-types', methods=['GET'])
def get_case_types():
    """Get list of available case types"""
    try:
        assistant = get_efiling_assistant()
        case_types = assistant.get_case_types()
        return jsonify({"case_types": case_types})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/efiling/requirements', methods=['POST'])
def get_requirements():
    """Get requirements for a specific case type"""
    try:
        data = request.json
        case_type = data.get('case_type')

        if not case_type:
            return jsonify({"error": "case_type is required"}), 400

        assistant = get_efiling_assistant()
        requirements = assistant.get_case_requirements(case_type)

        return jsonify({"requirements": requirements})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/efiling/wizard', methods=['POST'])
def efiling_wizard():
    """Get step-by-step guidance for e-filing"""
    try:
        data = request.json
        case_type = data.get('case_type')
        step = data.get('step', 1)

        if not case_type:
            return jsonify({"error": "case_type is required"}), 400

        assistant = get_efiling_assistant()
        steps = assistant.get_filing_steps(case_type)

        if step < 1 or step > len(steps):
            return jsonify({"error": "Invalid step number"}), 400

        guidance_prompt = assistant.get_step_guidance(case_type, step)

        _llm = get_llm()
        if _llm:
            try:
                guidance = _llm.invoke(guidance_prompt)
                guidance_text = guidance.content if hasattr(guidance, 'content') else str(guidance)
            except:
                guidance_text = f"Step {step}: {steps[step-1]}"
        else:
            guidance_text = f"Step {step}: {steps[step-1]}"

        return jsonify({
            "step": step,
            "total_steps": len(steps),
            "step_name": steps[step-1],
            "guidance": guidance_text,
            "has_next": step < len(steps),
            "has_previous": step > 1
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/efiling/court-fee', methods=['POST'])
def calculate_court_fee():
    """Calculate court fee for a case"""
    try:
        data = request.json
        case_type = data.get('case_type')
        claim_amount = data.get('claim_amount')

        if not case_type:
            return jsonify({"error": "case_type is required"}), 400

        assistant = get_efiling_assistant()
        fee_info = assistant.calculate_court_fee(case_type, claim_amount)

        return jsonify(fee_info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/efiling/validate', methods=['POST'])
def validate_filing():
    """Validate if filing is complete"""
    try:
        data = request.json
        case_type = data.get('case_type')
        documents = data.get('documents', [])

        if not case_type:
            return jsonify({"error": "case_type is required"}), 400

        assistant = get_efiling_assistant()
        validation = assistant.validate_filing(case_type, documents)

        return jsonify(validation)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/efiling/checklist', methods=['POST'])
def get_filing_checklist():
    """Get complete filing checklist"""
    try:
        data = request.json
        case_type = data.get('case_type')

        if not case_type:
            return jsonify({"error": "case_type is required"}), 400

        assistant = get_efiling_assistant()
        checklist = assistant.get_checklist(case_type)

        return jsonify(checklist)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================
# RUN SERVER
# ============================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🏛️  JuriSight Server Starting...")
    print("="*50)
    print(f"📍 Server: http://localhost:5000")
    print(f"🔧 API Endpoint: http://localhost:5000/api/chat")
    print("="*50 + "\n")

    app.run(debug=False, host='0.0.0.0', port=5000)