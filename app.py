"""
Youssef Khaled — RAG Personal Q&A System
Streamlit Interface — No PyTorch, No langchain.chains
"""

import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Youssef Khaled | AI Portfolio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #0a0a0f;
    --surface: #111118;
    --card:    #16161f;
    --border:  #2a2a3a;
    --accent:  #7c6af7;
    --accent2: #3ecfcf;
    --text:    #e8e8f0;
    --muted:   #6b6b80;
}
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stMainBlockContainer"] { padding: 2rem 3rem !important; max-width: 1100px !important; }

.hero {
    background: linear-gradient(135deg, #12121c 0%, #1a1030 50%, #0d1a2a 100%);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50%; right: -20%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,106,247,0.15) 0%, transparent 70%);
}
.hero-name {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #fff 0%, var(--accent) 50%, var(--accent2) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.1; margin-bottom: 0.3rem;
}
.hero-title {
    font-size: 1rem; color: var(--accent2); font-family: 'Space Mono', monospace;
    letter-spacing: 3px; text-transform: uppercase; margin-bottom: 1rem;
}
.hero-desc { color: var(--muted); font-size: 0.95rem; max-width: 600px; line-height: 1.7; }
.badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.2rem; }
.badge {
    background: rgba(124,106,247,0.12); border: 1px solid rgba(124,106,247,0.3);
    color: var(--accent); padding: 0.25rem 0.8rem; border-radius: 999px;
    font-size: 0.75rem; font-family: 'Space Mono', monospace; letter-spacing: 1px;
}
.badge.teal { background: rgba(62,207,207,0.1); border-color: rgba(62,207,207,0.3); color: var(--accent2); }

.section-title {
    font-size: 0.7rem; font-family: 'Space Mono', monospace;
    letter-spacing: 4px; text-transform: uppercase; color: var(--muted);
    margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
}
.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.stat-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.2rem; text-align: center; }
.stat-num {
    font-size: 1.8rem; font-weight: 800;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.stat-label { font-size: 0.7rem; color: var(--muted); font-family: 'Space Mono', monospace; letter-spacing: 2px; text-transform: uppercase; margin-top: 0.2rem; }

.chat-wrap {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 1.5rem;
    min-height: 400px; max-height: 520px; overflow-y: auto; margin-bottom: 1rem;
}
.msg-user { display: flex; justify-content: flex-end; margin-bottom: 1rem; }
.msg-user .bubble {
    background: linear-gradient(135deg, var(--accent), #5a4fd4);
    color: #fff; padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 4px 18px; max-width: 70%;
    font-size: 0.9rem; line-height: 1.6; box-shadow: 0 4px 20px rgba(124,106,247,0.3);
}
.msg-ai { display: flex; justify-content: flex-start; margin-bottom: 1rem; gap: 0.6rem; align-items: flex-start; }
.ai-avatar {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, var(--accent2), #2a9d9d);
    border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 1rem; flex-shrink: 0;
}
.msg-ai .bubble {
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    padding: 0.8rem 1.2rem; border-radius: 4px 18px 18px 18px; max-width: 75%;
    font-size: 0.9rem; line-height: 1.7;
}
.welcome-msg { text-align: center; padding: 3rem 1rem; color: var(--muted); }
.welcome-msg .icon { font-size: 3rem; margin-bottom: 1rem; }
.source-note {
    font-size: 0.72rem; color: var(--muted); font-family: 'Space Mono', monospace;
    margin-top: 0.4rem; padding-left: 0.5rem; border-left: 2px solid var(--accent2);
}

.stTextInput input {
    background: var(--card) !important; border: 1px solid var(--border) !important;
    border-radius: 12px !important; color: var(--text) !important;
    padding: 0.8rem 1rem !important; font-size: 0.9rem !important;
}
.stTextInput input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(124,106,247,0.15) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5a4fd4) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-weight: 700 !important; padding: 0.6rem 1.5rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(124,106,247,0.4) !important; }

.sidebar-link {
    display: flex; align-items: center; gap: 0.6rem;
    background: rgba(255,255,255,0.04); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.7rem 1rem; margin-bottom: 0.6rem;
    text-decoration: none; color: var(--text) !important; font-size: 0.85rem;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# RAG Pipeline — Pure Python, no PyTorch, no langchain.chains
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_pipeline(groq_key: str):
    # Step 1 — Load
    loader = TextLoader("about_me.txt", encoding="utf-8")
    docs   = loader.load()

    # Step 2 — Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks   = splitter.split_documents(docs)
    texts    = [c.page_content for c in chunks]

    # Step 3 & 4 — TF-IDF index (no PyTorch!)
    vectorizer = TfidfVectorizer(stop_words='english')
    matrix     = vectorizer.fit_transform(texts)

    # Step 6 — Groq LLM
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=groq_key)

    return texts, vectorizer, matrix, llm


def ask(query, texts, vectorizer, matrix, llm, k=5):
    # Rephrase vague queries to be more specific about Youssef
    vague_queries = ["tell about yourself", "tell me about yourself", "who are you",
                     "introduce yourself", "about you", "who is youssef", "about youssef"]
    if query.strip().lower() in vague_queries or len(query.strip().split()) <= 4:
        search_query = "Youssef Khaled background education experience skills projects"
    else:
        search_query = query

    # Step 5 — Retrieve top-k chunks
    q_vec   = vectorizer.transform([search_query])
    scores  = cosine_similarity(q_vec, matrix).flatten()
    top_k   = scores.argsort()[-k:][::-1]
    context = "\n\n".join([texts[i] for i in top_k])

    # Step 6 — Generate grounded answer
    prompt = f"""You are Youssef Khaled's personal AI assistant. The user is asking about Youssef.
Use the context below to answer. Always assume questions are about Youssef Khaled.
If asked to introduce or describe Youssef, give a full friendly summary based on the context.
Only say information is missing if it truly is not in the context at all.

Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return response.content


# ══════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════
if "messages"       not in st.session_state: st.session_state.messages       = []
if "pipeline_ready" not in st.session_state: st.session_state.pipeline_ready = False
if "pending_q"      not in st.session_state: st.session_state.pending_q      = None


# ══════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 1rem;">
        <div style="width:80px;height:80px;background:linear-gradient(135deg,#7c6af7,#3ecfcf);
                    border-radius:50%;margin:0 auto 1rem;display:flex;align-items:center;
                    justify-content:center;font-size:2rem;">🤖</div>
        <div style="font-size:1.1rem;font-weight:800;color:#e8e8f0;">Youssef Khaled</div>
        <div style="font-size:0.7rem;color:#6b6b80;font-family:'Space Mono',monospace;
                    letter-spacing:2px;margin-top:0.2rem;">AI · NLP · DEVELOPER</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔑 Groq API Key</div>', unsafe_allow_html=True)
    groq_key = st.text_input("API Key", type="password", placeholder="gsk_...", label_visibility="collapsed")

    if groq_key:
        if not st.session_state.pipeline_ready:
            with st.spinner("⚙️ Loading pipeline..."):
                try:
                    texts, vectorizer, matrix, llm = build_pipeline(groq_key)
                    st.session_state.texts      = texts
                    st.session_state.vectorizer = vectorizer
                    st.session_state.matrix     = matrix
                    st.session_state.llm        = llm
                    st.session_state.pipeline_ready = True
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.pipeline_ready:
            st.markdown("""
            <div style="background:rgba(62,207,207,0.1);border:1px solid rgba(62,207,207,0.3);
                        border-radius:10px;padding:0.6rem 1rem;font-size:0.8rem;color:#3ecfcf;
                        font-family:'Space Mono',monospace;text-align:center;margin-top:0.5rem;">
                ✅ RAG PIPELINE READY
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(124,106,247,0.08);border:1px solid rgba(124,106,247,0.2);
                    border-radius:10px;padding:0.8rem;font-size:0.78rem;color:#6b6b80;margin-top:0.5rem;">
            Get a free key at<br>
            <a href="https://console.groq.com" target="_blank" style="color:#7c6af7;">console.groq.com</a>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:1.5rem;">🔗 Links</div>', unsafe_allow_html=True)
    st.markdown("""
    <a class="sidebar-link" href="https://github.com/YoussefK-Ismail" target="_blank">⌨️ GitHub Profile</a>
    <a class="sidebar-link" href="https://youssefkhaledportfolio.netlify.app" target="_blank">🌐 Portfolio</a>
    <a class="sidebar-link" href="https://youssef-qna-langchain.streamlit.app" target="_blank">🤖 Q&A App (Live)</a>
    """, unsafe_allow_html=True)

    if st.session_state.messages:
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">💬 Chat</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# Main Page
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-name">Youssef Khaled</div>
    <div class="hero-title">AI & NLP Developer</div>
    <div class="hero-desc">Ask me anything about Youssef — his skills, projects, education,
    experience, or goals. Powered by LangChain + Groq AI + TF-IDF RAG pipeline.</div>
    <div class="badge-row">
        <span class="badge">LangChain</span>
        <span class="badge">Groq AI</span>
        <span class="badge">TF-IDF</span>
        <span class="badge teal">RAG System</span>
        <span class="badge teal">Llama3</span>
        <span class="badge">Scikit-learn</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-grid">
    <div class="stat-card"><div class="stat-num">3+</div><div class="stat-label">Years Exp</div></div>
    <div class="stat-card"><div class="stat-num">3</div><div class="stat-label">Live Projects</div></div>
    <div class="stat-card"><div class="stat-num">13+</div><div class="stat-label">Certifications</div></div>
</div>
""", unsafe_allow_html=True)

# Suggested questions
st.markdown('<div class="section-title">💡 Suggested Questions</div>', unsafe_allow_html=True)
suggestions = [
    "What are Youssef's main skills?",
    "Tell me about his projects",
    "Where does Youssef study?",
    "What certifications does he have?",
    "What are his career goals?",
    "Where does Youssef work?",
]
cols = st.columns(3)
for i, s in enumerate(suggestions):
    with cols[i % 3]:
        if st.button(s, key=f"chip_{i}", use_container_width=True):
            st.session_state.pending_q = s
            st.rerun()

# Chat area
st.markdown('<div class="section-title">💬 Chat</div>', unsafe_allow_html=True)
chat_html = '<div class="chat-wrap">'
if not st.session_state.messages:
    chat_html += """
    <div class="welcome-msg">
        <div class="icon">🤖</div>
        <p>Hi! I'm Youssef's AI assistant.<br>
        Ask me anything about his background, skills, projects, or goals.</p>
    </div>"""
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_html += f'<div class="msg-user"><div class="bubble">{msg["content"]}</div></div>'
        else:
            chat_html += f"""
            <div class="msg-ai">
                <div class="ai-avatar">🤖</div>
                <div>
                    <div class="bubble">{msg["content"]}</div>
                    <div class="source-note">▸ Retrieved from about_me.txt via RAG</div>
                </div>
            </div>"""
chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# Input
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("q", placeholder="Ask anything about Youssef Khaled...", label_visibility="collapsed", key="user_input")
with col2:
    send = st.button("Send →", use_container_width=True)

# Handle suggestion click
if st.session_state.pending_q:
    user_input = st.session_state.pending_q
    st.session_state.pending_q = None
    send = True

# Process question
if send and user_input:
    if not st.session_state.pipeline_ready:
        st.warning("⚠️ Please enter your Groq API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("🔍 Searching profile..."):
            try:
                answer = ask(
                    user_input,
                    st.session_state.texts,
                    st.session_state.vectorizer,
                    st.session_state.matrix,
                    st.session_state.llm,
                )
            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;color:#3a3a50;
            font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:2px;">
    BUILT WITH LANGCHAIN · GROQ · SCIKIT-LEARN · STREAMLIT
</div>
""", unsafe_allow_html=True)
