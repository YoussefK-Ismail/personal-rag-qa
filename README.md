# 🤖 Personal RAG Q&A System — Youssef Khaled

> An AI-powered personal assistant that answers questions about me using a RAG (Retrieval-Augmented Generation) pipeline built with LangChain, Groq AI, and Streamlit.

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq_AI-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![Python](https://img.shields.io/badge/Python_3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

## 📸 Screenshots

<!-- Add your screenshots here after uploading them to the repo -->
<!-- Example: -->
<!-- ![App Screenshot](screenshots/home.png) -->
<!-- ![Chat Screenshot](screenshots/chat.png) -->

> 📌 *Screenshots coming soon*

---

## 🚀 Features

- 💬 **Conversational Q&A** — Ask anything about my background, skills, projects, and goals
- 🔍 **RAG Pipeline** — Retrieves relevant context from `about_me.txt` before generating answers
- ⚡ **Groq AI (Llama 3.3)** — Ultra-fast LLM inference, completely free
- 🧠 **TF-IDF Retrieval** — Lightweight semantic search with no GPU or PyTorch required
- 🎨 **Dark UI** — Professional Streamlit interface with custom CSS
- 💡 **Suggested Questions** — One-click question chips for quick exploration
- 🔗 **Portfolio Links** — Direct links to GitHub, Portfolio, and Live Projects

---

## 🏗️ RAG Pipeline Architecture

```
about_me.txt
     │
     ▼  Step 1
 TextLoader ──────────────── Load personal profile
     │
     ▼  Step 2
 RecursiveCharacterTextSplitter ── Split into 500-char chunks
     │
     ▼  Step 3 & 4
 TF-IDF Vectorizer ───────── Convert chunks to vectors (no GPU needed)
     │
     ▼  Step 5
 Cosine Similarity Retriever ── Find top-5 most relevant chunks
     │
     ▼  Step 6
 Groq ChatGroq (Llama-3.3-70b) ── Generate grounded answer
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Streamlit + Custom CSS |
| **LLM** | Groq AI — `llama-3.3-70b-versatile` |
| **RAG Framework** | LangChain |
| **Retrieval** | TF-IDF + Cosine Similarity (Scikit-learn) |
| **Knowledge Base** | Plain text file (`about_me.txt`) |
| **Language** | Python 3.10+ |

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YoussefK-Ismail/personal-rag-qa
cd personal-rag-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Go to [console.groq.com](https://console.groq.com) → Create a free account → Generate API Key

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Enter your Groq API key in the sidebar and start asking! ✅

---

## 📁 Project Structure

```
personal-rag-qa/
│
├── app.py            # Main Streamlit application
├── about_me.txt      # Personal profile (knowledge base)
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## 📦 Requirements

```
streamlit
langchain
langchain-community
langchain-groq
langchain-text-splitters
scikit-learn
```

---

## 💬 Example Questions

| Question | Type |
|----------|------|
| Tell about yourself | General intro |
| What are Youssef's main skills? | Skills |
| Tell me about his projects | Projects |
| Where does Youssef study? | Education |
| What certifications does he have? | Certifications |
| Where does Youssef work? | Experience |
| What are his career goals? | Goals |

---

## 🔗 Links

- 🌐 **Portfolio:** [youssefkhaledportfolio.netlify.app](https://youssefkhaledportfolio.netlify.app)
- ⌨️ **GitHub:** [github.com/YoussefK-Ismail](https://github.com/YoussefK-Ismail)
- 🤖 **Live Q&A App:** [youssef-qna-langchain.streamlit.app](https://youssef-qna-langchain.streamlit.app)
- 🦾 **Live Chatbot:** [youssefkhaled-app-langchain.streamlit.app](https://youssefkhaled-app-langchain-ucoh82nblw2lzjnajupkmn.streamlit.app)

---

## 👨‍💻 Author

**Youssef Khaled Ismail Hassan**
AI & NLP Developer | Alamein International University
Currently @ Ceulla Technologies

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
