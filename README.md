# 📚 PDF RAG Chatbot

> A production-style Retrieval-Augmented Generation (RAG) chatbot that lets you have an intelligent conversation with any PDF document — powered by LangChain, FAISS, CrossEncoder reranking, and Groq Llama 3.3.

---

## 📌 Short Description

PDF RAG Chatbot is a document-grounded question answering system built with a multi-stage retrieval pipeline. It goes beyond simple vector similarity search by combining **MultiQueryRetriever**, **CrossEncoder reranking**, and **strict prompt grounding** to deliver accurate, faithful answers from your uploaded PDF — while refusing to answer questions that fall outside the document.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔍 **MultiQueryRetriever** | Generates multiple query variants to maximize recall from the vector store |
| 📐 **CrossEncoder Reranking** | Re-scores retrieved chunks for precision before passing context to the LLM |
| 🧠 **Groq Llama 3.3 70B** | Fast, high-quality LLM inference via Groq's API |
| 📄 **Document-Only Grounding** | Strict prompt engineering prevents the LLM from using outside knowledge |
| 🚫 **Out-of-Scope Detection** | Few-shot examples in the prompt teach the LLM when to refuse |
| 👋 **Casual Conversation Routing** | Greetings and small talk are detected and handled before retrieval runs |
| 📚 **Bibliography Exclusion** | References and bibliography sections are automatically removed before chunking |
| 📊 **Real-Time Answer Evaluation** | Every answer is evaluated for Faithfulness, Groundedness, Hallucination Risk, and Quality |
| 🔎 **Source Page Attribution** | Retrieved chunk source pages are displayed with every answer |
| 🧪 **LangSmith Tracing** | Full pipeline observability — retrieval, reranking, generation, and evaluation are all traced |
| 💬 **Chat History** | Conversation history is preserved across the session |

---

## 🎬 Demo

> 📹 **Demo video coming soon**
>
> ![App Demo](video/demo.mp4)

---

## 📸 Screenshots

> **Upload and Chat**
> <img width="1917" height="962" alt="chat" src="https://github.com/user-attachments/assets/2ea4e6d7-0f4b-4981-8a58-5df857e11569" />


> **Answer with Source Pages**
> <img width="1917" height="907" alt="answer" src="https://github.com/user-attachments/assets/73443a20-17f1-42b5-a985-f7b09a25351f" />


> **Retrieval Evaluation Panel**
> <img width="1915" height="902" alt="evaluation" src="https://github.com/user-attachments/assets/9c6b7c32-b35a-4a46-be84-93e641b9c8f3" />



> **Retrieved Chunks Expander**
> <img width="1917" height="910" alt="retrieval" src="https://github.com/user-attachments/assets/28d74229-0ff9-46cf-bc61-f38ce3468106" />



---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                    (Streamlit Web App)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ User Question
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  Casual Conversation Gate                   │
│         (Fast path: exact match → LLM classifier)           │
│     Greetings / small talk → skip retrieval entirely        │
└─────────────────────┬───────────────────────────────────────┘
                      │ Real Question
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RETRIEVAL STAGE                          │
│                                                             │
│   MultiQueryRetriever (LangChain)                           │
│   ├── Generates N query variants from the user question     │
│   ├── Runs each variant against FAISS vector store          │
│   └── Returns up to k=20 deduplicated chunks                │
│                                                             │
│   Embeddings: HuggingFace all-MiniLM-L6-v2                  │
└─────────────────────┬───────────────────────────────────────┘
                      │ 20 candidate chunks
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    RERANKING STAGE                          │
│                                                             │
│   CrossEncoder: ms-marco-MiniLM-L-6-v2                      │
│   ├── Scores each (question, chunk) pair                    │
│   ├── Sorts chunks by relevance score descending            │
│   └── Keeps top 5 chunks for context                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ Top 5 reranked chunks
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   GENERATION STAGE                          │
│                                                             │
│   Groq Llama 3.3 70B                                        │
│   ├── Strict document-only prompt                           │
│   ├── Few-shot examples for boundary cases                  │
│   └── Refuses if topic not in document                      │
└─────────────────────┬───────────────────────────────────────┘
                      │ Answer
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION STAGE                          │
│                                                             │
│   Groq Llama 3.3 70B (evaluator)                            │
│   ├── Faithfulness: answer supported by context?            │
│   ├── Groundedness: stays within context?                   │
│   ├── Hallucination Risk: outside info introduced?          │
│   └── Answer Quality: 1–10 score with reason                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
              Response + Source Pages
              + Evaluation Dashboard
              displayed to User
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **LLM** | Groq — Llama 3.3 70B Versatile |
| **Embeddings** | HuggingFace — `sentence-transformers/all-MiniLM-L6-v2` |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Retriever** | LangChain MultiQueryRetriever |
| **Reranker** | CrossEncoder — `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF Loader** | LangChain PyPDFLoader |
| **Text Splitter** | LangChain RecursiveCharacterTextSplitter |
| **Tracing** | LangSmith |
| **Orchestration** | LangChain |
| **Environment** | Python `dotenv` |

---

## 📁 Project Structure

```
pdf-rag-chatbot/
│
├── app.py              # Main Streamlit application
├── logger.py           # Logger configuration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── temp/               # Temporary PDF storage (git-ignored)
```

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/yourusername/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

**5. Run the app**

```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory with the following:

```env
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=pdf-rag-chatbot
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key for Llama inference |
| `LANGCHAIN_API_KEY` | ✅ Yes | LangSmith API key for tracing |
| `LANGCHAIN_TRACING_V2` | ✅ Yes | Enables LangSmith tracing |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name |

Get your free API keys:
- Groq: https://console.groq.com
- LangSmith: https://smith.langchain.com

---

## 🚀 Usage Instructions

1. Open the app in your browser at `http://localhost:8501`
2. Upload any PDF using the sidebar uploader
3. Wait for the PDF to be processed — you'll see a success message
4. Type your question in the chat input
5. View the answer, source pages, retrieval evaluation, and retrieved chunks

---

## 💡 Example Questions

Once you upload a PDF, try questions like:

```
What are the main findings of this document?
Summarize the key points from chapter 3.
What methodology was used in this study?
What are the limitations mentioned by the authors?
What does the document say about [specific topic]?
```

---

## 🔬 How the RAG Pipeline Works

### 1. PDF Ingestion
- The uploaded PDF is loaded using `PyPDFLoader`
- The **references and bibliography section is automatically detected and removed** before chunking — preventing citation fragments from polluting retrieval
- The cleaned text is split into chunks of 1000 characters with 200-character overlap using `RecursiveCharacterTextSplitter`
- Chunks are embedded using `all-MiniLM-L6-v2` and stored in a **FAISS** vector index

### 2. Retrieval Strategy
Standard RAG uses a single query against the vector store — this often misses relevant chunks if the user's phrasing doesn't match the document's phrasing.

This project uses **MultiQueryRetriever**, which:
- Automatically generates multiple rephrased versions of the user's question
- Runs each variant against the FAISS index
- Deduplicates and merges results
- Returns up to 20 candidate chunks

This significantly improves **recall** — finding relevant content even when the exact wording differs.

### 3. Reranking Strategy
Having 20 candidate chunks is too much context to pass directly to the LLM. But picking only the top-5 by vector similarity score is unreliable — bi-encoder embeddings optimise for speed, not precision.

**CrossEncoder reranking** solves this:
- Each candidate chunk is scored against the original question as a pair: `(question, chunk)`
- CrossEncoders read both inputs jointly, producing a precise relevance score
- Chunks are sorted by score and the top 5 are selected
- Rerank scores are logged and displayed in the Retrieved Chunks expander

This improves **precision** — ensuring the LLM only sees the most relevant content.

### 4. Prompting Strategy
The LLM receives a strict prompt with:
- A document-only grounding rule — no outside knowledge permitted
- A distinction rule — topics "partially mentioned" are answered from context; topics only appearing as passing words are refused
- **Few-shot examples** showing the exact boundary between answering and refusing
- The top 5 reranked chunks as context

### 5. Casual Conversation Routing
Before retrieval runs, every message is classified:
- **Fast path**: exact match against a set of known casual phrases (`"hi"`, `"thanks"`, `"bye"`, etc.)
- **Slow path**: LLM classifier for combinations (`"hi who are you?"`, `"hey thanks"`)

If classified as casual, the pipeline exits immediately — no retrieval, no reranking, no evaluation cards shown.

---

## 📊 Evaluation Methodology

Every answer is automatically evaluated by a second LLM call using the following criteria:

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer supported by the retrieved context? |
| **Groundedness** | Does the answer stay within the provided context? |
| **Hallucination Risk** | Did the LLM introduce information not found in the context? |
| **Answer Quality** | Overall quality score from 1 to 10 with a one-sentence reason |

Results are displayed as color-coded badges (green = good, red = bad) with a progress bar for the quality score — visible after every answer.

Retrieval statistics are also shown:
- Retrieved Chunks count
- Unique Pages count
- Context Length (characters)
- Answer Length (words)
- Page Diversity %

---
## 🎯 Learning Outcomes

Through this project, I gained hands-on experience with:

- Retrieval-Augmented Generation (RAG)
- Semantic Search using FAISS
- HuggingFace Embeddings
- CrossEncoder Reranking
- LangChain Retrieval Pipelines
- Prompt Engineering
- Streamlit Application Development
- LLM Evaluation and Grounding
- LangSmith Tracing and Debugging

---
## 👤 Author

**Gowthami**
- GitHub: [@Gowthami-bot](https://github.com/Gowthami-bot)
- LinkedIn: [gowthami-v-s-a-44099a328](www.linkedin.com/in/gowthami-v-s-a-44099a328)

---

<div align="center">

If this project helped you or you found it interesting, please consider giving it a ⭐ on GitHub.
It helps others discover the project and keeps me motivated to improve it.

**[⭐ Star this repository](https://github.com/Gowthami-bot/pdf-rag-chatbot)**

</div>
