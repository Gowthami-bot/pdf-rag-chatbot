import os
import hashlib
import streamlit as st
import re
from langsmith import traceable
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from logger import logger

# =====================================================
# ENV
# =====================================================

load_dotenv()

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.caption("Upload a PDF and chat with it")

# =====================================================
# CACHE MODELS
# =====================================================

@st.cache_resource
def load_embeddings():
    logger.info("Loading HuggingFace Embeddings")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_reranker():
    logger.info("Loading Reranker")
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_llm():
    logger.info("Loading Groq LLM")
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

reranker = load_reranker()
embeddings = load_embeddings()
llm = load_llm()

# =====================================================
# TRACEABLE FUNCTIONS
# =====================================================

@traceable
def generate_answer(prompt):
    return llm.invoke(prompt)

@traceable
def evaluate_answer(prompt):
    return llm.invoke(prompt)

@traceable
def classify_message(text):
    return llm.invoke(text)

@traceable(name="retrieval")
def retrieve_docs(question, retriever):
    return retriever.invoke(question)

@traceable(name="reranking")
def rerank_docs(pairs):
    return reranker.predict(pairs)

# =====================================================
# HELPERS
# =====================================================

def parse_evaluation(text):
    fields = {}
    for key in [
        "Faithfulness",
        "Groundedness",
        "Hallucination Risk",
        "Answer Quality",
        "Reason"
    ]:
        m = re.search(
            rf"{key}:\s*(.+?)(?=\n\w|\Z)",
            text,
            re.DOTALL | re.IGNORECASE
        )
        fields[key] = m.group(1).strip() if m else "—"
    return fields

# =====================================================
# HELPERS — PDF CLEANING
# =====================================================

def remove_references_section(docs):
    """
    Removes bibliography/references section from loaded PDF pages.
    Once the references heading is found, all subsequent content is dropped.
    """
    

    reference_patterns = [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
        r"^\s*works cited\s*$",
        r"^\s*\d+\.\s*references\s*$",
    ]

    cleaned_docs = []
    references_found = False

    for doc in docs:
        if references_found:
            break

        lines = doc.page_content.split("\n")
        clean_lines = []

        for line in lines:
            if any(
                re.match(pattern, line.strip(), re.IGNORECASE)
                for pattern in reference_patterns
            ):
                references_found = True
                logger.info(
                    f"References section found on page "
                    f"{doc.metadata.get('page', 0) + 1} — truncating"
                )
                break
            clean_lines.append(line)

        if clean_lines:
            doc.page_content = "\n".join(clean_lines)
            cleaned_docs.append(doc)

    logger.info(
        f"Pages after removing references: "
        f"{len(cleaned_docs)} / {len(docs)}"
    )
    return cleaned_docs

def badge_color(val):
    v = val.lower()
    if v == "high": return "background:#EAF3DE;color:#27500A"
    if v == "low":  return "background:#FCEBEB;color:#791F1F"
    return "background:#F1EFE8;color:#5F5E5A"

def hall_badge_color(val):
    v = val.lower()
    if v == "low":  return "background:#EAF3DE;color:#27500A"
    if v == "high": return "background:#FCEBEB;color:#791F1F"
    return "background:#F1EFE8;color:#5F5E5A"

CASUAL_PHRASES = {
    "hi", "hello", "hey", "hii", "helo",
    "good morning", "good evening", "good afternoon",
    "how are you", "how r u", "how are u",
    "who are you", "what are you", "what is your name",
    "thanks", "thank you", "thank u", "ty",
    "bye", "goodbye", "see you", "ok", "okay",
    "cool", "nice", "great", "awesome", "got it"
}

def is_casual(text):
    # fast path — exact match
    normalized = text.strip().lower().rstrip("!?.")
    if normalized in CASUAL_PHRASES:
        return True

    # slow path — LLM classifier for anything not in the set
    classification_prompt = f"""Classify this message as CASUAL or QUESTION.
CASUAL: greetings, small talk, thanks, farewells, "who are you", combinations of these.
QUESTION: anything requiring document search.
Message: "{text}"
Reply with only CASUAL or QUESTION."""

    result = classify_message(classification_prompt)
    logger.info(f"LLM classification: {result}")
    return result == "CASUAL"

# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =====================================================
# PDF PROCESSING
# =====================================================

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_hash = hashlib.md5(file_bytes).hexdigest()

    if current_hash != st.session_state.pdf_hash:
        logger.info(f"New PDF Uploaded: {uploaded_file.name}")

        os.makedirs("temp", exist_ok=True)
        pdf_path = os.path.join("temp", uploaded_file.name)

        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        with st.spinner("Processing PDF..."):
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            logger.info(f"Pages Loaded: {len(docs)}")

            docs = remove_references_section(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)
            logger.info(f"Chunks Created: {len(chunks)}")

            vectorstore = FAISS.from_documents(chunks, embeddings)
            st.session_state.vectorstore = vectorstore
            st.session_state.pdf_hash = current_hash
            logger.info("FAISS Vectorstore Created")

        st.sidebar.success(f"PDF Loaded ({len(docs)} pages)")

# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "pages" in message and message["pages"]:
            st.caption(
                f"📄 Retrieved from pages: {', '.join(map(str, message['pages']))}"
            )

# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input("Ask a question about the uploaded PDF...")

# =====================================================
# RAG PIPELINE
# =====================================================

if question:

    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
        st.stop()

    # ----------------------------------------
    # USER MESSAGE
    # ----------------------------------------

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    logger.info(f"Question: {question}")

     # ----------------------------------------
    # CASUAL CONVERSATION GATE
    # ----------------------------------------

    if is_casual(question):
        casual_prompt = f"""You are a helpful PDF assistant chatbot.
The user said: "{question}"
Respond briefly and naturally. Do not mention the PDF or document."""

        with st.chat_message("assistant"):
            casual_response = generate_answer(casual_prompt).content
            st.markdown(casual_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": casual_response,
            "pages": []
        })

        logger.info("Casual message — skipped retrieval pipeline")
        st.stop()

    # ----------------------------------------
    # RETRIEVAL
    # ----------------------------------------

    retriever = MultiQueryRetriever.from_llm(
        retriever=st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 20}
        ),
        llm=llm
    )
    docs = retrieve_docs(question, retriever)
    logger.info(f"Retrieved {len(docs)} chunks before reranking")

    # ----------------------------------------
    # RERANKING
    # ----------------------------------------

    pairs = [(question, doc.page_content) for doc in docs]
    scores = list(rerank_docs(pairs))

    scored_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (doc, score) in enumerate(scored_docs, 1):
        doc.metadata["rerank_score"] = float(score)
        logger.info(
            f"Rank {rank} | Page {doc.metadata.get('page', 0) + 1} "
            f"| Score {score:.4f}"
        )

    TOP_K = 5
    top_docs = [doc for doc, score in scored_docs[:TOP_K]]
    logger.info(f"Using top {TOP_K} reranked chunks for context")

    # ----------------------------------------
    # CONTEXT
    # ----------------------------------------

    if not top_docs:
        context = "No relevant context found."
        pages = []
    else:
        context = "\n\n".join(
            doc.page_content for doc in top_docs
        )
        pages = sorted(
            {doc.metadata.get("page", 0) + 1 for doc in top_docs}
        )

    logger.info(f"Source Pages: {pages}")

    # ----------------------------------------
    # PROMPT
    # ----------------------------------------

    prompt = f"""You are a document assistant. Answer the user's question using only the document context provided below.

RULES:
- Use ONLY the document context. No outside knowledge.
- If the document discusses the topic even partially, summarize what it says.
- If the document has no relevant discussion about the topic at all, respond with exactly:
  "This information is not available in the uploaded document."
- Never invent or assume facts not present in the document.
- Use bullet points when listing multiple items.
- For greetings, thanks, or casual conversation respond briefly and naturally.



--------------------------------
CONTEXT
--------------------------------

{context}

--------------------------------
USER QUESTION
--------------------------------

{question}

--------------------------------
ANSWER
--------------------------------
"""

    # ----------------------------------------
    # GENERATE ANSWER
    # ----------------------------------------

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = generate_answer(prompt)
            answer = response.content

            # Hide pages if answer is a refusal
            if "not available in the uploaded document" in answer.lower():
                pages = []

            evaluation_prompt = f"""You are an expert RAG evaluator.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Evaluate using these criteria:
1. Faithfulness - Is the answer supported by the context?
2. Groundedness - Does the answer stay within the provided context?
3. Hallucination Risk - Did the answer introduce information not found in the context?
4. Answer Quality - Score from 1 to 10.

Return ONLY in this exact format:
Faithfulness: <High/Low>
Groundedness: <High/Low>
Hallucination Risk: <High/Low>
Answer Quality: <number>/10
Reason: <one sentence>
"""
            evaluation_response = evaluate_answer(evaluation_prompt)
            evaluation = evaluation_response.content
            logger.info(f"Evaluation:\n{evaluation}")

            retrieved_chunks = len(top_docs)
            unique_pages = len(
                {doc.metadata.get("page", 0) + 1 for doc in top_docs}
            )
            page_diversity = (
                unique_pages / retrieved_chunks * 100
            ) if retrieved_chunks > 0 else 0
            context_length = len(context)
            answer_length = len(answer.split())

        st.markdown(answer)

        if pages:
            st.caption(
                f"📄 Retrieved from pages: {', '.join(map(str, pages))}"
            )
        else:
            st.caption("📄 Sources: None")

        st.subheader("📊 Retrieval Evaluation")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Retrieved Chunks", retrieved_chunks)
            st.metric("Unique Pages", unique_pages)
        with col2:
            st.metric("Context Length", context_length)
            st.metric("Answer Length", answer_length)
            st.metric("Page Diversity %", f"{page_diversity:.0f}")

        ev = parse_evaluation(evaluation)
        score_raw = re.search(r"(\d+)", ev["Answer Quality"])
        score = int(score_raw.group(1)) if score_raw else 0
        bar_pct = score * 10

        st.subheader("📋 Answer evaluation")
        st.markdown(f"""
<style>
.ev-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:14px}}
.ev-card{{background:var(--secondary-background-color);border-radius:8px;padding:12px 14px}}
.ev-lbl{{font-size:11px;color:gray;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px}}
.ev-val{{font-size:14px;font-weight:500}}
.badge{{display:inline-block;font-size:11px;font-weight:500;padding:2px 9px;border-radius:99px}}
.bar-bg{{display:flex;flex:1;height:6px;background:#e2e8f0;border-radius:99px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:99px;background:#1D9E75}}
.reason-box{{border-left:3px solid #ccc;padding:8px 14px;font-size:13px;color:gray;line-height:1.6}}
</style>
<div class="ev-grid">
  <div class="ev-card">
    <div class="ev-lbl">Faithfulness</div>
    <div class="ev-val">
      <span class="badge" style="{badge_color(ev['Faithfulness'])}">{ev["Faithfulness"]}</span>
    </div>
  </div>
  <div class="ev-card">
    <div class="ev-lbl">Groundedness</div>
    <div class="ev-val">
      <span class="badge" style="{badge_color(ev['Groundedness'])}">{ev["Groundedness"]}</span>
    </div>
  </div>
  <div class="ev-card">
    <div class="ev-lbl">Hallucination</div>
    <div class="ev-val">
      <span class="badge" style="{hall_badge_color(ev['Hallucination Risk'])}">{ev["Hallucination Risk"]}</span>
    </div>
  </div>
  <div class="ev-card">
    <div class="ev-lbl">Answer quality</div>
    <div class="ev-val">
      <div style="display:flex;align-items:center;gap:8px">
        <span style="font-size:15px;font-weight:500">{score}/10</span>
        <div class="bar-bg"><div class="bar-fill" style="width:{bar_pct}%"></div></div>
      </div>
    </div>
  </div>
</div>
<p style="font-size:12px;font-weight:500;color:gray;margin:0 0 6px">Reason</p>
<div class="reason-box">{ev["Reason"]}</div>
""", unsafe_allow_html=True)

        with st.expander("🔍 Retrieved Chunks"):
            for i, doc in enumerate(top_docs, start=1):
                st.markdown(f"### Chunk {i}")
                st.write(f"Source Page: {doc.metadata.get('page', 0) + 1}")
                st.write(
                    f"Rerank Score: {doc.metadata.get('rerank_score', 0.0):.4f}"
                )
                st.write(doc.page_content)
                st.divider()

    logger.info("Answer Generated")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "pages": pages
    })