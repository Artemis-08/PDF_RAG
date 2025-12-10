import streamlit as st
from PyPDF2 import PdfReader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Fallback simple splitter if langchain isn't available or has different layout.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=10000, chunk_overlap=1000, length_function=len):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function

        def split_text(self, text: str):
            if not text:
                return []
            chunks = []
            start = 0
            L = len(text)
            while start < L:
                end = min(start + self.chunk_size, L)
                chunks.append(text[start:end])
                start = end - self.chunk_overlap if end < L else end
            return chunks
import os

import json
import math
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Google GenAI client (support different genai versions)
api_key = os.getenv("GOOGLE_API_KEY")
try:
    genai.configure(api_key=api_key)
except Exception:
    try:
        genai.config.api_key = api_key
    except Exception:
        pass

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, persist_directory="./chromadb_store"):
    """Create a simple JSON-backed vector index for the texts.

    Stores index at `persist_directory/index.json` with keys: ids, documents, metadatas, embeddings.
    Returns the index path.
    """
    index_dir = persist_directory
    index_path = os.path.join(index_dir, "index.json")
    os.makedirs(index_dir, exist_ok=True)

    # Try to batch-embed via Google GenAI
    embeddings = []
    try:
        resp = genai.embed_content(model="models/embedding-001", content=text_chunks)
        if isinstance(resp, dict) and "embeddings" in resp:
            embeddings = [item["embedding"] if isinstance(item, dict) and "embedding" in item else item for item in resp["embeddings"]]
        elif isinstance(resp, dict) and "embedding" in resp:
            embeddings = [resp["embedding"]]
        else:
            embeddings = [getattr(r, "embedding", None) for r in resp]
            embeddings = [e for e in embeddings if e is not None]
    except Exception:
        embeddings = []

    # Fallback deterministic lightweight embeddings
    if len(embeddings) != len(text_chunks):
        embeddings = []
        for t in text_chunks:
            vec = [float((ord(c) % 97) / 97.0) for c in t[:128]]
            embeddings.append(vec)

    ids = [f"doc_{i}" for i in range(len(text_chunks))]
    metadatas = [{"source": f"chunk_{i}"} for i in range(len(text_chunks))]

    index = {"ids": ids, "documents": text_chunks, "metadatas": metadatas, "embeddings": embeddings}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f)
    return index_path
def generate_answer_from_context(question: str, context: str) -> str:
    """Try to generate an answer using Google Generative AI. If that fails, return the context as a fallback."""
    prompt = (
        "Answer the question as detailed as possible from the provided context. "
        "If the answer is not in the context, say you don't know. Do not hallucinate.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"
    )
    try:
        # prefer ChatSession if available
        try:
            model = genai.get_model("models/chat-bison-001")
            session = genai.ChatSession(model=model)
            res = session.send([{"type": "text", "text": prompt}])
            # try to extract a textual reply
            if isinstance(res, dict) and "candidates" in res:
                cand = res["candidates"][0]
                if isinstance(cand, dict):
                    return cand.get("content", {}).get("text", "")
                return str(cand)
            return str(res)
        except Exception:
            # fallback: some genai versions don't expose ChatSession
            return context
    except Exception:
        return context


def _cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def user_input(user_question: str):
    index_path = os.path.join("./chromadb_store", "index.json")
    if not os.path.exists(index_path):
        st.write("No documents uploaded yet. Please upload PDFs in the sidebar and press Submit & Process.")
        return

    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # embed the user question
    qemb = None
    try:
        qresp = genai.embed_content(model="models/embedding-001", content=[user_question])
        if isinstance(qresp, dict) and "embeddings" in qresp:
            first = qresp["embeddings"][0]
            qemb = first.get("embedding") if isinstance(first, dict) else first
        elif isinstance(qresp, dict) and "embedding" in qresp:
            qemb = qresp["embedding"]
    except Exception:
        qemb = None

    docs = []
    if qemb is not None:
        sims = [(_cosine_sim(qemb, emb), doc) for emb, doc in zip(index.get("embeddings", []), index.get("documents", []))]
        sims.sort(key=lambda x: x[0], reverse=True)
        docs = [doc for score, doc in sims[:3] if score > 0]

    if not docs:
        flat = index.get("documents", [])
        hits = [d for d in flat if user_question.lower() in d.lower()][:3]
        docs = hits

    context = "\n\n".join(docs)
    answer = generate_answer_from_context(user_question, context)

    st.write("Reply:")
    st.write(answer)
    if docs:
        st.write("\n---\nRetrieved documents:")
        for i, d in enumerate(docs):
            st.markdown(f"**Chunk {i+1}:** {d}")

def main():
    st.set_page_config(page_title="LangGraph")
    st.header("Chat with multiple PDF documents")

    user_question = st.text_input("Ask your question about the documents")

    if user_question:
        user_input(user_question)


    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF documents here", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.success("All done! You can now ask questions about your documents.")

if __name__ == '__main__':
    main()

