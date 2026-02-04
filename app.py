# app.py
# Modern Streamlit Frontend for YouTube-Based RAG Chatbot
# Author: Muhammad Saad Zubair

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import re
import requests

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

def fetch_transcript_api(video_id, api_key=None):
    """
    Fetch transcript from YouTube using YouTube Data API v3
    """
    if not api_key:
        return None  # fallback to manual input

    try:
        # Get list of captions for video
        url = f"https://www.googleapis.com/youtube/v3/captions?part=snippet&videoId={video_id}&key={api_key}"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        data = r.json()
        if "items" not in data or len(data["items"]) == 0:
            return None

        # Find English caption
        caption_id = None
        for item in data["items"]:
            if item["snippet"]["language"] == "en":
                caption_id = item["id"]
                break
        if not caption_id:
            return None

        # Download caption text in SRT format
        caption_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?tfmt=srt&key={api_key}"
        r2 = requests.get(caption_url)
        if r2.status_code != 200:
            return None

        return r2.text

    except Exception:
        return None


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🎥 YouTube AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown(
    """
    <style>
    body {
        background-color: #0f172a;
    }

    .main {
        background-color: #0f172a;
    }

    .title {
        text-align: center;
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 0.5rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #cbd5f5;
        margin-bottom: 2rem;
    }

    .chat-box {
        background: #020617;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }

    .user-msg {
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        color: white;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        max-width: 80%;
    }

    .bot-msg {
        background: #1e293b;
        color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        max-width: 80%;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        margin-top: 2rem;
        font-size: 0.9rem;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #38bdf8);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }

    .stTextInput > div > div > input {
        background-color: #020617;
        color: white;
        border-radius: 10px;
        border: 1px solid #334155;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- UI HEADER ----------------
st.markdown("<div class='title'>🎥 YouTube AI Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask questions from any YouTube video using AI + Vector Search</div>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    hf_token = st.text_input("Hugging Face API Token", type="password")
    model_repo = st.text_input("HF Model Repo", value="HuggingFaceH4/zephyr-7b-beta")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
    k_docs = st.slider("Top K Chunks", 1, 10, 5)

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
    """
    1. Fetches YouTube transcript (via youtube_transcript_api)
    2. Splits into chunks
    3. Creates embeddings
    4. Stores in FAISS
    5. Answers using Hugging Face LLM
    """
    )
    yt_api_key = st.text_input("YouTube Data API Key", type="password")

# ---------------- SESSION STATE ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- HELPERS ----------------
def extract_video_id(url_or_id):
    pattern = r"(?:v=|youtu.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url_or_id)
    return match.group(1) if match else url_or_id


def load_and_index(video_id, k=5):
    text = fetch_transcript_api(video_id, yt_api_key)

    # Fallback if transcript fetch fails
    if not text:
        st.info("Unable to fetch transcript automatically. Please paste transcript manually:")
        text = st.text_area("Paste transcript here")
        if not text:
            return None, "No transcript provided."

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([text])
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store, f"Indexed {len(chunks)} chunks successfully."
    except Exception as e:
        return None, f"Error creating vector store: {str(e)}"





def build_chain(vector_store, hf_token, model_repo, temperature, k_docs):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k_docs}
    )

    hf_endpoint = HuggingFaceEndpoint(
        repo_id=model_repo,
        temperature=temperature,
        huggingfacehub_api_token=hf_token
    )

    llm = ChatHuggingFace(llm=hf_endpoint)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Use ONLY the provided transcript context to answer the question.
        If the answer is not in the context, say "I don't know."

        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    parallel_chain = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough()
    )

    parser = StrOutputParser()

    return parallel_chain | prompt | llm | parser

# ---------------- MAIN UI ----------------
col1, col2 = st.columns([3, 1])

with col1:
    video_input = st.text_input("🔗 Enter YouTube Video URL or ID")

with col2:
    index_btn = st.button("Index Video")

if index_btn and video_input:
    with st.spinner("Indexing video transcript..."):
        vid = extract_video_id(video_input)
        store, msg = load_and_index(vid, k_docs)
        if store:
            st.session_state.vector_store = store
            st.success(msg)
        else:
            st.error(msg)

st.markdown("---")

# ---------------- CHAT UI ----------------
st.markdown("### Chat with the Video")

chat_container = st.container()

with chat_container:
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

question = st.text_input("Ask something about the video...")

if st.button("Ask") and question:
    if not st.session_state.vector_store:
        st.warning("Please index a video first")
    elif not hf_token:
        st.warning("Please enter Hugging Face API Token in sidebar")
    else:
        chain = build_chain(
            st.session_state.vector_store,
            hf_token,
            model_repo,
            temperature,
            k_docs
        )

        with st.spinner("Thinking..."):
            answer = chain.invoke(question)

        st.session_state.chat_history.append(("user", question))
        st.session_state.chat_history.append(("bot", answer))

        st.experimental_rerun()

# ---------------- FOOTER ----------------
st.markdown(
    "<div class='footer'>Built by M.Saad </div>",
    unsafe_allow_html=True
)


