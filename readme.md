# 🎥 YouTube AI Chatbot

[![Streamlit App](https://img.shields.io/badge/Live-App-blue?style=for-the-badge)](https://youtubetranscriptchatbot-by-saad2004.streamlit.app/)

**YouTube AI Chatbot** lets you ask questions from any YouTube video using AI-powered retrieval. The app fetches video transcripts, splits them into searchable chunks, creates embeddings, and uses a Hugging Face LLM to answer your questions based on the transcript.

---

## 🚀 Live Demo

Try it now: [https://youtubetranscriptchatbot-by-saad2004.streamlit.app/](https://youtubetranscriptchatbot-by-saad2004.streamlit.app/)

---

## 🛠 Features

- Fetches YouTube transcripts automatically using `youtube-transcript-api`.
- Falls back to manual transcript input if unavailable.
- Splits transcripts into searchable chunks for efficient retrieval.
- Creates vector embeddings using Hugging Face `sentence-transformers`.
- Answers questions accurately using a Hugging Face LLM.
- Beautiful, modern Streamlit interface with chat-style UI.
- Fully customizable: choose your LLM, temperature, and top-K search chunks.

---

## ⚙️ Settings

In the sidebar:

- **Hugging Face API Token:** Required for LLM access.
- **HF Model Repo:** Default: `HuggingFaceH4/zephyr-7b-beta`.
- **Temperature:** Control randomness in responses (0–1).
- **Top K Chunks:** Number of transcript chunks to search.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/YouTube-Transcript-Chatbot.git
cd YouTube-Transcript-Chatbot
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
streamlit run app.py
