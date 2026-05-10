# 🤖 SupportMindBot

An AI-powered customer support chatbot that answers questions based on your company's PDF guidelines and policies.

Built with LangChain, ChromaDB, OpenAI, and Streamlit.

---

## ✨ Features

- 📄 Multi PDF support — load multiple policy documents at once
- 🧠 Conversation memory — remembers last 5 exchanges
- 📚 Source citations — shows exact page answer came from
- ⚡ Streaming responses — token by token like ChatGPT
- 🎛️ Adjustable retrieval — control chunks via UI slider
- 🗑️ Clear chat history — reset memory anytime

---

## 🛠️ Tech Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| Framework       | LangChain                     |
| LLM             | OpenAI GPT-4o-mini            |
| Embeddings      | OpenAI text-embedding-3-small |
| Vector Store    | ChromaDB                      |
| UI              | Streamlit                     |
| Package Manager | uv                            |

---

## 📁 Project Structure

    SupportMindBot/
    ├── app.py           → Streamlit web interface
    ├── ingest.py        → Load and embed PDF documents
    ├── rag_chain.py     → Terminal chat interface
    ├── config.py        → Environment configuration
    ├── .env             → API keys (never commit!)
    └── .gitignore

---

## ⚙️ Installation

### 1. Clone the repository

    git clone https://github.com/satte1/SupportMindBot.git
    cd SupportMindBot

### 2. Install uv

    # Windows
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"

    # Mac/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

### 3. Create virtual environment

    uv venv
    .venv\Scripts\activate   # Windows
    source .venv/bin/activate  # Mac/Linux

### 4. Install dependencies

    uv pip install langchain langchain-openai langchain-community
    uv pip install langchain-text-splitters langchain-chroma
    uv pip install pypdf chromadb tiktoken
    uv pip install sentence-transformers streamlit
    uv pip install python-dotenv

### 5. Set up environment variables

Create a `.env` file in project root:

    OPENAI_API_KEY=sk-proj-your-key-here

---

## 🚀 Usage

### Step 1 — Add your PDF guidelines

    Drop your company policy/guideline PDFs into docs/ folder

### Step 2 — Ingest documents (run once)

    python ingest.py

### Step 3 — Run the web app

    streamlit run app.py

Opens at: http://localhost:8501

### Alternative — Terminal chat

    python rag_chain.py

---

## 💬 How it Works

    Customer Question
          ↓
    Convert to Vector
    (OpenAI Embeddings)
          ↓
    Search ChromaDB
    (find similar chunks)
          ↓
    Top 4 Chunks Retrieved
    from Policy Documents
          ↓
    GPT-4o-mini generates
    answer from chunks
          ↓
    Answer + Source Page
    shown to customer

---

## ⚠️ Important Notes

- Never commit your `.env` file
- Run `ingest.py` again when you add new PDFs
- `chroma_db/` is auto generated — no need to commit
- PDFs stay on your local machine — never uploaded

---

## 🔮 Roadmap

- [ ] Upload PDFs directly from Streamlit UI
- [ ] Hybrid search (keyword + semantic)
- [ ] Chat history saved to database
- [ ] Support for CSV and Excel files
- [ ] Deploy to cloud with Pinecone
- [ ] Multi language support
- [ ] User authentication

---

## 👨‍💻 Author

Built by [satte1](https://github.com/satte1)

---

## 📝 License

MIT License — free to use and modify.clear
# 🤖 SupportMindBot

🌐 **Live Demo** → https://supportmindbot.streamlit.app

An AI-powered customer support chatbot...