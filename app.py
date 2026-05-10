import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Bot",
    page_icon="📄",
    layout="centered"
)

# ─── Title ─────────────────────────────────────────────────
st.title("📄 PDF Q&A Bot")
st.caption("Ask questions about your documents — powered by RAG + LangChain")

# ─── Load Vector Store ──────────────────────────────────────
chroma_path = r"D:\lang_basic\chroma_db"

@st.cache_resource  # load only once, not on every message
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
        return None
    vectorstore = Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

# ─── Check Vector Store ─────────────────────────────────────
if vectorstore is None:
    st.error("❌ No vector store found! Please run ingest.py first.")
    st.code("python ingest.py", language="bash")
    st.stop()

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # show total vectors
    total_vectors = vectorstore._collection.count()
    st.metric("Total Vectors", total_vectors)

    # show PDFs loaded
    st.subheader("📂 Documents Loaded")
    docs_path = r"D:\lang_basic\docs"
    pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]
    for pdf in pdf_files:
        st.write(f"📄 {pdf}")

    # number of chunks to retrieve
    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Higher k = more context but slower"
    )

    # clear memory button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + ChromaDB + Streamlit")

# ─── Initialize Session State ───────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ─── Display Chat History ───────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── RAG Chain Setup ────────────────────────────────────────
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": k_value}
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using 
ONLY the context provided below.
If the answer is not in the context, say 'I don't know based on the document.'

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ─── Chat Input ─────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):

    # show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # keep last 5 exchanges
            history = st.session_state.chat_history[-10:]

            chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "chat_history": lambda _: history
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # stream response
            response = st.write_stream(chain.stream(question))

        # show sources
        with st.expander("📚 Sources used"):
            source_docs = retriever.invoke(question)
            for i, doc in enumerate(source_docs):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                st.caption(f"**Source {i+1}:** {source} — Page {page}")
                st.text(doc.page_content[:200] + "...")
                st.divider()

    # save to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))