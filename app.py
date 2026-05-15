import os
import streamlit as st
from langchain_pinecone import PineconeVectorStore
import pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="SupportMindBot",
    page_icon="🤖",
    layout="centered"
)

# ─── Title ──────────────────────────────────────────────────
st.title("🤖 SupportMindBot")
st.caption("AI customer support chatbot powered by RAG + LangChain")

# ─── Load Vector Store ──────────────────────────────────────
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name="supportmindbot",
        embedding=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()

# ─── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.success("✅ Connected to Pinecone")

    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=8,
        value=4,
        help="Higher k = more context but slower"
    )

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangChain + Pinecone + Streamlit")

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
    ("system", """You are a helpful customer support assistant.
Answer the question using ONLY the context provided below.
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
if question := st.chat_input("Ask a question about our policies..."):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
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

            response = st.write_stream(chain.stream(question))

        with st.expander("📚 Sources used"):
            source_docs = retriever.invoke(question)
            for i, doc in enumerate(source_docs):
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                st.caption(f"**Source {i+1}:** {source} — Page {page}")
                st.text(doc.page_content[:200] + "...")
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=response))