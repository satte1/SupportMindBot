import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 2 - Check and load vector store
chroma_path = r"D:\lang_basic\chroma_db"

if not os.path.exists(chroma_path) or not os.listdir(chroma_path):
    print("❌ No vector store found! Please run ingest.py first.")
    exit()

print("Loading vector store from disk...")
vectorstore = Chroma(
    persist_directory=chroma_path,
    embedding_function=embeddings
)
print(f"✅ Vector store loaded! Total vectors: {vectorstore._collection.count()}")

# Step 3 - Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Step 4 - Simple memory using plain list
chat_history = []

# Step 5 - Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using 
ONLY the context provided below.
If the answer is not in the context, say 'I don't know based on the document.'

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Step 6 - LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Step 7 - Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 8 - Ask function with memory
def ask(question):
    # keep last 5 exchanges = 10 messages
    history = chat_history[-10:]

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

    # stream answer token by token
    full_answer = ""
    print("\nBot: ", end="")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
        full_answer += chunk
    print()

    # save question and answer to memory
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=full_answer))

    return full_answer

# Step 9 - Chat loop
print("\n✅ SQL Document Q&A Bot with Memory ready!")
print("Type your question or 'exit' to quit.")
print("Type 'clear' to reset memory.\n")
print("-" * 40)

while True:
    question = input("\nYou: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question.lower() == "clear":
        chat_history.clear()
        print("✅ Memory cleared!")
        continue

    if question.strip() == "":
        print("Please enter a question!")
        continue

    ask(question)