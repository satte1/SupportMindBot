import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Initialize Pinecone vector store
print("Loading vector store from Pinecone...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(
    index_name="supportmindbot",
    embedding=embeddings
)
print("✅ Vector store loaded!")

# Step 2 - Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Step 3 - Simple memory using plain list
chat_history = []

# Step 4 - Prompt with memory
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful customer support assistant. 
Answer the question using ONLY the context provided below.
If the answer is not in the context, say 'I don't know based on the document.'

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Step 5 - LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Step 6 - Format docs helper
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Step 7 - Ask function with memory
def ask(question):
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

    full_answer = ""
    print("\nBot: ", end="")
    for chunk in chain.stream(question):
        print(chunk, end="", flush=True)
        full_answer += chunk
    print()

    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=full_answer))

    return full_answer

# Step 8 - Chat loop
print("\n✅ SupportMindBot ready!")
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