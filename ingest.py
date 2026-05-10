import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "supportmindbot"

# Step 2 - Load PDF
docs_path = r"D:\lang_basic\docs"
pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]

if not pdf_files:
    print("❌ No PDF files found in docs folder!")
    exit()

print(f"📂 Found {len(pdf_files)} PDF(s):")
for pdf in pdf_files:
    print(f"   - {pdf}")

# Step 3 - Load all PDFs
print("\nLoading all PDFs...")
all_pages = []
for pdf_file in pdf_files:
    pdf_path = os.path.join(docs_path, pdf_file)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    all_pages.extend(pages)
    print(f"   ✅ {pdf_file} → {len(pages)} pages loaded")

print(f"\nTotal pages loaded: {len(all_pages)}")

# Step 4 - Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(all_pages)
print(f"Total chunks created: {len(chunks)}")

# Step 5 - Embeddings
print("\nInitializing embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 6 - Upload to Pinecone
print("\nUploading to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)
print(f"\n✅ Successfully uploaded to Pinecone!")
print(f"   Index name: {index_name}")
print(f"   Total chunks uploaded: {len(chunks)}")