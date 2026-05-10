import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Step 1 - Load ALL PDFs from docs folder
docs_path = r"D:\lang_basic\docs"
chroma_path = r"D:\lang_basic\chroma_db"

# Check if docs folder has any PDFs
pdf_files = [f for f in os.listdir(docs_path) if f.endswith(".pdf")]

if not pdf_files:
    print("❌ No PDF files found in docs folder!")
    print(f"   Please add PDFs to: {docs_path}")
    exit()

print(f"📂 Found {len(pdf_files)} PDF(s):")
for pdf in pdf_files:
    print(f"   - {pdf}")

# Step 2 - Load all PDFs
print("\nLoading all PDFs...")
all_pages = []

for pdf_file in pdf_files:
    pdf_path = os.path.join(docs_path, pdf_file)
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    all_pages.extend(pages)
    print(f"   ✅ {pdf_file} → {len(pages)} pages loaded")

print(f"\nTotal pages loaded: {len(all_pages)}")

# Step 3 - Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(all_pages)
print(f"Total chunks created: {len(chunks)}")

# Step 4 - Show chunks per document
print("\nChunks per document:")
chunk_sources = {}
for chunk in chunks:
    source = os.path.basename(chunk.metadata.get("source", "unknown"))
    chunk_sources[source] = chunk_sources.get(source, 0) + 1

for source, count in chunk_sources.items():
    print(f"   - {source}: {count} chunks")

# Step 5 - Embeddings
print("\nInitializing embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 6 - Clear old vector store and create fresh
if os.path.exists(chroma_path):
    shutil.rmtree(chroma_path)
    print("🗑️  Old vector store cleared!")

print("Embedding all chunks and saving to ChromaDB...")
print("(This may take a moment depending on number of PDFs...)")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=chroma_path
)
print(f"\n✅ Vector store saved!")
print(f"   Total vectors stored: {vectorstore._collection.count()}")
print(f"   Location: {chroma_path}")

# Step 7 - Test similarity search across all docs
print("\n--- Testing Search Across All Documents ---")
query = "what is this document about?"
results = vectorstore.similarity_search(query, k=4)

for i, doc in enumerate(results):
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page", "?")
    print(f"\nResult {i+1}:")
    print(f"   Source : {source}")
    print(f"   Page   : {page}")
    print(f"   Content: {doc.page_content[:150]}...")