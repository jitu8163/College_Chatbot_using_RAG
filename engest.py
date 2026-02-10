import os
from dotenv import load_dotenv
from math import ceil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

assert QDRANT_URL, "QDRANT_URL not set"
assert QDRANT_API_KEY, "QDRANT_API_KEY not set"

print("\n========== PDF INGESTION STARTED ==========\n")

# Embeddings
print("[1/6] Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
embedding_dim = len(embeddings.embed_query("test"))
print("Embedding model loaded.\n")

# Qdrant
print("[2/6] Connecting to Qdrant and recreating collection...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE
    )
)
print("Collection recreated (old data deleted).\n")

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# Text splitter
print("[3/6] Initializing text splitter...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
print("Text splitter ready.\n")

pdf_dir = "dataset"
all_docs = []

print("[4/6] Loading PDFs and extracting pages...")
pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
print(f"Total PDFs found: {len(pdf_files)}\n")

for i, file in enumerate(pdf_files, start=1):
    path = os.path.join(pdf_dir, file)
    print(f"  → ({i}/{len(pdf_files)}) Reading {file} ...")

    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"     Pages loaded: {len(pages)}")

    print(f"     Chunking started for {file} ...")
    docs = splitter.split_documents(pages)

    for d in docs:
        d.metadata["source_file"] = file

    print(f"     Chunks created: {len(docs)}\n")
    all_docs.extend(docs)

print(f"[5/6] Total chunks from all PDFs: {len(all_docs)}\n")

# Batch ingestion
batch_size = 32
total_batches = ceil(len(all_docs) / batch_size)

print("[6/6] Starting embedding + Qdrant ingestion...\n")

for i in range(0, len(all_docs), batch_size):
    batch_num = (i // batch_size) + 1
    remaining = total_batches - batch_num

    batch = all_docs[i:i + batch_size]

    print(f"  → Batch {batch_num}/{total_batches}")
    print(f"     Embedding + storing {len(batch)} chunks...")
    print(f"     Remaining batches after this: {remaining}")

    vectorstore.add_documents(batch)

    print(f"     Batch {batch_num} completed.\n")

print("========== INGESTION COMPLETED SUCCESSFULLY ==========")
print(f"Total PDFs       : {len(pdf_files)}")
print(f"Total chunks     : {len(all_docs)}")
print(f"Total batches    : {total_batches}")
print("All data is now embedded and stored in Qdrant.")

