import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# -------- ENV VALIDATION --------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION")

assert GROQ_API_KEY, "GROQ_API_KEY not set"
assert QDRANT_URL, "QDRANT_URL not set"

# -------- LLM --------
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# -------- EMBEDDINGS --------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------- VECTOR STORE --------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)

# 🔑 IMPORTANT: higher k enables multi-section retrieval
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)

# -------- CONTEXT FORMATTER --------
def format_docs(docs):
    if not docs:
        return ""

    formatted = []
    for i, d in enumerate(docs, 1):
        source = d.metadata.get("source_file", "unknown")
        row = d.metadata.get("row_index", "unknown")

        formatted.append(
            f"Section {i} (source: {source}, row: {row}):\n{d.page_content}"
        )

    return "\n\n".join(formatted)

# -------- PROMPT (HANDLES ALL 3 SCENARIOS) --------
prompt = ChatPromptTemplate.from_template("""
You are a Answering chatbot of Telepathy College Of Medical Science.

Rules (strict):
1. Use ONLY the information provided in the context. if thq question is not greeting.
2. If question is greeting, then greet in a humble manner.
3. You MAY combine information from multiple sections if needed.
4. If the answer is not present in the context, say exactly: "I don't know."
5. Do NOT use external knowledge.
6. If articles/sections are mentioned in context, include them in the answer.

Context:
{context}

Question:
{question}

Answer:
""")

# -------- RAG CHAIN --------
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# -------- PUBLIC API --------
def ask(question: str) -> str:
    response = rag_chain.invoke(question)

    # Safety: empty context → forced refusal
    if not response.content.strip():
        return "I don't know."

    return response.content.strip()

