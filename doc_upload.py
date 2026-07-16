from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS

# -------------------------------
# Step 1: Load PDF
# -------------------------------
loader = PyPDFLoader(
    "/Users/rajasekharreddykasireddy/Desktop/RajasekharReddy_DataEngineer.pdf"
)

documents = loader.load()

# -------------------------------
# Step 2: Split into chunks
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# -------------------------------
# Step 3: Create Embeddings
# -------------------------------
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -------------------------------
# Step 4: Store in FAISS
# -------------------------------
vector_db = FAISS.from_documents(
    chunks,
    embeddings
)

vector_db.save_local("faiss_index")

print("FAISS index created successfully.")

# -------------------------------
# Step 5: Load LLM
# -------------------------------
llm = OllamaLLM(
    model="llama3.2"
)

# -------------------------------
# Step 6: Ask Questions
# -------------------------------
while True:

    question = input("\nAsk your question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # Retrieve top 3 relevant chunks
    docs = vector_db.similarity_search(
        question,
        k=3
    )

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the context below.

If the answer is not available in the context, reply:
"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nAnswer:")
    print(response)