#RAG pipeline & LLM Logic
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from qdrant_client.http.models import PointStruct
from app.database import client, COLLECTION_NAME

# 1. Instantiate the local Embedding Engine
# This model converts strings into 1024-dimensional numeric arrays capturing semantic meaning.
embedding_engine = OllamaEmbeddings(model="mxbai-embed-large")

# 2. Instantiate the local Language Model Engine
# We lock the temperature at 0.0 to prevent creative writing (hallucinations) in production.
llm = ChatOllama(model="phi3", temperature=0.0)


def ingest_document(doc_id: int, text: str):
    """
    Transforms a raw text document into an embedding and saves it to Qdrant.
    """
    # Line A: Generate the mathematical vector representing the text chunk
    vector = embedding_engine.embed_query(text)
    
    # Line B: Structure the data payload for the database
    # PointStruct is a native Qdrant class that packages the vector along with its metadata.
    point = PointStruct(
        id=doc_id,
        vector=vector,
        payload={"content": text}
    )
    
    # Line C: Perform a synchronous upsert (insert/update) into our vector database
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )


def query_rag_pipeline(user_query: str) -> str:
    """
    Searches the vector database for matching context and prompts the local LLM.
    """
    # Line D: Convert the user's incoming question into the same 1024-dimensional vector space
    query_vector = embedding_engine.embed_query(user_query)
    
    # Line E: Query Qdrant for the top 2 closest matching text points using Cosine Similarity
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=2
    )
    
    # Line F: Extract text data out of the database points and concatenate into a single block
    retrieved_contexts = [hit.payload["content"] for hit in search_results]
    context_block = "\n---\n".join(retrieved_contexts)
    
    # Line G: Build a structured prompt enforcing strict boundaries (Prompt Engineering)
    system_prompt = (
        f"You are a precise corporate assistant. Answer the question based ONLY on the context provided below.\n"
        f"If you do not know the answer, say 'I cannot find that in the internal documentation.'\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {user_query}"
    )
    
    # Line H: Stream the unified prompt to the local LLM and return the string response
    response = llm.invoke(system_prompt)
    return response.content