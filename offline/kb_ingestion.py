from core.qdrant_client import qdrant_db
from core.neo4j_client import neo4j_db

def run_knowledge_ingestion():
    """
    Module 1.1: Knowledge Base Ingestion
    """
    print("Starting Offline Foundation Pipeline - Knowledge Base Ingestion...")
    # Step A: Text Splitter / Chunking (LangChain/LlamaIndex)
    print("[STEP A] Cleaning & Chunking text")
    
    # Step B: Vectorization (sentence-transformers)
    print("[STEP B] Embedding & Qdrant Storage")
    
    # Step C: LLM Knowledge Extraction (List[Triple])
    print("[STEP C] LLM Triplet Extraction")
    
    # Step D: Graph Database Write (Lock truth is_locked=True)
    print("[STEP D] Neo4j Locked Write")

if __name__ == "__main__":
    run_knowledge_ingestion()
