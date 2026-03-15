# -*- coding: utf-8 -*-
import sys
import os
import asyncio

# Ensure Windows prints UTF-8 properly for Chinese characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.rag_service import rag_pipeline

async def main():
    print("Testing Dual-Prompt LightRAG Hybrid Retrieval System...")
    
    # Send a query that is specifically targeted at the chunk we just ingested
    initial_state = {
        "query": "狗狗耳朵一直甩，裡面有黑黑像咖啡渣的耳垢，那是耳疥蟲嗎？該怎麼辦？",
        "image_base64": None,
        "context": "",
    }
    
    final_state = await rag_pipeline.ainvoke(initial_state)
    
    print("\n================ RAG PIPELINE RESULT =================\n")
    print(f"1. Entity Keywords Extracted: {final_state.get('entity_keywords')}")
    print(f"2. Relation Sentence Extracted: {final_state.get('relation_sentence')}")
    print(f"\n3. Context Fetched (Cross-DB):\n{final_state.get('retrieved_texts')}")
    print(f"\n4. Final LLM Answer:\n{final_state.get('final_answer')}")
    print("\n======================================================\n")

if __name__ == "__main__":
    asyncio.run(main())
