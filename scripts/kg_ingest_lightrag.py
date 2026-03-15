# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import uuid
import logging
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from core.neo4j_client import neo4j_db
from core.qdrant_client import qdrant_db
from core.llm_client import llm_client
from core.pg_client import pg_db
from scripts.supernodes_list import PREDEFINED_SUPERNODES
from qdrant_client.http.models import PointStruct, VectorParams, Distance

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LightRAG_Ingester")

# Initialize models
llm = llm_client.get_llm()
embeddings = llm_client.get_embeddings()

class EntityMapping(BaseModel):
    name: str = Field(description="實體的具象名稱 (例如: '頻尿', '耳疥蟲', '高油脂食物')")
    super_node: str = Field(description="該實體唯一歸屬的預定義超節點。如果都不符合，請填寫 'UNALIGNED'")

class Relationship(BaseModel):
    subject: str = Field(description="主體實體名稱 (必須存在於 EntityMapping 裡)")
    object_: str = Field(description="受體實體名稱 (必須存在於 EntityMapping 裡)")
    summary: str = Field(description="這兩個實體之間的簡短知識總結，請用完整的短句表達 (例如: '若攝取過多高油脂食物容易導致胰臟炎。')")

class LightRAGExtraction(BaseModel):
    entities: List[EntityMapping] = Field(description="所有具體被提到的實體及其對應的唯一預定義超節點 (或 UNALIGNED)")
    relationships: List[Relationship] = Field(description="實體之間的關聯與短知識摘要")

EXTRACT_PROMPT = PromptTemplate.from_template("""
你是一個專業的寵物醫療與照護知識圖譜萃取引擎，負責建立「有醫療診斷與照護行為價值」的專家大腦。
請閱讀以下提供的「原始知識 Json 區塊」，並將其轉換成符合 LightRAG 架構的階層式向量圖譜格式。

非常重要：我們只要【高價值的醫療/照護專業知識】，不要湊字數或抓取無意義的通用詞彙（如「獸醫」、「寵物」、「問題」、「身體」、「部位」、「狀況」等）。

以下是系統【已經預先定義好的 50+ 權威分類清單 (SuperNodes)】：
{supernodes_list}

要求如下：
1. 【Entities (實體)】：抓取內文具象且【有醫療或照護意義】的名詞（例如：耳疥蟲、頻尿、處方飼料、左旋肉鹼、黑色素瘤等）。
   - 🚫 絕對禁止提取無實際知識價值的泛稱名詞：如「狗狗」、「動物」、「獸醫」、「檢查」、「藥物」、「部位」、「疾病」等。
   - 嚴格規則：**一個實體只能且必須對齊唯一一個上面列出的預定義超節點**。你要自己判斷它屬於表單中的哪一項並填入「一模一樣的字串」。
   - 如果這個實體（例如某種罕見毒素或新療法）在上面的權威清單中完全找不到適合的分類，請你在 super_node 欄位直接填寫 "UNALIGNED"（稍後會交由系統自動演化分群）。
2. 【Relationships (知識關聯)】：萃取實體與實體之間的關係，必須包含「濃縮的簡短知識總結 (summary)」。
   - 如果這兩個實體之間的互動沒有實質的照護指導意義（例如：「獸醫」-[檢查]->「耳朵」這種廢話），請直接捨棄這組關聯。
   - 格式：(主體) - [知識總結] -> (受體)
   - 範例：(木醣醇) - [誤食糖果中的木醣醇可能導致貓狗中毒甚至致命] -> (中毒)

## 原始知識 Json：
{text}
""")

def extract_knowledge(json_text: str) -> LightRAGExtraction:
    """Uses LLM to extract LightRAG structure."""
    supernodes_str = "\n".join([f"- {sn}" for sn in PREDEFINED_SUPERNODES])
    chain = EXTRACT_PROMPT | llm.with_structured_output(LightRAGExtraction)
    return chain.invoke({"text": json_text, "supernodes_list": supernodes_str})

def ensure_qdrant_collection():
    """Ensure the LightRAG collection exists."""
    collection_name = "pet_light_rag"
    try:
        qdrant_db.client.get_collection(collection_name)
    except Exception:
        logger.info(f"Creating Qdrant collection {collection_name}...")
        qdrant_db.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
    return collection_name

def ingest_chunk(chunk_data: dict, collection_name: str):
    """Processes a single JSON chunk, extracts info, and saves to Neo4j & Qdrant."""
    source_id = chunk_data.get('new_id', 'unknown')
    logger.info(f"Processing chunk {source_id}...")
    
    # 僅提取核心有醫療意義的欄位給 LLM，避免冗餘資料混淆並節省 Token
    core_information = {
        "possible_symptoms": chunk_data.get("possible_symptoms", ""),
        "causes": chunk_data.get("causes", []),
        "solutions": chunk_data.get("solutions", [])
    }
    chunk_str = json.dumps(core_information, ensure_ascii=False)
    try:
        extraction: LightRAGExtraction = extract_knowledge(chunk_str)
    except Exception as e:
        logger.error(f"LLM Extraction failed: {e}")
        return

    # 1. Write to Neo4j Graph
    graph = neo4j_db.get_graph()
    
    # Merge Entities and their BELONGS_TO edges to SuperNode (if not UNALIGNED)
    for ent in extraction.entities:
        graph.query("""
            MERGE (e:Entity {name: $ent_name})
            ON CREATE SET e.source = 'lightrag'
        """, {"ent_name": ent.name})

        if ent.super_node != "UNALIGNED":
            graph.query("""
                MATCH (e:Entity {name: $ent_name})
                MATCH (s:SuperNode {name: $sn_name})
                MERGE (e)-[:BELONGS_TO]->(s)
            """, {"ent_name": ent.name, "sn_name": ent.super_node})
        
    # Pre-generate UUIDs for relationships (for cross-referencing Neo4j ↔ Qdrant)
    rel_uuids = [str(uuid.uuid4()) for _ in extraction.relationships]

    # Merge Relationships with properties + qdrant_id
    for rel, rel_uuid in zip(extraction.relationships, rel_uuids):
        graph.query("""
            MERGE (a:Entity {name: $subj})
            MERGE (b:Entity {name: $obj})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r.summary = $summary
            SET r.source = 'lightrag'
            SET r.source_id = $source_id
            SET r.qdrant_id = $qdrant_id
        """, {"subj": rel.subject, "obj": rel.object_, "summary": rel.summary, 
              "source_id": source_id, "qdrant_id": rel_uuid})

    # 2. Embed and write to Qdrant
    points = []
    
    # Embed Entities & handle UNALIGNED Fallback
    if extraction.entities:
        ent_texts = [e.name for e in extraction.entities]
        ent_vectors = embeddings.embed_documents(ent_texts)
        for ent, vec in zip(extraction.entities, ent_vectors):
            
            # If UNALIGNED, toss to HDBSCAN candidate pool
            if ent.super_node == "UNALIGNED":
                candidate_uuid = str(uuid.uuid4())
                pg_db.insert_candidate(vector=vec, uuid=candidate_uuid, dialog_text=ent.name)
                logger.info(f"Entity '{ent.name}' is UNALIGNED. Saved to Candidate Pool {candidate_uuid}.")
            
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload={"text": ent.name, "type": "entity", "target_sn": ent.super_node, "source_id": source_id}
            ))
            
    # Embed Relationships (The summary knowledge) — use pre-generated UUIDs
    if extraction.relationships:
        rel_texts = [f"{r.subject} - [{r.summary}] -> {r.object_}" for r in extraction.relationships]
        rel_vectors = embeddings.embed_documents(rel_texts)
        for rel, text, vec, rel_uuid in zip(extraction.relationships, rel_texts, rel_vectors, rel_uuids):
            points.append(PointStruct(
                id=rel_uuid,
                vector=vec,
                payload={
                    "text": text, 
                    "type": "relationship", 
                    "subject": rel.subject, 
                    "object": rel.object_, 
                    "summary": rel.summary,
                    "original_content": chunk_str,
                    "source_id": source_id
                }
            ))

    # Upsert to Qdrant if we generated any points
    if points:
        qdrant_db.client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Successfully upserted {len(points)} vectors (Entities, SuperNodes, Relationships) to Qdrant.")

def run_dry_test():
    """Runs a single chunk from the dataset for user validation."""
    collection_name = ensure_qdrant_collection()
    
    # Read the first element from the total_KG.json
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "total_KG_analysis_output.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    sample_chunk = data[0] # The one about ear mites
    
    logger.info("============== DRY RUN EXTRACTION ==============")
    core_information = {
        "possible_symptoms": sample_chunk.get("possible_symptoms", ""),
        "causes": sample_chunk.get("causes", []),
        "solutions": sample_chunk.get("solutions", [])
    }
    chunk_str = json.dumps(core_information, ensure_ascii=False)
    extraction = extract_knowledge(chunk_str)
    
    print("\n[Parsed Entities & Constraints]:")
    for e in extraction.entities:
        print(f" - [{e.name}] 對齊 ===> ({e.super_node})")
        
    print("\n[Parsed Relationships / Short Knowledge Summary]:")
    for r in extraction.relationships:
        print(f" - ({r.subject}) -[{r.summary}]-> ({r.object_})")
        
    print("\n=================================================")
    print("Ingesting this sample into DB...")
    ingest_chunk(sample_chunk, collection_name)
    print("Done! Check Neo4j and Qdrant.")

def run_full_ingestion():
    """Runs the full ingestion pipeline for the entire dataset."""
    import time
    from tqdm import tqdm
    collection_name = ensure_qdrant_collection()
    
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "total_KG_analysis_output.json")
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    logger.info(f"Starting FULL Ingestion for {len(data)} chunks...")
    
    # Optional: Start from a specific index if it crashed previously
    start_idx = 0
    
    for i, chunk in enumerate(tqdm(data[start_idx:], desc="Ingesting Chunks", unit="chunk")):
        try:
            ingest_chunk(chunk, collection_name)
        except Exception as e:
            logger.error(f"Fatal error on chunk {chunk.get('new_id')}: {e}")
            continue
            
        # Give OpenAI API and our databases a short breather
        time.sleep(1.0)
        
    logger.info("🎉 FULL Ingestion logic completed successfully!")

if __name__ == "__main__":
    # run_dry_test()
    run_full_ingestion()
