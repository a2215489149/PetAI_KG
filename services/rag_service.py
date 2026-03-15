import json
import logging
from typing import TypedDict, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from config.settings import settings
from core.qdrant_client import qdrant_db
from core.neo4j_client import neo4j_db
from core.llm_client import llm_client
from services.multimodal_service import multimodal_inference

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

# --- Graph State ---
class RagState(TypedDict):
    query: str
    image_base64: Optional[str]
    context: str
    user_id: Optional[str]
    entity_keywords: Optional[List[str]]
    relation_sentence: Optional[str]
    observed_image_json: Optional[Dict[str, Any]]
    neo4j_ids: Optional[List[str]]
    retrieved_texts: Optional[str]
    final_answer: Optional[str]

class DualPromptOutput(BaseModel):
    entity_keywords: List[str] = Field(description="從問題中提取的具體實體關鍵字陣列（如: ['貓', '頻尿', '處方飼料']）")
    relation_sentence: str = Field(description="將用戶情境轉換為完整的短敘述句，用以搜尋知識關聯（如: '貓咪頻尿應該給予什麼處方飼料及醫療處置'）")


# --- Nodes ---
async def multimodal_node(state: RagState) -> RagState:
    """Invokes vLLM for image observations if an image is provided."""
    if state.get("image_base64"):
        observation = await multimodal_inference(state["image_base64"], state["query"])
        return {"observed_image_json": observation}
    return {"observed_image_json": None}

def prompt_builder_node(state: RagState) -> RagState:
    """Uses LLM to synthesize a dual retrieval prompt (Entities & Relations)."""
    text = state["query"]
    obs = state.get("observed_image_json")
    history = state.get("context", "")
    
    template = """
    你是一個寵物醫療與照護 AI 系統的檢索引擎大腦。
    請根據以下的對話歷史、最新使用者提問，以及可選的畫面觀察結果，產生兩組完全不同維度的搜尋字串 (Prompt)，以供 LightRAG 混合檢索使用。
    
    1. entity_keywords: 專門對齊「具象名詞/醫學名詞」的關鍵字清單。
    2. relation_sentence: 專門對齊「完整知識句/關聯邊」的敘述句。
    
    <history>{history}</history>
    <vision_observation>{obs}</vision_observation>
    <user_input>{text}</user_input>
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm_client.get_llm().with_structured_output(DualPromptOutput)
    
    result: DualPromptOutput = chain.invoke({
        "history": history, 
        "obs": json.dumps(obs, ensure_ascii=False) if obs else "None", 
        "text": text
    })
    
    logger.info(f"[Dual-Prompt] Entities: {result.entity_keywords}")
    logger.info(f"[Dual-Prompt] Relation: {result.relation_sentence}")
    
    return {
        "entity_keywords": result.entity_keywords,
        "relation_sentence": result.relation_sentence
    }

def retrieve_hybrid_node(state: RagState) -> RagState:
    """Executes the dual-track LightRAG retrieval querying Qdrant Entities and Relations."""
    entity_keywords = state.get("entity_keywords", [])
    relation_sentence = state.get("relation_sentence", "")
    
    embeddings = llm_client.get_embeddings()
    client = qdrant_db.get_client()
    from qdrant_client.http import models
    from collections import defaultdict
    
    retrieved_texts = []
    
    # 1. Search Relations in pet_light_rag
    rel_hits = []
    if relation_sentence:
        try:
            rel_vec = embeddings.embed_query(relation_sentence)
            rel_results = client.query_points(
                collection_name="pet_light_rag", 
                query=rel_vec, 
                limit=5,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="relationship"))]
                )
            )
            rel_hits = rel_results.points
        except Exception as e:
            logger.warning(f"Qdrant Relation Search Error: {e}")
            
    # 2. Search Entities in pet_light_rag
    entity_hits = []
    if entity_keywords:
        try:
            ent_query = " ".join(entity_keywords)
            ent_vec = embeddings.embed_query(ent_query)
            ent_results = client.query_points(
                collection_name="pet_light_rag", 
                query=ent_vec, 
                limit=5,
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="entity"))]
                )
            )
            entity_hits = ent_results.points
        except Exception as e:
            logger.warning(f"Qdrant Entity Search Error: {e}")

    # 2.5 Search SuperNodes in pet_light_rag
    supernode_hits = []
    if entity_keywords:
        try:
            sn_results = client.query_points(
                collection_name="pet_light_rag", 
                query=ent_vec, 
                limit=3,
                score_threshold=0.85, # Only confident hits
                query_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="supernode"))]
                )
            )
            supernode_hits = sn_results.points
        except Exception as e:
            logger.warning(f"Qdrant SuperNode Search Error: {e}")

    # 3. Search original conversational chunks in pet_ai_collection as fallback context
    conv_hits = []
    if relation_sentence:
        try:
            conv_vec = embeddings.embed_query(relation_sentence)
            conv_results = client.query_points(
                collection_name=qdrant_db.collection_name, 
                query=conv_vec, 
                limit=3
            )
            conv_hits = conv_results.points
        except Exception as e:
            logger.warning(f"Qdrant Conversational Search Error: {e}")

    # --- Hybrid Score (混和加權) Logic ---
    # Give a score boost if an entity hit shares a subject/object with a relation hit.
    hybrid_scored_relations = []
    
    # Store entity names mapped to their similarity score
    entity_hit_scores = {e.payload.get("text"): e.score for e in entity_hits if e.payload.get("text")}
    
    for r in rel_hits:
        subj = r.payload.get("subject", "")
        obj = r.payload.get("object", "")
        base_score = r.score  # "如果跟邊相似度特別高 那就回傳這個邊" (Natural highly-scored edges will float to top)
        
        highest_matched_ent_score = 0
        if subj in entity_hit_scores:
            highest_matched_ent_score = max(highest_matched_ent_score, entity_hit_scores[subj])
        if obj in entity_hit_scores:
            highest_matched_ent_score = max(highest_matched_ent_score, entity_hit_scores[obj])
            
        # Boost based on "如果跟某節點相似度特別高 比如達到0.95 那也要回傳這個節點連接邊相似度高的"
        if highest_matched_ent_score >= 0.95:
            base_score += 0.8  # Massive boost to ensure connected edges of perfect entities are returned
        elif highest_matched_ent_score > 0:
            base_score += 0.2  # Standard hybrid overlap boost
            
        hybrid_scored_relations.append((base_score, r))
        
    hybrid_scored_relations.sort(key=lambda x: x[0], reverse=True)
    
    # Format top knowledge snippets
    for score, r in hybrid_scored_relations[:5]:
        rel_txt = r.payload.get("text", "")
        original_content = r.payload.get("original_content", "")
        
        if rel_txt:
            snippet = f"[權威醫學關聯 - 混和信心度 {score:.2f}]: {rel_txt}"
            if original_content:
                snippet += f"\n[原始完整醫學文獻]: {original_content}"
            retrieved_texts.append(snippet)
            
    for e in entity_hits[:3]:
        ent_name = e.payload.get("text", "")
        sn_name = e.payload.get("target_sn", "")
        if ent_name:
            retrieved_texts.append(f"[實體歸類 - 單點信心度 {e.score:.2f}]: '{ent_name}' 屬於超節點 '{sn_name}'")
            
    for c in conv_hits[:2]:
        conv_txt = c.payload.get("page_content", "")
        if conv_txt:
            retrieved_texts.append(f"[過往對話/其他紀錄 - 信心度 {c.score:.2f}]: {conv_txt}")

    # 4. Macro GraphRAG Traversal (Top-Down Multi-Hop Reasoning via SuperNodes)
    active_supernodes = set()
    for sn in supernode_hits:
        if sn.payload and sn.payload.get("text"):
            active_supernodes.add(sn.payload["text"])
            
    # Also grab SuperNodes from the top retrieved entities (Bottom-Up to Top-Down bridging)
    for e in entity_hits[:2]:
        sn_name = e.payload.get("target_sn", "")
        if sn_name and sn_name != "UNALIGNED":
            active_supernodes.add(sn_name)
            
    if active_supernodes:
        try:
            # Step A: Find Top 3 Semantically Similar Entities under these SuperNodes
            macro_ent_results = client.query_points(
                collection_name="pet_light_rag",
                query=ent_vec,
                limit=3,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value="entity")),
                        models.FieldCondition(key="target_sn", match=models.MatchAny(any=list(active_supernodes)))
                    ]
                )
            )
            macro_ent_names = [e.payload.get("text") for e in macro_ent_results.points if e.payload and e.payload.get("text")]
            
            # Step B: Find Top 3 Semantically Similar Relationships connected to these Top 5 Entities
            if macro_ent_names:
                # Use relation_sentence vector (conv_vec) if available, else fallback to entity vector (ent_vec)
                macro_query_vec = conv_vec if 'conv_vec' in locals() and conv_vec else ent_vec
                
                macro_rel_results = client.query_points(
                    collection_name="pet_light_rag",
                    query=macro_query_vec,
                    limit=3,
                    query_filter=models.Filter(
                        must=[models.FieldCondition(key="type", match=models.MatchValue(value="relationship"))],
                        should=[
                            models.FieldCondition(key="subject", match=models.MatchAny(any=macro_ent_names)),
                            models.FieldCondition(key="object", match=models.MatchAny(any=macro_ent_names))
                        ]
                    )
                )
                
                macro_texts = []
                for r in macro_rel_results.points:
                    if r.payload:
                        subj = r.payload.get("subject", "")
                        rel_sum = r.payload.get("text", "")
                        obj = r.payload.get("object", "")
                        if subj and rel_sum and obj:
                            macro_texts.append(f"({subj}) -[{rel_sum}]-> ({obj})")
                
                if macro_texts:
                    retrieved_texts.append(f"[巨觀圖譜多跳推理 (語意雙重過濾版)]: 關於大分類 {list(active_supernodes)} 裡，最符合情境的精準知識：\n" + "\n".join(macro_texts))
        except Exception as e:
            logger.warning(f"Qdrant Macro Traversal Error: {e}")

    # 5. Personal Memory Retrieval — fetch this user's personal triples from Neo4j
    user_id = state.get("user_id")
    if user_id:
        try:
            graph = neo4j_db.get_graph()
            if graph:
                personal_cypher = """
                MATCH (subject:Entity)-[r]->(object)
                WHERE r.user_id = $user_id
                RETURN subject.name AS subject_name, type(r) AS relation, r.type AS rel_detail, 
                       object.name AS object_name, labels(object) AS obj_labels, r.summary AS summary
                LIMIT 10
                """
                personal_results = graph.query(personal_cypher, {"user_id": user_id})
                if personal_results:
                    memory_lines = []
                    for rec in personal_results:
                        subj = rec.get("subject_name", "")
                        rel = rec.get("rel_detail") or rec.get("relation", "")
                        obj = rec.get("object_name", "")
                        summary = rec.get("summary", "")
                        
                        if subj and obj:
                            line = f"({subj}) -[{rel}]-> ({obj})"
                            if summary:
                                line += f" | 知識摘要: {summary}"
                            memory_lines.append(line)
                    if memory_lines:
                        retrieved_texts.append(f"[飼主個人記憶 — 此用戶的寵物檔案]:\n" + "\n".join(memory_lines))
                        logger.info(f"Retrieved {len(memory_lines)} personal memory triples for user {user_id[:8]}...")
        except Exception as e:
            logger.warning(f"Personal memory retrieval error: {e}")

    final_context = "\n".join(retrieved_texts)
    logger.info(f"Hybrid retrieval fetched {len(retrieved_texts)} snippets.")

    return {"retrieved_texts": final_context}

def generate_node(state: RagState) -> RagState:
    """Answers the query using context and retrieved documents via LLM."""
    query = state["query"]
    retrieved = state["retrieved_texts"]
    history = state.get("context", "")
    
    # ===== Clean structured logging =====
    logger.info("=" * 60)
    logger.info("[Generate] 最終送入 LLM 的完整輸入")
    logger.info("=" * 60)
    logger.info(f"[Generate] 飼主問題: {query}")
    logger.info("-" * 40)
    logger.info(f"[Generate] 對話歷史:\n{history if history else '(無)'}")
    logger.info("-" * 40)
    logger.info(f"[Generate] 檢索到的知識文件:\n{retrieved if retrieved else '(無檢索結果)'}")
    logger.info("=" * 60)

    template = """你是一位親切、專業的寵物貼身助理，說話的語氣像是飼主最信賴的好朋友。

重要規則：
1. 你的回答必須完全基於下方「檢索到的知識文件」中的內容。如果知識文件中沒有相關資訊，請誠實告知飼主你目前沒有這方面的資料，不要自行編造。
2. 【最重要】回覆必須是純文字，嚴格禁止使用任何格式化符號。禁止清單：** ** (粗體)、* *(斜體)、# (標題)、- (列表)、``` (代碼塊)、> (引用)、[] () (連結)。如果你想強調某個詞，直接用「」括起來即可。
3. 使用繁體中文回覆。
4. 語氣要溫暖親切，像貼身助理一樣關心飼主和寵物。
5. 如果有相關的圖片觀察結果在對話歷史中，請自然地融入你的回覆。

<對話歷史>
{history}
</對話歷史>

<檢索到的知識文件>
{retrieved}
</檢索到的知識文件>

<飼主的問題>
{query}
</飼主的問題>"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm_client.get_llm()
    result = chain.invoke({"history": history, "retrieved": retrieved, "query": query})

    # Post-processing: force-strip any residual markdown symbols
    import re
    answer = result.content
    answer = re.sub(r'\*\*(.+?)\*\*', r'「\1」', answer)  # **粗體** → 「粗體」
    answer = re.sub(r'\*(.+?)\*', r'\1', answer)           # *斜體* → 斜體
    answer = re.sub(r'^#{1,6}\s*', '', answer, flags=re.MULTILINE)  # # 標題 → 標題
    answer = re.sub(r'^[-•]\s*', '', answer, flags=re.MULTILINE)    # - 列表 → 列表
    answer = re.sub(r'```[\s\S]*?```', '', answer)          # 代碼塊移除
    answer = answer.strip()

    logger.info(f"[Generate] LLM 回覆: {answer[:200]}...")
    logger.info("=" * 60)

    return {"final_answer": answer}

# --- Graph Orchestration Definition ---
workflow = StateGraph(RagState)

workflow.add_node("multimodal", multimodal_node)
workflow.add_node("prompt_builder", prompt_builder_node)
workflow.add_node("hybrid_retrieve", retrieve_hybrid_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("multimodal")
workflow.add_edge("multimodal", "prompt_builder")
workflow.add_edge("prompt_builder", "hybrid_retrieve")
workflow.add_edge("hybrid_retrieve", "generate")
workflow.add_edge("generate", END)

rag_pipeline = workflow.compile()
