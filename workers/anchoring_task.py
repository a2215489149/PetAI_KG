import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import json
import logging
from typing import List

from langchain_core.prompts import PromptTemplate
from core.llm_client import llm_client
from core.neo4j_client import neo4j_db
from core.qdrant_client import qdrant_db
from core.knowledge_graph import KnowledgeGraph, process_triples_in_tx

logger = logging.getLogger(__name__)

def extract_entities_and_relations(text: str) -> List[dict]:
    """Uses LLM to extract subject-relation-object triples from dialogue."""
    llm = llm_client.get_llm()
    system_prompt = (
        "你是一位資料分析師，你的專業是從訊息中精確的找出或推測出飼主或寵物的個人資訊(不'推測'品種，不蒐集飼主名稱)，並組成合理的三元關係\n"
        "請對訊息中[飼主針對自己與寵物]提取實體與關係，提取內容需要根據訊息'推測'或'明確寫出''飼主與寵物'各自的'個性'、'朋友'與'喜好'\n"
        "或是訊息明確表示的'疾病'、'年齡'、'品種'、'身型'、'穿搭'等資訊(這些屬於不可推測)，提取目的是用於未來的商品推薦與醫療照護，如果看不出以上需求資訊回傳空矩陣\n"
        "subject屬性中只能是'飼主'、'寵物'、'明確的寵物名稱(比如:小黑、牙牙、毛毛...等較可愛的代名詞通常才是)'這三個名詞，如果沒有明確指出寵物名稱，則以'寵物'代替輸出(品種不能是名稱)\n"
        "通常飼主再分享一個喜好個性等等時，如果用了'他'、'牠'、'它'等代名詞，通常是在指自己的寵物\n"
        "你需要根據我的要求提取正確的 JSON 格式回傳，\n"
        "輸出的 JSON 格式應該是這樣：\n"
        "[\n"
        "    {{\"subject\": \"小白\", \"relation\": \"品種\", \"object\": \"土狗\", \"summary\": \"小白是一隻品種為土狗的寵物\"}},\n"
        "    {{\"subject\": \"小黃\", \"relation\": \"病史\", \"object\": \"糖尿病\", \"summary\": \"小黃曾經有過糖尿病的病史\"}},\n"
        "    {{\"subject\": \"小黑\", \"relation\": \"年齡\", \"object\": \"三歲\", \"summary\": \"小黑今年大約三歲\"}},\n"
        "    {{\"subject\": \"寵物\", \"relation\": \"身型\", \"object\": \"中大型\", \"summary\": \"這隻寵物的體型屬於中大型\"}},\n"
        "    {{\"subject\": \"依依\", \"relation\": \"品種\", \"object\": \"黃金獵犬\", \"summary\": \"依依是一隻可愛的黃金獵犬\"}},\n"
        "    {{\"subject\": \"飼主\", \"relation\": \"注重\", \"object\": \"寵物健康\", \"summary\": \"飼主非常注重寵物的身體健康\"}},\n"
        "    {{\"subject\": \"飼主\", \"relation\": \"喜歡\", \"object\": \"分享知識\", \"summary\": \"飼主喜歡分享關於寵物的知識\"}}\n"
        "]\n"
        "JSON中的提示（如小白, 土狗）僅為示例，請依照用戶的真實訊息做提取。\n"
        "請根據以下訊息做提取，並僅使用繁體中文。回傳的內容必須是合法的 JSON 陣列，不要有 markdown 語法 (```json...```)。\n\n"
        "訊息: {text}"
    )
    prompt = PromptTemplate.from_template(system_prompt)
    chain = prompt | llm
    
    try:
        result = chain.invoke({"text": text})
        content = result.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        data = json.loads(content.strip())
        if isinstance(data, list) and all(all(k in x for k in ("subject", "relation", "object", "summary")) for x in data):
            # ===== Clean structured logging =====
            if data:
                logger.info("-" * 40)
                logger.info(f"[Anchoring] 成功提取 {len(data)} 組三元组知識")
                for i, t in enumerate(data):
                    logger.info(f"  {i+1}. ({t['subject']}) -[{t['relation']}]-> ({t['object']})")
                logger.info("-" * 40)
            return data
    except Exception as e:
        logger.error(f"Failed to extract entities/relations JSON: {e}")
    return []

def process_anchoring(user_id: str, text: str, ai_response: str):
    """
    Background Task: Dynamically anchors new information and populates the Knowledge Graph.
    """
    logger.info(f"Background Activity: Starting graph anchoring for [user:{user_id}]")
    try:
        # 1. Unconditionally Process and Save Entity Nodes (Triples)
        triples_data = extract_entities_and_relations(text)
        graph = neo4j_db.get_graph()
        
        new_chunk_uuid = str(uuid.uuid4())
        
        if triples_data and graph:
            kg_instance = KnowledgeGraph(driver=graph._driver)
            try:
                 kg_instance.create_constraints()
            except Exception as e:
                 logger.warning(f"Neo4j constraints creation skipped/failed: {e}")

            # Execute extraction logic inside a transaction unconditionally
            with kg_instance.driver.session() as session:
                processed_triples = session.execute_write(
                    process_triples_in_tx,
                    triples_data,
                    user_id,
                    kg_instance,
                    text,
                    new_chunk_uuid
                )
            if processed_triples:
                # Filter out deferred triples before updating profile
                active_triples = [(t["subject"], t["relation"], t["object"]) for t in processed_triples if not t["deferred"]]
                if active_triples:
                    kg_instance.update_profile_from_triples(user_id, active_triples)
                logger.info(f"Successfully processed {len(processed_triples)} relationships ({len(active_triples)} anchored, {len(processed_triples) - len(active_triples)} deferred) for user {user_id}")
        else:
            logger.info("No actionable entities/relations extracted or Graph DB unavailable.")

        # 2. Process Experience Fragments (Dialogue) with Expert Review
        q_client = qdrant_db.get_client()
        embeddings = llm_client.get_embeddings()
        
        if not q_client or not embeddings or not graph:
            return
            
        vector = embeddings.embed_query(text)
        search_results = []
        from qdrant_client.http import models
        try:
            search_response = q_client.query_points(
                collection_name=qdrant_db.collection_name,
                query=vector,
                limit=1,
                score_threshold=0.88,
                query_filter=models.Filter(
                    must_not=[
                         models.FieldCondition(key="type", match=models.MatchValue(value="entity"))
                    ]
                )
            )
            search_results = search_response.points
        except Exception as e:
            logger.warning(f"Qdrant Evaluation failed for experience: {e}")

        dialogue_text = f"User: {text}\nAI: {ai_response}"
        # new_chunk_uuid was generated higher up

        # Mount standard relationship in Neo4j (attach UUID and summary to the edges)
        if triples_data:
            for triple_dict in triples_data:
                subj = triple_dict["subject"]
                rel = triple_dict["relation"]
                obj = triple_dict["object"]
                summary = triple_dict["summary"]
                
                # Pre-generate UUID for this specific relationship (cross-ref Neo4j <-> Qdrant)
                rel_uuid = str(uuid.uuid4())
                
                # Identify if it's a global or local relation to choose the right edge type
                is_pet = subj not in ("飼主", "寵物")
                if rel in KnowledgeGraph.GLOBAL_RELATIONS:
                    rel_schema = KnowledgeGraph.RELATION_SCHEMAS.get(rel, {"edge": "RELATES_TO", "object_label": "Global", "object_key": "name"})
                    edge_type = rel_schema["edge"]
                    label = rel_schema["object_label"]
                    key = rel_schema["object_key"]
                    cypher_mount = f"""
                    MERGE (s:Entity {{name: $subj}})
                    ON CREATE SET s.is_pet_name = $is_pet
                    MERGE (o:{label} {{{key}: $obj}})
                    MERGE (s)-[r:{edge_type}]->(o)
                    SET r.type = $rel, r.user_id = $user_id, r.summary = $summary, 
                        r.source = 'dynamic_learning', r.qdrant_id = $rel_uuid
                    RETURN id(s) as s_id
                    """
                else:
                    edge_type = KnowledgeGraph.LOCAL_RELATION_EDGES.get(rel, "RELATED_TO")
                    cypher_mount = f"""
                    MERGE (s:Entity {{name: $subj}})
                    ON CREATE SET s.is_pet_name = $is_pet
                    MERGE (o:Entity {{name: $obj}})
                    MERGE (s)-[r:{edge_type}]->(o)
                    SET r.type = $rel, r.user_id = $user_id, r.summary = $summary, 
                        r.source = 'dynamic_learning', r.qdrant_id = $rel_uuid
                    RETURN id(s) as s_id
                    """
                
                try:
                    # Also ensure profile is updated if it's a new pet name
                    if is_pet:
                        with graph._driver.session() as session:
                            session.execute_write(kg_instance.update_or_create_pet_entity, user_id, subj, rel, obj)

                    result_records = graph.query(cypher_mount, {
                        "subj": subj, 
                        "obj": obj, 
                        "rel": rel, 
                        "user_id": user_id, 
                        "summary": summary,
                        "rel_uuid": rel_uuid,
                        "is_pet": is_pet
                    })
                    
                    # Also write to Qdrant immediately for this relationship
                    # Following LightRAG format: "{subject} - [{summary}] -> {object}"
                    rel_text = f"{subj} - [{summary}] -> {obj}"
                    rel_vector = embeddings.embed_query(rel_text)
                    
                    q_client.upsert(
                        collection_name=qdrant_db.collection_name,
                        points=[
                            {
                                "id": rel_uuid,
                                "vector": rel_vector,
                                "payload": {
                                    "text": rel_text,
                                    "type": "relationship",
                                    "subject": subj,
                                    "object": obj,
                                    "summary": summary,
                                    "source_id": user_id,
                                    "original_content": dialogue_text
                                }
                            }
                        ]
                    )
                    logger.info(f"Anchored and Vectorized knowledge: {rel_text}")
                except Exception as re:
                    logger.warning(f"Failed to mount/vectorize relationship: {re}")

        # 3. Save the full dialogue as an 'experience' fragment (Experience tracking)
        try:
            exp_uuid = str(uuid.uuid4())
            q_client.upsert(
                collection_name=qdrant_db.collection_name,
                points=[
                    {
                        "id": exp_uuid,
                        "vector": vector, # Original text vector
                        "payload": {
                            "type": "experience",
                            "page_content": dialogue_text,
                            "user_id": user_id
                        }
                    }
                ]
            )
        except Exception as e:
            logger.error(f"Failed to save experience fragment: {e}")

    except Exception as e:
        logger.error(f"Error in async anchoring background task: {e}")
