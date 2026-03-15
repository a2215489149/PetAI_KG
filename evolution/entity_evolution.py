import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import hdbscan
import uuid

from core.llm_client import llm_client
from core.neo4j_client import neo4j_db
from core.qdrant_client import qdrant_db

logger = logging.getLogger(__name__)

def evolve_unknown_entities():
    """
    Cron Job Logic: Clusters unknown entities using HDBSCAN and creates new recognized topics in Neo4j.
    """
    logger.info("Starting Offline Entity Evolution...")
    
    # 1. Fetch real candidates from the database pool
    from core.pg_client import pg_db
    candidates = pg_db.fetch_isolated_vectors()
    
    if len(candidates) < 5:
        logger.info(f"Not enough candidates for clustering (found {len(candidates)}, need 5).")
        return

    candidate_vectors = [c["vector"] for c in candidates]
    candidate_texts = [c["dialog_text"] for c in candidates]
    candidate_uuids = [c["uuid"] for c in candidates]
    
    # 2. Density Clustering using cosine metric for embeddings
    # HDBSCAN expects a numpy array.
    vectors_np = np.array(candidate_vectors)
    try:
        # Cosine mapping via pairwise_distances is robust, or directly if metric='cosine' is supported.
        # But for 1536 dim, euclidean on normalized vectors is identical to cosine distance topology.
        # We will use metric='euclidean' since text embeddings (like OpenAI/Qwen) are usually normalized.
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean') # Production setting
        cluster_labels = clusterer.fit_predict(vectors_np)
    except Exception as e:
        logger.error(f"HDBSCAN clustering failed: {e}")
        return
    
    unique_labels = set(cluster_labels)
    llm = llm_client.get_llm()
    graph = neo4j_db.get_graph()

    processed_uuids = []

    for label in unique_labels:
        if label == -1:
             continue # Ignore noise
             
        # Find indices of this cluster
        indices = np.where(cluster_labels == label)[0]
        cluster_texts = [candidate_texts[i] for i in indices]
        cluster_uuids = [candidate_uuids[i] for i in indices]
        
        logger.info(f"Formed Cluster {label} with {len(cluster_texts)} items.")
        
        # 3. Name the proxy topic
        prompt = f"""
        你是一個寵物醫療與照護的本體論擴充引擎。以下是一群透過相似度演算法被歸類在一起的新詞彙或短語：
        詞彙群組：{cluster_texts}
        
        請你根據這些詞彙，總結出一個『高階、概括性的醫學、行為或照護大分類名稱』（字數限制在 8 個字以內）。
        絕對禁止使用具體的商品名稱、品牌名稱或太過瑣碎的詞彙。例如：不要回答「脈衝電磁墊」，請回答「寵物物理治療設備」；不要回答「渴望飼料」，請回答「貓狗無穀飼料」。
        只能輸出有代表性的該總結名稱的繁體中文，也可以是一個比較抽象的詞，不要輸出任何解釋字眼。
        """
        try:
            result = llm.invoke(prompt)
            entity_name = result.content.strip()
        except Exception as e:
            logger.error(f"LLM naming failed for cluster {label}: {e}")
            continue
        
        logger.info(f"LLM Assigned Topic Name: {entity_name}")
        
        # 4. Conjure new node in Neo4j and link constituent entities
        if graph:
             try:
                 cypher = """
                 MERGE (n:SuperNode {name: $name})
                 ON CREATE SET n:Entity, n.is_locked = false, n.source = 'evolved'
                 WITH n
                 UNWIND $entity_names AS e_name
                 MATCH (e:Entity {name: e_name})
                 MERGE (e)-[:BELONGS_TO]->(n)
                 """
                 graph.query(cypher, {"name": entity_name, "entity_names": cluster_texts})
                 logger.info(f"Successfully evolved {entity_name} in Neo4j. Mapped {len(cluster_texts)} component entities.")
             except Exception as e:
                 logger.error(f"Error writing evolved cluster {entity_name} to Neo4j: {e}")
                 continue
                 
             # Write the new Super Node to Qdrant
             try:
                 new_id = str(uuid.uuid4())
                 vec = llm_client.get_embeddings().embed_query(entity_name)
                 qdrant_db.get_client().upsert(
                     collection_name="pet_light_rag",
                     points=[{
                         "id": new_id,
                         "vector": vec,
                         "payload": {"type": "supernode", "text": entity_name}
                     }]
                 )
                 # Only after BOTH Neo4j and Qdrant succeed, mark for deletion
                 processed_uuids.extend(cluster_uuids)
                 logger.info(f"SuperNode {entity_name} fully persisted. {len(cluster_uuids)} candidates queued for cleanup.")
             except Exception as qe:
                 logger.error(f"Error writing SuperNode {entity_name} to Qdrant: {qe}. Candidates will NOT be deleted.")
                 
    # 5. Clear successfully clustered items from the candidate pool
    if processed_uuids:
        pg_db.clear_candidates(processed_uuids)
                 
if __name__ == "__main__":
    evolve_unknown_entities()
