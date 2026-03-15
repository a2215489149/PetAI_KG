import uuid
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neo4j_client import neo4j_db
from core.llm_client import llm_client
from core.qdrant_client import qdrant_db

logger = logging.getLogger(__name__)

def run_consolidation_loop(node_limit: int = 10):
    """
    Module 3.3: Consolidation & Expert Audit Loop
    Finds hot nodes with > N UUIDs.
    """
    logger.info("Starting Memory Consolidation & Expert Audit Loop...")
    graph = neo4j_db.get_graph()
    q_client = qdrant_db.get_client()
    llm = llm_client.get_llm()

    if not graph:
        logger.error("Neo4j not connected. Aborting consolidation.")
        return

    # 1. Cron Garbage Collection
    # Find active nodes that have more than `node_limit` Experience fragments connected
    # User requested: Do not consolidate Super Nodes.
    cypher_find_hot_nodes = f"""
    MATCH (n:Entity)<-[:EXPERIENCED_BY]-(f:DialogueFragment)
    WHERE NOT n:SuperNode AND (f.consolidated IS NULL OR f.consolidated = false)
    WITH n, collect(f) as fragments
    WHERE size(fragments) >= {node_limit}
    RETURN id(n) AS node_id, n.name AS name, [frag in fragments | frag.text] AS texts, [frag in fragments | frag.chunk_uuid] AS uuids
    """
    
    try:
        results = graph.query(cypher_find_hot_nodes)
        for record in results:
            node_id = record["node_id"]
            name = record["name"]
            texts = record["texts"]
            old_uuids = record["uuids"]
            
            logger.info(f"[STEP A] Shrinking node '{name}' (ID: {node_id}) with {len(texts)} fragments.")
            
            # Summarize texts via LLM
            prompt = f"Summarize the following user experiences into a concise community consensus about '{name}'. Fragments: {texts}"
            summary = llm.invoke(prompt).content
            
            # Send to SQLite Pending Review Queue
            from core.pg_client import pg_db
            queue_id = pg_db.insert_pending_review(node_id=node_id, entity_name=name, summary=summary, old_uuids=old_uuids)
            
            if queue_id:
                logger.info(f"Consolidation pending for node '{name}'. Replaced {len(texts)} fragments. Awaiting expert review (Queue ID: {queue_id}).")
                
                # Trigger LINE Bot for Expert Review
                from linebot import LineBotApi
                from linebot.models import TextSendMessage
                from config.settings import settings
                
                try:
                    line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
                    if line_bot_api and settings.EXPERT_REVIEW_GROUP_ID:
                        review_msg = (
                            f"【系統經驗收斂審核】\n"
                            f"發現實體「{name}」掛載了過多零碎經驗 ({len(texts)} 筆)。\n"
                            f"GPT 總結共識為：\n「{summary}」\n"
                            f"請至審核後台檢閱 (Review ID: {queue_id})。"
                        )
                        line_bot_api.push_message(settings.EXPERT_REVIEW_GROUP_ID, TextSendMessage(text=review_msg))
                except Exception as line_e:
                    logger.warning(f"LINE push failed for consolidation review: {line_e}")
            else:
                logger.error(f"Failed to queue consolidation for '{name}'.")
                
    except Exception as e:
        logger.error(f"Consolidation Error: {e}")

if __name__ == "__main__":
    run_consolidation_loop()
