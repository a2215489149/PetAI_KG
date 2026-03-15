"""
Recovery Script: Restores UNALIGNED entity candidates from Neo4j back into SQLite.

Usage:
    python scripts/recover_candidates.py

This script queries Neo4j for all Entity nodes that do NOT have a [:BELONGS_TO] 
relationship to any SuperNode, re-embeds their names, and inserts them back into 
the SQLite candidate_pool for HDBSCAN clustering.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import logging
from core.neo4j_client import neo4j_db
from core.pg_client import pg_db
from core.llm_client import llm_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def recover_candidates():
    graph = neo4j_db.get_graph()
    embeddings = llm_client.get_embeddings()
    
    if not graph:
        logger.error("Neo4j not connected. Cannot recover candidates.")
        return
    
    # Find all Entity nodes that have no BELONGS_TO edge to any SuperNode
    # Also exclude SuperNode nodes themselves and special placeholder nodes
    cypher = """
    MATCH (e:Entity)
    WHERE NOT (e)-[:BELONGS_TO]->(:SuperNode)
      AND NOT e:SuperNode
      AND e.source = 'lightrag'
    RETURN e.name AS name
    """
    results = graph.query(cypher)
    names = [r["name"] for r in results if r.get("name")]
    
    logger.info(f"Found {len(names)} unaligned entities in Neo4j to recover.")
    
    if not names:
        logger.info("No candidates to recover. Exiting.")
        return
    
    success_count = 0
    for i, name in enumerate(names):
        try:
            vec = embeddings.embed_query(name)
            candidate_uuid = str(uuid.uuid4())
            pg_db.insert_candidate(vector=vec, uuid=candidate_uuid, dialog_text=name)
            success_count += 1
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(names)} recovered...")
        except Exception as e:
            logger.error(f"Failed to recover '{name}': {e}")
    
    logger.info(f"Recovery complete! {success_count}/{len(names)} candidates restored to SQLite.")

if __name__ == "__main__":
    recover_candidates()
