from fastapi import APIRouter, BackgroundTasks, Request, HTTPException
import logging

from core.line_bot import line_bot_app

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/callback")
async def handle_callback(request: Request, background_tasks: BackgroundTasks):
    """
    Handles incoming LINE webhook events natively using line-bot-sdk.
    """
    try:
        # line_bot_app validates the signature and triggers self._process_event internally
        result = await line_bot_app.handle_request(request, background_tasks)
        if result.get("status") == "error":
            raise HTTPException(status_code=400, detail=result.get("message"))
        
        return "OK"
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error handling webhook: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/approve_consensus/{review_id}")
async def approve_consensus(review_id: int):
    """
    Expert Approval Webhook: mounts the offline consensus into the active RAG.
    """
    import sqlite3
    import uuid
    import json
    from core.pg_client import pg_db
    from core.neo4j_client import neo4j_db
    from core.qdrant_client import qdrant_db
    from core.llm_client import llm_client
    
    try:
        with sqlite3.connect(pg_db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT node_id, entity_name, summary, old_uuids_json, status FROM pending_expert_review WHERE id = ?", (review_id,))
            row = cursor.fetchone()
            
            if not row:
                raise HTTPException(status_code=404, detail="Review ID not found")
                
            node_id, name, summary, uuids_str, status = row
            if status == "approved":
                return {"message": "Already approved."}
                
            old_uuids = json.loads(uuids_str)
            graph = neo4j_db.get_graph()
            q_client = qdrant_db.get_client()
            
            if not graph or not q_client:
                raise HTTPException(status_code=500, detail="DBs not connected.")
                
            # Embed the new summary
            new_uuid = str(uuid.uuid4())
            vector = llm_client.get_embeddings().embed_query(summary)
            q_client.upsert(
                collection_name=qdrant_db.collection_name, 
                points=[{"id": new_uuid, "vector": vector, "payload": {"type": "consensus", "page_content": summary, "neo4j_id": node_id}}]
            )
            
            # Neo4j: Write ConsensusFragment and mark old ones as consolidated
            cypher_mount = """
            MATCH (n) WHERE id(n) = $node_id
            CREATE (c:ConsensusFragment {chunk_uuid: $new_uuid, text: $summary, status: 'expert_verified'})
            CREATE (n)-[:HAS_CONSENSUS]->(c)
            WITH n
            MATCH (f:DialogueFragment)-[r:EXPERIENCED_BY]->(n) WHERE f.chunk_uuid IN $old_uuids
            SET f.consolidated = true
            """
            graph.query(cypher_mount, {
                "node_id": node_id, 
                "old_uuids": old_uuids, 
                "new_uuid": new_uuid, 
                "summary": summary
            })
            
            cursor.execute("UPDATE pending_expert_review SET status = 'approved' WHERE id = ?", (review_id,))
            conn.commit()
            
        return {"status": "success", "message": f"Consensus for '{name}' approved and mounted."}
    except Exception as e:
        logger.error(f"Failed to approve consensus {review_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
