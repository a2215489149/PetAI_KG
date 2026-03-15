import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from core.llm_client import llm_client
from core.qdrant_client import qdrant_db
from core.neo4j_client import neo4j_db
import logging

try:
    import geoopt
except ImportError:
    geoopt = None

logger = logging.getLogger(__name__)

class HyperbolicPredictor(nn.Module):
    def __init__(self, embedding_dim: int, c=1.0):
        super().__init__()
        self.manifold = geoopt.PoincareBall(c=c)
        self.encoder = nn.Linear(embedding_dim, embedding_dim)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Projects euclidean features into the Poincare Ball."""
        x = self.encoder(x)
        # using exponential map from origin
        zero_origin = torch.zeros_like(x)
        return self.manifold.expmap(zero_origin, x)

    def dist(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Computes Poincare distance."""
        return self.manifold.dist(u, v)

def run_hyperbolic_computation(score_threshold=0.6):
    """
    SA-H2GT offline pipeline.
    1. Extracts node IDs and names from Neo4j.
    2. Fetches their Euclidean vectors directly from Qdrant (saving API cost).
    3. Maps onto Hyperbolic space (geoopt PoincareBall).
    4. Computes pairwise distances efficiently via PyTorch vectorization.
    5. Inserts PREDICTED_LINK dual relationships for highly related nodes.
    """
    if geoopt is None:
        logger.error("geoopt is not installed. Please install it to use Hyperbolic Link Prediction.")
        return

    logger.info("Initializing SA-H2GT Hyperbolic Link Predictor...")
    q_client = qdrant_db.get_client()
    n_graph = neo4j_db.get_graph()

    # Step 1: Query Neo4j for target entities
    cypher_entities = """
    MATCH (n:Entity) 
    RETURN id(n) as node_id, n.name as name
    """
    records = n_graph.query(cypher_entities)
    neo4j_entities = {r['name']: r['node_id'] for r in records}
    
    if len(neo4j_entities) < 2:
        logger.info("Not enough entities in Neo4j to predict links.")
        return

    # Step 2: Fetch vectors directly from Qdrant using Scroll API
    from qdrant_client.http import models
    logger.info("Fetching embedding vectors directly from Qdrant to save API calls...")
    
    qdrant_vectors = {}
    offset = None
    while True:
        try:
            scroll_result, next_page_offset = q_client.scroll(
                collection_name="pet_light_rag",
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="entity"))]
                ),
                limit=1000,
                offset=offset,
                with_vectors=True
            )
            for point in scroll_result:
                name = point.payload.get("text")
                if name and point.vector:
                    qdrant_vectors[name] = point.vector
                    
            if next_page_offset is None:
                break
            offset = next_page_offset
        except Exception as e:
            logger.error(f"Qdrant scroll failed: {e}")
            break

    # 3. Match Neo4j nodes with Qdrant vectors
    matched_entities = []
    embeddings_list = []
    
    for name, node_id in neo4j_entities.items():
        if name in qdrant_vectors:
            matched_entities.append((node_id, name))
            embeddings_list.append(qdrant_vectors[name])

    if len(matched_entities) < 2:
        logger.info("Not enough matching vectors between Neo4j and Qdrant to proceed.")
        return

    logger.info(f"Matched {len(matched_entities)} entities. Proceeding to Poincare projection...")

    # Step 4: Push into Geoopt manifold space
    euclidean_tensor = torch.tensor(embeddings_list, dtype=torch.float32)
    predictor = HyperbolicPredictor(embedding_dim=euclidean_tensor.size(1))
    
    predicted_links = []
    
    # 5. Compute pairwise distances using PyTorch tensor vectorization for extreme speed
    with torch.no_grad():
        hyperbolic_coords = predictor.project(euclidean_tensor)
        
        # calculate upper triangle of distance matrix efficiently
        # dist(u, v) using pdist or broadcasting
        u = hyperbolic_coords.unsqueeze(1) # shape: (N, 1, D)
        v = hyperbolic_coords.unsqueeze(0) # shape: (1, N, D)
        
        # dist_matrix will be (N, N)
        dist_matrix = predictor.dist(u, v)
        
        # Extract upper triangle indices where distance < score_threshold
        # offset=1 to exclude the diagonal (self-links)
        upper_tri_indices = torch.triu_indices(len(matched_entities), len(matched_entities), offset=1)
        
        for i, j in zip(upper_tri_indices[0], upper_tri_indices[1]):
            d = dist_matrix[i, j].item()
            if d < score_threshold:
                src_id, src_name = matched_entities[i]
                tgt_id, tgt_name = matched_entities[j]
                predicted_links.append((src_id, src_name, tgt_id, tgt_name, d))

    # Step 6: Write PREDICTED_LINK dual-write relationships back to Neo4j
    if predicted_links:
        logger.info(f"Found and writing {len(predicted_links)} PREDICTED_LINKs back to Neo4j.")
        write_cypher = """
        UNWIND $batch AS row
        MATCH (a:Entity) WHERE id(a) = row.source_id
        MATCH (b:Entity) WHERE id(b) = row.target_id
        MERGE (a)-[r:PREDICTED_LINK]->(b)
        SET r.weight = row.distance, r.source = 'SA-H2GT'
        """
        batch_data = [{"source_id": s_id, "target_id": t_id, "distance": d} for s_id, s_name, t_id, t_name, d in predicted_links]
        n_graph.query(write_cypher, {"batch": batch_data})
    else:
        logger.info("No links met the Hyperbolic threshold.")

    logger.info("Hyperbolic Link Prediction completed successfully.")

if __name__ == "__main__":
    run_hyperbolic_computation()
