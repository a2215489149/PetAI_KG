import os
import json
import logging
from typing import List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

def normalize_text(s: str) -> str:
    return s.strip()

class KnowledgeGraph:
    GLOBAL_RELATIONS = {"品種"}
    RELATION_SCHEMAS = {
        "品種": {"edge": "HAS_BREED", "object_label": "Breed", "object_key": "name"},
    }
    LOCAL_RELATION_EDGES = {
        "病史": "HAS_MEDICAL_HISTORY",
        "喜好": "LIKES",
        "喜歡": "LIKES",
        "年齡": "HAS_AGE",
        "個性": "HAS_PERSONALITY",
        "身型": "HAS_BODY_TYPE",
        "朋友": "HAS_FRIEND",
        "穿搭": "HAS_STYLE",
    }
    PROFILE_DIR = os.getenv("PROFILE_DIR", "profiles")

    def __init__(self, driver):
        self.driver = driver

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (b:Breed) REQUIRE b.name IS UNIQUE")
            # Drop old composite constraint if it exists (migration from user_id-on-node era)
            try:
                session.run("DROP CONSTRAINT constraint_entity_name_user_id IF EXISTS")
            except Exception:
                pass
            # Shared entity nodes — unique by name only
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            except Exception:
                pass  # May already exist

    def add_entity(self, tx, name, is_pet_name=False):
        """Creates or merges a shared entity node. All entities are global."""
        norm_name = normalize_text(name)
        if is_pet_name:
            tx.run(
                "MERGE (e:Entity {name: $name}) "
                "SET e.is_pet_name = true, e.pet_name = $name",
                name=norm_name
            )
        else:
            tx.run(
                "MERGE (e:Entity {name: $name})",
                name=norm_name
            )

    def add_global_entity(self, tx, label, key, value):
        q = f"MERGE (g:{label} {{{key}: $value}})"
        tx.run(q, value=normalize_text(value))

    def add_relation(self, tx, subject, relation, object_, user_id, summary=None):
        """Creates entities (shared) and a relationship edge carrying user_id."""
        rel = normalize_text(relation)
        subj = normalize_text(subject)
        obj  = normalize_text(object_)

        # Subject entity (shared node)
        is_pet = subj not in ("飼主", "寵物")
        self.add_entity(tx, subj, is_pet_name=is_pet)

        if rel in self.GLOBAL_RELATIONS:
            schema = self.RELATION_SCHEMAS.get(
                rel,
                {"edge": "RELATES_TO", "object_label": "Global", "object_key": "name"}
            )
            edge = schema["edge"]
            label = schema["object_label"]
            key = schema["object_key"]

            self.add_global_entity(tx, label, key, obj)

            # Edge carries user_id
            q = f"""
            MATCH (s:Entity {{name: $subj}})
            MATCH (o:{label} {{{key}: $obj}})
            MERGE (s)-[r:{edge} {{user_id: $user_id}}]->(o)
            """
            if summary:
                q += "\n            SET r.summary = $summary"
            tx.run(q, subj=subj, user_id=user_id, obj=obj, summary=summary)
        else:
            # Object entity (shared node)
            self.add_entity(tx, obj)
            edge = self.LOCAL_RELATION_EDGES.get(rel, "RELATES_TO")

            # Edge carries user_id — this is the ownership marker
            q = f"""
            MATCH (s:Entity {{name: $subj}}),
                  (o:Entity {{name: $obj}})
            MERGE (s)-[r:{edge} {{type: $rel, user_id: $user_id}}]->(o)
            """
            if summary:
                q += "\n            SET r.summary = $summary"
            tx.run(q, subj=subj, obj=obj, rel=rel, user_id=user_id, summary=summary)

    def update_or_create_pet_entity(self, tx, user_id, subject, relation, object_):
        """When a concrete pet name appears as subject (not '飼主'/'寵物'):
        1. Create the pet entity node (shared, is_pet_name=True)
        2. If this user has no known pet yet → set as primary pet in profile
        3. If user already has a known pet and this is a DIFFERENT name → additional pet
        4. Pet names are immutable once set in the profile.
        """
        self.add_entity(tx, subject, is_pet_name=True)

        profile = self._load_profile_json(user_id)
        pet_names = profile.get("pet_names", [])

        if subject not in pet_names:
            pet_names.append(subject)
            profile["pet_names"] = pet_names
            logger.info(f"[Pet] New pet '{subject}' registered for user {user_id[:8]}...")

        # First pet discovered → set as the default for '寵物' placeholder resolution
        if not profile.get("current_pet_name"):
            profile["current_pet_name"] = subject
            logger.info(f"[Pet] Primary pet set to '{subject}' for user {user_id[:8]}...")

        self._save_profile_json(user_id, profile)
        return {"subject": subject, "relation": relation, "object": object_}

    def update_pet_node_for_subject(self, tx, user_id, subject, relation, object_):
        """Resolves '寵物' placeholder to actual pet name.
        Strategy 1: JSON profile (fast, authoritative)
        Strategy 2: Neo4j edge traversal (fallback, confirms via user_id on edges)
        Once resolved, the pet name is locked in the profile.
        """
        if subject != "寵物":
            return {"subject": subject, "relation": relation, "object": object_}

        # Strategy 1: JSON profile — authoritative source
        profile = self._load_profile_json(user_id)
        known_pet = profile.get("current_pet_name")
        if known_pet:
            logger.info(f"Subject replaced: '寵物' -> '{known_pet}' (from profile)")
            return {"subject": known_pet, "relation": relation, "object": object_}

        # Strategy 2: Edge traversal — find any entity with is_pet_name=true
        # that has an edge carrying this user's user_id
        try:
            result = tx.run("""
                MATCH (p:Entity {is_pet_name: true})-[r {user_id: $user_id}]->()
                RETURN p.name AS pet_name LIMIT 1
            """, user_id=user_id)
            record = result.single()
            if record and record.get("pet_name"):
                pet_name = record["pet_name"]
                # Lock it in the profile so it's immutable going forward
                profile["current_pet_name"] = pet_name
                pet_names = profile.get("pet_names", [])
                if pet_name not in pet_names:
                    pet_names.append(pet_name)
                    profile["pet_names"] = pet_names
                self._save_profile_json(user_id, profile)
                logger.info(f"Subject replaced: '寵物' -> '{pet_name}' (from edge traversal, now locked)")
                return {"subject": pet_name, "relation": relation, "object": object_}
        except Exception as e:
            logger.warning(f"Edge-based pet resolution failed: {e}")

        # Fallback: keep "寵物" placeholder, create it as a shared node
        self.add_entity(tx, "寵物")
        logger.info("No known pet name found, using placeholder '寵物'")
        return {"subject": "寵物", "relation": relation, "object": object_}

    def _load_profile_json(self, user_id: str) -> dict:
        os.makedirs(self.PROFILE_DIR, exist_ok=True)
        path = os.path.join(self.PROFILE_DIR, f"{user_id}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except Exception as e:
                    logger.error(f"[Profile] JSON read failed user_id={user_id}: {e}")
                    return {"user_id": user_id, "entities": {}}
        else:
            return {"user_id": user_id, "entities": {}}

    def _save_profile_json(self, user_id: str, profile: dict):
        os.makedirs(self.PROFILE_DIR, exist_ok=True)
        path = os.path.join(self.PROFILE_DIR, f"{user_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    def update_profile_from_triples(self, user_id: str, triples):
        profile = self._load_profile_json(user_id)
        entities = profile.setdefault("entities", {})

        def migrate_placeholder_if_needed(target_name: str):
            if target_name in ("飼主", "寵物"): return
            if "寵物" not in entities: return

            placeholder = entities["寵物"]
            target = entities.setdefault(target_name, {"relations": {}})
            target_rel = target.setdefault("relations", {})
            ph_rel = placeholder.get("relations", {})

            for rel, objs in ph_rel.items():
                lst = target_rel.setdefault(rel, [])
                for o in objs:
                    if o not in lst:
                        lst.append(o)

            del entities["寵物"]
            logger.info(f"[Profile] Migrated placeholder '寵物' to '{target_name}'")

        for subject, relation, object_ in triples:
            subj = normalize_text(subject)
            rel = normalize_text(relation)
            obj  = normalize_text(object_)

            if not subj or not rel or not obj:
                continue

            migrate_placeholder_if_needed(subj)

            entity_data = entities.setdefault(subj, {"relations": {}})
            rel_map = entity_data.setdefault("relations", {})
            obj_list = rel_map.setdefault(rel, [])

            if obj not in obj_list:
                obj_list.append(obj)

        self._save_profile_json(user_id, profile)

# --- Standalone Functions ---

def align_entity_name(name: str, embeddings, q_client) -> Tuple[str, bool, Any]:
    """Returns (aligned_name, is_new, vector). Searches shared entities in pet_light_rag."""
    from qdrant_client.http import models
    if name in ("飼主", "寵物"):
        return name, False, None
    
    vector = embeddings.embed_query(name)
    try:
        response = q_client.query_points(
            collection_name="pet_light_rag",
            query=vector,
            limit=1,
            score_threshold=0.88,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="entity"))
                ]
            )
        )
        results = response.points
        if results:
            return results[0].payload.get("text", name), False, vector
        else:
            return name, True, vector
    except Exception as e:
        logger.warning(f"Entity alignment failed for {name}: {e}")
        return name, True, vector

def align_super_node(vector, q_client) -> str:
    """Checks if a new entity belongs to an existing Super Node category."""
    from qdrant_client.http import models
    try:
        response = q_client.query_points(
            collection_name="pet_light_rag",
            query=vector,
            limit=1,
            score_threshold=0.85,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="type", match=models.MatchValue(value="supernode"))
                ]
            )
        )
        results = response.points
        if results:
            return results[0].payload.get("text")
    except Exception as e:
        logger.warning(f"Super node alignment failed: {e}")
    return None

def write_entity_to_qdrant(name: str, vector, q_client):
    """Writes a shared entity to the unified pet_light_rag collection."""
    from qdrant_client.http import models
    new_id = os.urandom(16).hex()
    q_client.upsert(
        collection_name="pet_light_rag",
        points=[
            models.PointStruct(
                id=new_id,
                vector=vector,
                payload={"type": "entity", "text": name, "target_sn": "UNALIGNED"}
            )
        ]
    )

def process_triples_in_tx(tx, data, user_id, kg_instance: KnowledgeGraph, raw_dialogue_text: str, chunk_uuid: str):
    from core.llm_client import llm_client
    from core.qdrant_client import qdrant_db
    from core.pg_client import pg_db
    
    q_client = qdrant_db.get_client()
    embeddings = llm_client.get_embeddings()
    
    processed_triples = []
    for item in data:
        raw_subject = item.get("subject")
        raw_object = item.get("object")
        relation = item.get("relation")
        summary = item.get("summary")
        
        if not raw_subject or not relation or not raw_object:
            continue
            
        # 1. Similarity Entity Alignment (Subject) — no user_id filter, all entities shared
        subject, subj_is_new, subj_vec = align_entity_name(raw_subject, embeddings, q_client)
        if subj_is_new and subject not in ("飼主", "寵物"):
            write_entity_to_qdrant(subject, subj_vec, q_client)

        # 2. Determine pet placeholder resolution
        true_subject = subject
        if subject != "飼主" and subject != "寵物":
            resolved = kg_instance.update_or_create_pet_entity(tx, user_id, subject, relation, raw_object)
            true_subject = resolved["subject"]
        elif subject == "寵物":
            resolved = kg_instance.update_pet_node_for_subject(tx, user_id, subject, relation, raw_object)
            true_subject = resolved["subject"]

        # 3. Similarity Entity Alignment (Object) — shared entities, no user_id
        object_, obj_is_new, obj_vec = align_entity_name(raw_object, embeddings, q_client)
        
        is_deferred = False
        if obj_is_new:
            super_node_name = align_super_node(obj_vec, q_client)
            if super_node_name:
                logger.info(f"Object '{object_}' aligned to Super Node '{super_node_name}'. Permitting creation.")
                write_entity_to_qdrant(object_, obj_vec, q_client)
            else:
                logger.info(f"Object '{object_}' is OOV and has no Super Node. Deferring to Candidate Pool.")
                extended_dialogue = f"Triple: ({true_subject}, {relation}, {object_}) | Original Context: {raw_dialogue_text}"
                pg_db.insert_candidate(vector=obj_vec, uuid=chunk_uuid, dialog_text=extended_dialogue)
                is_deferred = True

        # 4. Construct Graph Relations — edge carries user_id
        if not is_deferred:
            kg_instance.add_relation(tx, true_subject, relation, object_, user_id, summary=summary)
            processed_triples.append({"subject": true_subject, "relation": relation, "object": object_, "deferred": False})
        else:
            kg_instance.add_entity(tx, true_subject, is_pet_name=(true_subject not in ("飼主", "寵物")))
            processed_triples.append({"subject": true_subject, "relation": relation, "object": object_, "deferred": True})

    return processed_triples
