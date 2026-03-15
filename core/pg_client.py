import sqlite3
import json
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class PostgresConnector:
    def __init__(self):
        # Using SQLite to simulate Postgres for simplicity and portability in this demo
        self.db_path = "candidate_pool.db"
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS candidate_pool (
                        uuid TEXT PRIMARY KEY,
                        vector TEXT NOT NULL,
                        dialog_text TEXT NOT NULL
                    )
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pending_expert_review (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id INTEGER NOT NULL,
                        entity_name TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        old_uuids_json TEXT NOT NULL,
                        status TEXT DEFAULT 'pending'
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite candidate_pool: {e}")

    def insert_candidate(self, vector, uuid, dialog_text):
        """Module 2.3: Insert OOV (Cosine < 0.88) into candidate_pool"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Store vector as JSON string
                vector_str = json.dumps(vector)
                cursor.execute(
                    "INSERT OR REPLACE INTO candidate_pool (uuid, vector, dialog_text) VALUES (?, ?, ?)",
                    (uuid, vector_str, dialog_text)
                )
                conn.commit()
                logger.info(f"Inserted isolated vector {uuid} into candidate_pool.")
                return True
        except Exception as e:
            logger.error(f"Failed to insert candidate {uuid}: {e}")
            return False
    
    def fetch_isolated_vectors(self):
        """Module 3.1: Fetch OOV candidate pool for HDBSCAN"""
        logger.info("Fetching isolated vectors for HDBSCAN.")
        candidates = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT uuid, vector, dialog_text FROM candidate_pool")
                rows = cursor.fetchall()
                for row in rows:
                    uuid, vector_str, dialog_text = row
                    vector = json.loads(vector_str)
                    candidates.append({
                        "uuid": uuid,
                        "vector": vector,
                        "dialog_text": dialog_text
                    })
                return candidates
        except Exception as e:
            logger.error(f"Failed to fetch candidates: {e}")
            return []

    def clear_candidates(self, uuids):
        """Remove processed candidates from the pool"""
        if not uuids: return
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                placeholders = ','.join(['?'] * len(uuids))
                cursor.execute(f"DELETE FROM candidate_pool WHERE uuid IN ({placeholders})", uuids)
                conn.commit()
                logger.info(f"Cleared {len(uuids)} processed candidates from pool.")
        except Exception as e:
            logger.error(f"Failed to clear candidates: {e}")

    def insert_pending_review(self, node_id, entity_name, summary, old_uuids):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                uuids_json = json.dumps(old_uuids)
                cursor.execute(
                    "INSERT INTO pending_expert_review (node_id, entity_name, summary, old_uuids_json) VALUES (?, ?, ?, ?)",
                    (node_id, entity_name, summary, uuids_json)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Failed to insert pending review: {e}")
            return None

pg_db = PostgresConnector()
