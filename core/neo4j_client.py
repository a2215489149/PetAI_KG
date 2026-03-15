import logging
from langchain_neo4j import Neo4jGraph
from config.settings import settings

logger = logging.getLogger(__name__)

class Neo4jDbClient:
    def __init__(self):
        self._graph = None
        self._initialized = False

    def _connect(self):
        """Lazy initialization — only connects to Neo4j on first actual use."""
        if self._initialized:
            return
        self._initialized = True
        try:
            self._graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                refresh_schema=False
            )
            logger.info("Neo4j connected successfully.")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Graph features will be unavailable.")
            self._graph = None

    def get_graph(self) -> Neo4jGraph:
        """Returns the Neo4j graph instance, connecting lazily on first call."""
        if not self._initialized:
            self._connect()
        return self._graph

neo4j_db = Neo4jDbClient()
