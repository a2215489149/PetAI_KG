import logging
import redis
from config.settings import settings

logger = logging.getLogger(__name__)

class RedisClient:
    def __init__(self):
        try:
            self.r = redis.from_url(settings.REDIS_URL, decode_responses=True)
            self.r.ping()  # Test connection immediately
            self._connected = True
            logger.info("Redis connected successfully.")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to in-memory cache.")
            self.r = None
            self._connected = False
            self._fallback = {}

    def set(self, key: str, value: str, ex: int = 3600):
        if self._connected:
            try:
                self.r.set(key, value, ex=ex)
            except Exception as e:
                logger.warning(f"Redis SET failed: {e}")
                self._fallback[key] = value
        else:
            self._fallback[key] = value

    def get(self, key: str) -> str:
        if self._connected:
            try:
                return self.r.get(key)
            except Exception as e:
                logger.warning(f"Redis GET failed: {e}")
                return self._fallback.get(key)
        else:
            return self._fallback.get(key)

    def delete(self, key: str):
        if self._connected:
            try:
                self.r.delete(key)
            except Exception as e:
                logger.warning(f"Redis DEL failed: {e}")
                self._fallback.pop(key, None)
        else:
            self._fallback.pop(key, None)

    def exists(self, key: str) -> bool:
        if self._connected:
            try:
                return bool(self.r.exists(key))
            except Exception as e:
                logger.warning(f"Redis EXISTS failed: {e}")
                return key in self._fallback
        else:
            return key in self._fallback
        
redis_db = RedisClient()
