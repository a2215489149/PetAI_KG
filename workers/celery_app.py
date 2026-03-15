from celery import Celery
from config.settings import settings

celery_app = Celery(
    "pet_ai_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Load tasks
import workers.anchoring_task
