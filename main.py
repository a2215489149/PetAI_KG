import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routes import router
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Import evolution jobs Let's assume they are fast synchronous for our demo
from evolution.entity_evolution import evolve_unknown_entities
from evolution.consolidation import run_consolidation_loop
from evolution.hyperbolic_prediction import run_hyperbolic_computation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

async def wrap_sync_job(func):
    """Utility to run synchronous evolution jobs within the async scheduler threadpool"""
    import asyncio
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, func)

async def job_entity_evolution():
    logger.info("[Scheduler] Triggering HDBSCAN Entity Evolution...")
    await wrap_sync_job(evolve_unknown_entities)

async def job_consolidation():
    logger.info("[Scheduler] Triggering Memory Consolidation...")
    await wrap_sync_job(run_consolidation_loop)

async def job_hyperbolic_prediction():
    logger.info("[Scheduler] Triggering Hyperbolic Link Prediction...")
    await wrap_sync_job(run_hyperbolic_computation)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # For demo purposes, we run them very frequently (every 10-30 minutes)
    # In production, use CronTrigger(hour=2, minute=0) etc.
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        job_entity_evolution, 
        trigger=IntervalTrigger(minutes=15),
        id="entity_evolution"
    )
    scheduler.add_job(
        job_consolidation, 
        trigger=IntervalTrigger(minutes=30),
        id="memory_consolidation"
    )
    scheduler.add_job(
        job_hyperbolic_prediction, 
        trigger=IntervalTrigger(minutes=60),
        id="hyperbolic_prediction"
    )
    scheduler.start()
    logger.info("APScheduler background tasks started.")
    yield  # Application runs here
    logger.info("Shutting down APScheduler...")
    scheduler.shutdown()

app = FastAPI(title="0301_petAI", description="Pet AI Evolving Memory System", lifespan=lifespan)

# Include webhook router
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
