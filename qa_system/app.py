from fastapi import FastAPI
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    try:
        # Startup
        yield
    finally:
        # Shutdown
        await document_store.cleanup()
        await vector_store.cleanup()
        logger.info("Successfully cleaned up all resources")

app = FastAPI(lifespan=lifespan) 