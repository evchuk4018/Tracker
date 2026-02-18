# FastAPI application entry point

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routers import analysis

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

app = FastAPI(
    title="Weightlifting Scene Analyzer",
    version="1.0.0",
    description="Upload a gym video → get annotated MP4 with bounding boxes, labels, and stick-figure pose overlay.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve annotated result files
uploads_dir = Path(__file__).resolve().parent.parent / "uploads"
uploads_dir.mkdir(exist_ok=True)
(uploads_dir / "results").mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

app.include_router(analysis.router, prefix="/api", tags=["analysis"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def _start_session_reaper():
    """Periodically clean up abandoned streaming processing sessions."""

    async def _reaper():
        while True:
            await asyncio.sleep(60)
            now = time.time()
            stale = []
            with analysis._sessions_lock:
                for _sid, session in analysis._active_sessions.items():
                    if now - session.start_time > 600:  # 10 minutes
                        stale.append(session)
            for session in stale:
                logging.getLogger(__name__).warning(
                    "Reaping stale session %s", session.session_id,
                )
                session.cleanup()

    asyncio.create_task(_reaper())
