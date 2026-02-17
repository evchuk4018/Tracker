# API models (Pydantic schemas)

from __future__ import annotations

from pydantic import BaseModel, Field


class VideoAnalysisResponse(BaseModel):
    """Response from video analysis endpoint."""
    download_url: str = Field(..., description="URL to download the annotated MP4")
    fps: float = Field(..., description="Frames per second of the input video")
    total_frames: int = Field(..., description="Total number of frames processed")
    duration_seconds: float = Field(..., description="Duration of the video in seconds")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    detection_summary: dict[str, int] = Field(
        default_factory=dict,
        description="Count of each detection label across all frames",
    )
    frames_with_pose: int = Field(0, description="Number of frames where pose was detected")
    processing_time_seconds: float = Field(..., description="Total server-side processing time")
