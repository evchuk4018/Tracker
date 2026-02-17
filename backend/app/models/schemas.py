# API models (Pydantic schemas)

from __future__ import annotations

from pydantic import BaseModel


class DetectionResult(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    z: float
    visibility: float


class PoseResult(BaseModel):
    keypoints: list[Keypoint]
    connections: list[list[int]]


class AnalysisResponse(BaseModel):
    detections: list[DetectionResult]
    pose: PoseResult | None
    annotated_image_url: str
