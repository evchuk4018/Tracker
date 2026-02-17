# Pose estimation service — wraps MediaPipe Pose.
# Swapping models: subclass PoseService or replace with OpenPose / MMPose.

from __future__ import annotations

import logging
from typing import Any

import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

mp_pose = mp.solutions.pose

# Connections that form the stick-figure skeleton
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    # Head
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.NOSE),
]


class PoseService:
    """Human pose estimation using MediaPipe Pose."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        logger.info("Initialising MediaPipe Pose (complexity=%d)", model_complexity)
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
        )

    def estimate(self, image_rgb: np.ndarray) -> dict[str, Any] | None:
        """Return keypoints + connections for the first detected person.

        Returns dict with:
            keypoints: list of {id, name, x, y, z, visibility}
            connections: list of [id_a, id_b]
        or None if no pose found.
        """
        results = self.pose.process(image_rgb)
        if not results.pose_landmarks:
            return None

        h, w, _ = image_rgb.shape
        keypoints: list[dict[str, Any]] = []
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            keypoints.append(
                {
                    "id": idx,
                    "name": mp_pose.PoseLandmark(idx).name.lower(),
                    "x": round(lm.x * w, 1),
                    "y": round(lm.y * h, 1),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 3),
                }
            )

        connections = [[a.value, b.value] for a, b in SKELETON_CONNECTIONS]

        return {"keypoints": keypoints, "connections": connections}

    def close(self) -> None:
        self.pose.close()
