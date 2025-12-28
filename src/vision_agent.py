import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import streamlink


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int


@dataclass
class GameState:
    hero_left: Optional[str]
    hero_right: Optional[str]
    stack_size: Optional[str]
    dealer_button: bool


class TemplateMatcher:
    def __init__(self, templates_dir: Path, match_threshold: float = 0.92) -> None:
        self.templates_dir = templates_dir
        self.match_threshold = match_threshold
        self.templates = self._load_templates()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for template_path in self.templates_dir.glob("*.png"):
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                self.logger.warning(
                    "Template %s could not be read and will be skipped", template_path
                )
                continue
            templates[template_path.stem] = template
            self.logger.debug(
                "Loaded template",
                extra={"event": "template_loaded", "template": template_path.stem},
            )
        return templates

    def match(self, roi: np.ndarray) -> Optional[str]:
        start = time.perf_counter()
        best_name = None
        best_score = 0.0
        for name, template in self.templates.items():
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                self.logger.debug(
                    "Skipping template due to ROI size",
                    extra={
                        "event": "template_skipped",
                        "template": name,
                        "roi_shape": roi.shape,
                        "template_shape": template.shape,
                    },
                )
                continue
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_name = name
        duration_ms = (time.perf_counter() - start) * 1000
        if best_score >= self.match_threshold and best_name:
            self.logger.info(
                "Template match succeeded",
                extra={
                    "event": "template_match",
                    "template": best_name,
                    "score": round(best_score, 3),
                    "duration_ms": round(duration_ms, 2),
                },
            )
            return best_name

        self.logger.warning(
            "No template matched above threshold",
            extra={
                "event": "template_match_miss",
                "best_score": round(best_score, 3),
                "duration_ms": round(duration_ms, 2),
            },
        )
        return None


class VisionAgent:
    def __init__(
        self,
        stream_url: str,
        templates_dir: Path,
        roi_hero_left: ROI,
        roi_hero_right: ROI,
        roi_stack: ROI,
        roi_button: ROI,
        match_threshold: float = 0.92,
    ) -> None:
        self.stream_url = stream_url
        self.templates_dir = templates_dir
        self.roi_hero_left = roi_hero_left
        self.roi_hero_right = roi_hero_right
        self.roi_stack = roi_stack
        self.roi_button = roi_button
        self.matcher = TemplateMatcher(templates_dir, match_threshold)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.match_threshold = match_threshold

    def _get_stream_url(self) -> str:
        self.logger.debug(
            "Resolving stream URL", extra={"event": "streamlink_resolve_start"}
        )
        streams = streamlink.streams(self.stream_url)
        if not streams:
            self.logger.critical(
                "No streams available for URL", extra={"event": "streamlink_resolve"}
            )
            raise RuntimeError("No streams found for the provided URL.")
        stream = streams.get("best") or next(iter(streams.values()))
        resolved = stream.to_url()
        self.logger.info(
            "Resolved stream URL",
            extra={"event": "streamlink_resolve", "resolved_url": resolved},
        )
        return resolved

    def capture_frame(self) -> np.ndarray:
        start_time = time.perf_counter()
        self.logger.debug("Capturing frame", extra={"event": "frame_capture_start"})
        stream_url = self._get_stream_url()
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            self.logger.error(
                "Unable to open stream with OpenCV", extra={"event": "capture_error"}
            )
            raise RuntimeError("Unable to open the stream via OpenCV.")
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            self.logger.error(
                "Frame read returned empty", extra={"event": "capture_error"}
            )
            raise RuntimeError("Failed to capture frame from stream.")
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(
            "Captured frame from stream",
            extra={"event": "frame_capture_complete", "duration_ms": round(elapsed_ms, 2)},
        )
        return frame

    def _crop_roi(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    def read_game_state(self) -> GameState:
        start_time = time.perf_counter()
        frame = self.capture_frame()
        hero_left = self.matcher.match(self._crop_roi(frame, self.roi_hero_left))
        hero_right = self.matcher.match(self._crop_roi(frame, self.roi_hero_right))
        dealer_button_match = self.matcher.match(self._crop_roi(frame, self.roi_button))
        stack_roi = self._crop_roi(frame, self.roi_stack)
        stack_size = f"{stack_roi.shape[1]}px"
        game_state = GameState(
            hero_left=hero_left,
            hero_right=hero_right,
            stack_size=stack_size,
            dealer_button=dealer_button_match is not None,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.logger.info(
            "Captured game state",
            extra={
                "event": "game_state_captured",
                "duration_ms": round(duration_ms, 2),
                "hero_left": hero_left,
                "hero_right": hero_right,
                "dealer_button": game_state.dealer_button,
            },
        )
        if not hero_left or not hero_right:
            self.logger.warning(
                "Missing hero card detection",
                extra={
                    "event": "card_detection_incomplete",
                    "hero_left_found": bool(hero_left),
                    "hero_right_found": bool(hero_right),
                },
            )
        return game_state
