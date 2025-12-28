import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np


@dataclass
class ROI:
    x: int
    y: int
    width: int
    height: int

    def validate_within(self, frame_shape: tuple[int, ...]) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

        frame_height, frame_width = frame_shape[:2]
        if self.x < 0 or self.y < 0:
            raise ValueError("ROI coordinates must be non-negative")

        if self.x + self.width > frame_width or self.y + self.height > frame_height:
            raise ValueError("ROI must fit entirely within the frame bounds")


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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.templates = self._load_templates()
        self.logger = logging.getLogger(self.__class__.__name__)

    def reload_templates(self) -> None:
        """Reload templates from disk."""
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        if not self.templates_dir.exists():
            self.logger.warning("Templates directory not found: %s", self.templates_dir)
            return templates
        if not self.templates_dir.is_dir():
            self.logger.warning("Templates path is not a directory: %s", self.templates_dir)
            return templates

        for template_path in self.templates_dir.glob("*.png"):
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                self.logger.warning("Failed to load template: %s", template_path)
                continue
            templates[template_path.stem] = template
            self.logger.info("Loaded template: %s", template_path.stem)

        self.logger.info("Total templates loaded: %s", len(templates))
        return templates

    def match(self, roi: np.ndarray) -> Optional[str]:
        if len(self.templates) == 0:
            return None

        start_time = time.perf_counter()
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

        if best_score >= self.match_threshold:
            self.logger.debug("Matched %s with score %.2f", best_name, best_score)
            return best_name

        duration_ms = (time.perf_counter() - start_time) * 1000
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
        self.match_threshold = match_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        self.matcher = TemplateMatcher(templates_dir, match_threshold)

        self.video_capture: Optional[cv2.VideoCapture] = None
        self.is_local_file = self._is_local_file()

    def reload_templates(self) -> None:
        """Reload matcher templates (used after auto extraction)."""
        self.matcher.reload_templates()

    def _is_local_file(self) -> bool:
        """Check if stream_url is a local file path."""
        return Path(self.stream_url).exists()

    def capture_frame(self) -> np.ndarray:
        """Capture a frame from video file or stream."""
        start_time = time.monotonic()

        if self.is_local_file:
            if self.video_capture is None:
                self.logger.info("Opening video file: %s", self.stream_url)
                self.video_capture = cv2.VideoCapture(self.stream_url)
                if not self.video_capture.isOpened():
                    raise RuntimeError(f"Unable to open video file: {self.stream_url}")

            success, frame = self.video_capture.read()
            if not success or frame is None:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = self.video_capture.read()
                if not success or frame is None:
                    raise RuntimeError("Failed to capture frame from video")
        else:
            raise RuntimeError("Streaming not supported. Please use a local video file.")

        elapsed = time.monotonic() - start_time
        self.logger.info("Frame captured in %.2f seconds", elapsed)
        return frame

    def _crop_roi(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        roi.validate_within(frame.shape)
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    def read_game_state(self) -> GameState:
        frame = self.capture_frame()
        hero_left = self.matcher.match(self._crop_roi(frame, self.roi_hero_left))
        hero_right = self.matcher.match(self._crop_roi(frame, self.roi_hero_right))
        dealer_button_match = self.matcher.match(self._crop_roi(frame, self.roi_button))

        stack_roi = self._crop_roi(frame, self.roi_stack)
        stack_size = f"{stack_roi.shape[1]}x{stack_roi.shape[0]}px"

        game_state = GameState(
            hero_left=hero_left,
            hero_right=hero_right,
            stack_size=stack_size,
            dealer_button=dealer_button_match is not None,
        )
        self.logger.info("Game state: %s", game_state)
        return game_state

    def __del__(self) -> None:
        """Cleanup video capture."""
        if self.video_capture is not None:
            self.video_capture.release()
