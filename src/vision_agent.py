import logging
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
        self.templates = self._load_templates()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_templates(self) -> Dict[str, np.ndarray]:
        templates: Dict[str, np.ndarray] = {}
        for template_path in self.templates_dir.glob("*.png"):
            template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
            if template is None:
                continue
            templates[template_path.stem] = template
        return templates

    def _get_stream_url(self) -> str:
        streams = streamlink.streams(self.stream_url)
        if not streams:
            raise RuntimeError("No streams found for the provided URL.")
        stream = streams.get("best") or next(iter(streams.values()))
        return stream.to_url()

    def _capture_frame(self) -> np.ndarray:
        stream_url = self._get_stream_url()
        capture = cv2.VideoCapture(stream_url)
        if not capture.isOpened():
            raise RuntimeError("Unable to open the stream via OpenCV.")
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            raise RuntimeError("Failed to capture frame from stream.")
        return frame

    def _crop_roi(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    def _match_template(self, roi: np.ndarray) -> Optional[str]:
        best_name = None
        best_score = 0.0
        for name, template in self.templates.items():
            if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
                continue
            result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > best_score:
                best_score = max_val
                best_name = name
        if best_score >= self.match_threshold:
            return best_name
        return None

    def read_game_state(self) -> GameState:
        frame = self._capture_frame()
        hero_left = self._match_template(self._crop_roi(frame, self.roi_hero_left))
        hero_right = self._match_template(self._crop_roi(frame, self.roi_hero_right))
        dealer_button_match = self._match_template(
            self._crop_roi(frame, self.roi_button)
        )
        stack_roi = self._crop_roi(frame, self.roi_stack)
        stack_size = f"{stack_roi.shape[1]}px"
        game_state = GameState(
            hero_left=hero_left,
            hero_right=hero_right,
            stack_size=stack_size,
            dealer_button=dealer_button_match is not None,
        )
        self.logger.info("Captured game state: %s", game_state)
        return game_state
