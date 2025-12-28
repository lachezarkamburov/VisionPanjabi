import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from vision_agent import ROI, TemplateMatcher, VisionAgent


@dataclass
class TableROISet:
    hero_left: ROI
    hero_right: ROI
    stack: ROI
    dealer_button: ROI


@dataclass
class TableState:
    table_id: str
    hero_left: Optional[str]
    hero_right: Optional[str]
    stack_size: Optional[str]
    dealer_button: bool

    def as_dict(self) -> Dict[str, object]:
        return {
            "cards": [self.hero_left, self.hero_right],
            "stack_size": self.stack_size,
            "dealer_button": self.dealer_button,
        }


class MultiTableVision:
    def __init__(
        self,
        stream_url: str,
        templates_dir: Path,
        base_rois: TableROISet,
        auto_detect: bool = True,
        max_tables: int = 6,
        manual_layouts: Optional[List[Dict[str, int]]] = None,
        match_threshold: float = 0.92,
    ) -> None:
        self.stream_url = stream_url
        self.base_rois = base_rois
        self.auto_detect = auto_detect
        self.max_tables = max_tables
        self.manual_layouts = manual_layouts or []
        self.matcher = TemplateMatcher(templates_dir, match_threshold)
        self.capture_agent = VisionAgent(
            stream_url=stream_url,
            templates_dir=templates_dir,
            roi_hero_left=base_rois.hero_left,
            roi_hero_right=base_rois.hero_right,
            roi_stack=base_rois.stack,
            roi_button=base_rois.dealer_button,
            match_threshold=match_threshold,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def capture_frame(self) -> np.ndarray:
        return self.capture_agent.capture_frame()

    def _detect_tables(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(grayscale, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 200 or h < 150:
                continue
            candidates.append((x, y, w, h))
        candidates.sort(key=lambda item: item[2] * item[3], reverse=True)
        return candidates[: self.max_tables]

    def _layouts(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.auto_detect:
            detected = self._detect_tables(frame)
            if detected:
                return detected
        layouts: List[Tuple[int, int, int, int]] = []
        for layout in self.manual_layouts:
            layouts.append((layout["x"], layout["y"], layout["width"], layout["height"]))
        return layouts

    def _offset_roi(self, roi: ROI, origin: Tuple[int, int]) -> ROI:
        return ROI(
            x=origin[0] + roi.x,
            y=origin[1] + roi.y,
            width=roi.width,
            height=roi.height,
        )

    def _crop_roi(self, frame: np.ndarray, roi: ROI) -> np.ndarray:
        return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]

    def _read_table(self, frame: np.ndarray, table_id: str, origin: Tuple[int, int]) -> TableState:
        hero_left_roi = self._offset_roi(self.base_rois.hero_left, origin)
        hero_right_roi = self._offset_roi(self.base_rois.hero_right, origin)
        stack_roi = self._offset_roi(self.base_rois.stack, origin)
        button_roi = self._offset_roi(self.base_rois.dealer_button, origin)

        hero_left = self.matcher.match(self._crop_roi(frame, hero_left_roi))
        hero_right = self.matcher.match(self._crop_roi(frame, hero_right_roi))
        dealer_button_match = self.matcher.match(self._crop_roi(frame, button_roi))
        stack_crop = self._crop_roi(frame, stack_roi)
        stack_size = f"{stack_crop.shape[1]}px"

        return TableState(
            table_id=table_id,
            hero_left=hero_left,
            hero_right=hero_right,
            stack_size=stack_size,
            dealer_button=dealer_button_match is not None,
        )

    def read_all_tables(self) -> Dict[str, Dict[str, object]]:
        frame = self.capture_frame()
        layouts = self._layouts(frame)
        results: Dict[str, Dict[str, object]] = {}
        for index, (x, y, _w, _h) in enumerate(layouts, start=1):
            table_state = self._read_table(frame, f"table_{index}", (x, y))
            results[table_state.table_id] = table_state.as_dict()
        self.logger.info("Detected %s tables", len(results))
        return results

    def to_json(self, data: Dict[str, Dict[str, object]]) -> str:
        return json.dumps(data, indent=2)
