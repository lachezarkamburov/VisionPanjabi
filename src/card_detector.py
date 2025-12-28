"""Automatic card detection using computer vision.

This module detects hero card regions of interest (ROIs) and table layouts
from poker video frames so the system can run without manual coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectedCard:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def as_dict(self) -> Dict[str, int | float]:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
            "confidence": float(self.confidence),
        }


class AutoCardDetector:
    """Automatically detect card positions in poker video frames."""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.frame: np.ndarray | None = None

    def load_frame(self) -> np.ndarray:
        """Load the first frame from the video file."""
        cap = cv2.VideoCapture(self.video_path)
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            raise RuntimeError(f"Failed to load video: {self.video_path}")

        self.frame = frame
        logger.info("Loaded frame: %sx%s", frame.shape[1], frame.shape[0])
        return frame

    def _prepare_debug_dir(self) -> Path:
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        return debug_dir

    def _detect_cards_edges(self, frame: np.ndarray, debug: bool = False) -> List[DetectedCard]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_cards: List[DetectedCard] = []
        frame_area = frame.shape[0] * frame.shape[1]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w == 0:
                continue

            aspect_ratio = h / w
            area = w * h
            relative_area = area / frame_area

            if not (0.8 <= aspect_ratio <= 2.0):
                continue
            if not (0.001 <= relative_area <= 0.10):
                continue
            if w < 30 or h < 40:
                continue

            ideal_aspect = 1.4
            aspect_confidence = 1.0 - abs(aspect_ratio - ideal_aspect) / ideal_aspect
            contour_area = cv2.contourArea(contour)
            rectangularity = contour_area / area if area > 0 else 0
            confidence = aspect_confidence * 0.6 + rectangularity * 0.4

            if confidence > 0.5:
                detected_cards.append(
                    DetectedCard(x=int(x), y=int(y), width=int(w), height=int(h), confidence=confidence)
                )

        detected_cards.sort(key=lambda c: c.confidence, reverse=True)

        if debug:
            debug_dir = self._prepare_debug_dir()
            cv2.imwrite(str(debug_dir / "01_original.png"), frame)
            cv2.imwrite(str(debug_dir / "02_grayscale.png"), gray)
            cv2.imwrite(str(debug_dir / "03_blurred.png"), blurred)
            cv2.imwrite(str(debug_dir / "04_edges.png"), edges)
            cv2.imwrite(str(debug_dir / "05_closed.png"), closed)

            debug_contours = frame.copy()
            cv2.drawContours(debug_contours, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(str(debug_dir / "06_all_contours.png"), debug_contours)

            self._save_debug_image(frame, [card.as_dict() for card in detected_cards], debug_dir / "07_detected_cards.png")
            logger.info("Debug images saved to debug/ folder")

        logger.info("Detected %s potential cards", len(detected_cards))
        return detected_cards

    def detect_cards_by_color(self, frame: np.ndarray, debug: bool = False) -> List[DetectedCard]:
        """
        Alternative detection using color thresholding.
        Playing cards are typically white/light colored.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_cards: List[DetectedCard] = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0
            area = w * h

            if 0.8 <= aspect_ratio <= 2.5 and area > 1000 and w >= 25 and h >= 35:
                detected_cards.append(
                    DetectedCard(x=int(x), y=int(y), width=int(w), height=int(h), confidence=0.8)
                )

        if debug:
            debug_dir = self._prepare_debug_dir()
            cv2.imwrite(str(debug_dir / "color_mask.png"), mask)

        return detected_cards

    def detect_cards(self, frame: np.ndarray, debug: bool = False) -> List[Dict[str, int | float]]:
        """Try edge detection first, fallback to color detection."""
        edge_cards = self._detect_cards_edges(frame, debug=debug)

        if len(edge_cards) >= 4:
            logger.info("Edge detection found %s cards", len(edge_cards))
            detected_cards = edge_cards
        else:
            logger.warning("Edge detection insufficient, trying color detection")
            color_cards = self.detect_cards_by_color(frame, debug=debug)

            if len(color_cards) >= 4:
                logger.info("Color detection found %s cards", len(color_cards))
                detected_cards = color_cards
            else:
                detected_cards = edge_cards + color_cards
                logger.warning("Combined detection found %s cards", len(detected_cards))

        if debug:
            debug_dir = self._prepare_debug_dir()
            self._save_debug_image(frame, [card.as_dict() for card in detected_cards], debug_dir / "07_detected_cards.png")

        return [card.as_dict() for card in detected_cards]

    def detect_card_pairs(
        self, frame: np.ndarray, debug: bool = False
    ) -> List[Tuple[Dict[str, int], Dict[str, int]]]:
        """Detect pairs of cards (hero cards) that are horizontally adjacent."""
        all_cards = self.detect_cards(frame, debug=debug)
        if len(all_cards) < 2:
            logger.warning("Not enough cards detected to find pairs")
            return []

        pairs: List[Tuple[Dict[str, int], Dict[str, int]]] = []
        used_cards: set[int] = set()

        for i, card1 in enumerate(all_cards):
            if i in used_cards:
                continue
            for j, card2 in enumerate(all_cards):
                if j <= i or j in used_cards:
                    continue

                y_diff = abs(card1["y"] - card2["y"])
                avg_height = (card1["height"] + card2["height"]) / 2

                if card1["x"] < card2["x"]:
                    left, right = card1, card2
                else:
                    left, right = card2, card1

                x_gap = right["x"] - (left["x"] + left["width"])
                avg_width = (left["width"] + right["width"]) / 2

                if (
                    y_diff < avg_height * 0.2
                    and 0 <= x_gap < avg_width * 0.5
                    and 0.7 <= left["width"] / right["width"] <= 1.3
                ):
                    pairs.append((left, right))
                    used_cards.update({i, j})
                    break

        logger.info("Found %s card pairs", len(pairs))
        if debug and pairs:
            debug_dir = self._prepare_debug_dir()
            self._save_debug_pairs(frame, pairs, debug_dir / "debug_card_pairs.png")
        return pairs

    def detect_tables_and_cards(
        self, frame: np.ndarray, num_tables: int = 4, debug: bool = False
    ) -> List[Dict[str, Dict[str, int]]]:
        """
        Detect poker tables and their card positions.

        Returns a list of table configurations with layouts and hero card ROIs
        relative to each table region.
        """
        height, width = frame.shape[:2]
        pairs = self.detect_card_pairs(frame, debug=debug)

        if len(pairs) < num_tables:
            logger.warning("Expected %s card pairs but found %s", num_tables, len(pairs))

        tables: List[Dict[str, Dict[str, int]]] = []
        table_width = width // 2 if num_tables >= 2 else width
        table_height = height // 2 if num_tables >= 2 else height

        grid_positions = self._grid_positions(num_tables, table_width, table_height)

        for index, (grid_x, grid_y) in enumerate(grid_positions):
            pair_in_cell = None
            for left, right in pairs:
                card_center_x = (left["x"] + right["x"] + right["width"]) / 2
                card_center_y = (left["y"] + right["y"] + right["height"]) / 2
                if grid_x <= card_center_x < grid_x + table_width and grid_y <= card_center_y < grid_y + table_height:
                    pair_in_cell = (left, right)
                    break

            if pair_in_cell:
                left, right = pair_in_cell
                tables.append(
                    {
                        "layout": {
                            "x": grid_x,
                            "y": grid_y,
                            "width": table_width,
                            "height": table_height,
                        },
                        "hero_left": {
                            "x": left["x"] - grid_x,
                            "y": left["y"] - grid_y,
                            "width": left["width"],
                            "height": left["height"],
                        },
                        "hero_right": {
                            "x": right["x"] - grid_x,
                            "y": right["y"] - grid_y,
                            "width": right["width"],
                            "height": right["height"],
                        },
                    }
                )
            else:
                logger.warning("No cards detected in table position %s", index + 1)

        logger.info("Configured %s tables with card positions", len(tables))
        if debug and tables:
            debug_dir = self._prepare_debug_dir()
            self._save_debug_tables(frame, tables, debug_dir / "debug_tables.png")
        return tables

    def _grid_positions(self, num_tables: int, table_width: int, table_height: int) -> List[Tuple[int, int]]:
        if num_tables <= 1:
            return [(0, 0)]
        if num_tables == 2:
            return [(0, 0), (table_width, 0)]
        if num_tables == 3:
            return [(0, 0), (table_width, 0), (0, table_height)]
        return [
            (0, 0),
            (table_width, 0),
            (0, table_height),
            (table_width, table_height),
        ]

    def _save_debug_image(self, frame: np.ndarray, cards: List[Dict[str, int | float]], output_path: Path) -> None:
        annotated = frame.copy()
        for card in cards:
            cv2.rectangle(
                annotated,
                (int(card["x"]), int(card["y"])),
                (int(card["x"] + card["width"]), int(card["y"] + card["height"])),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated,
                f"{card['confidence']:.2f}",
                (int(card["x"]), int(card["y"] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        cv2.imwrite(str(output_path), annotated)
        logger.info("Saved debug image to %s", output_path)

    def _save_debug_pairs(
        self, frame: np.ndarray, pairs: List[Tuple[Dict[str, int], Dict[str, int]]], output_path: Path
    ) -> None:
        annotated = frame.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for index, (left, right) in enumerate(pairs):
            color = colors[index % len(colors)]
            cv2.rectangle(
                annotated,
                (int(left["x"]), int(left["y"])),
                (int(left["x"] + left["width"]), int(left["y"] + left["height"])),
                color,
                2,
            )
            cv2.rectangle(
                annotated,
                (int(right["x"]), int(right["y"])),
                (int(right["x"] + right["width"]), int(right["y"] + right["height"])),
                color,
                2,
            )
            left_center = (int(left["x"] + left["width"] / 2), int(left["y"] + left["height"] / 2))
            right_center = (int(right["x"] + right["width"] / 2), int(right["y"] + right["height"] / 2))
            cv2.line(annotated, left_center, right_center, color, 2)
            cv2.putText(
                annotated,
                f"Pair {index + 1}",
                (int(left["x"]), int(left["y"] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.imwrite(str(output_path), annotated)
        logger.info("Saved card pairs debug image to %s", output_path)

    def _save_debug_tables(self, frame: np.ndarray, tables: List[Dict[str, Dict[str, int]]], output_path: Path) -> None:
        annotated = frame.copy()
        for index, table in enumerate(tables, start=1):
            layout = table["layout"]
            cv2.rectangle(
                annotated,
                (layout["x"], layout["y"]),
                (layout["x"] + layout["width"], layout["y"] + layout["height"]),
                (255, 255, 255),
                2,
            )
            for card_key in ("hero_left", "hero_right"):
                card = table[card_key]
                abs_x = layout["x"] + card["x"]
                abs_y = layout["y"] + card["y"]
                cv2.rectangle(
                    annotated,
                    (abs_x, abs_y),
                    (abs_x + card["width"], abs_y + card["height"]),
                    (0, 255, 0),
                    2,
                )
            cv2.putText(
                annotated,
                f"Table {index}",
                (layout["x"] + 10, layout["y"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        cv2.imwrite(str(output_path), annotated)
        logger.info("Saved table debug image to %s", output_path)


def main() -> None:
    """Run automatic detection and update config.yaml with detected ROIs."""
    import yaml

    config_path = Path("config.yaml")
    if not config_path.exists():
        raise FileNotFoundError("config.yaml not found in working directory")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    video_path = config["stream"]["url"]
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    detector = AutoCardDetector(video_path)
    frame = detector.load_frame()
    tables = detector.detect_tables_and_cards(frame, num_tables=config.get("multi_table", {}).get("max_tables", 4), debug=True)

    if not tables:
        print("\n❌ No cards detected! Try adjusting detection parameters or check your video.")
        return

    first_table = tables[0]
    hero_left = first_table["hero_left"]
    hero_right = first_table["hero_right"]

    stack_roi = {
        "x": hero_left["x"],
        "y": max(hero_left["y"] + hero_left["height"] + 10, 0),
        "width": hero_left["width"] + hero_right["width"] + 30,
        "height": 30,
    }
    dealer_button_roi = {
        "x": hero_left["x"] + hero_left["width"] // 2,
        "y": max(hero_left["y"] - 50, 0),
        "width": 40,
        "height": 40,
    }

    config["roi"] = {
        "hero_left": hero_left,
        "hero_right": hero_right,
        "stack": stack_roi,
        "dealer_button": dealer_button_roi,
    }
    config.setdefault("multi_table", {})["layouts"] = [table["layout"] for table in tables]

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print("\n" + "=" * 60)
    print("DETECTED CONFIGURATION:")
    print("=" * 60)
    print(f"Detected {len(tables)} tables with card positions")
    print(f"Updated config saved to {config_path}")
    print("You can now run: python src/main.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
