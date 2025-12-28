import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml

from card_detector import AutoCardDetector
from card_recognizer import CardRecognizer
from multi_table import MultiTableVision, TableROISet
from vision_agent import ROI


logger = logging.getLogger(__name__)


def _crop(frame: np.ndarray, roi: ROI) -> np.ndarray:
    roi.validate_within(frame.shape)
    return frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


def _build_preview(templates: List[np.ndarray]) -> np.ndarray:
    """Create a simple horizontal preview image of extracted templates."""
    if not templates:
        raise ValueError("No templates provided for preview image")

    max_height = max(img.shape[0] for img in templates)
    padded: List[np.ndarray] = []
    for img in templates:
        pad_bottom = max_height - img.shape[0]
        padded.append(
            cv2.copyMakeBorder(img, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        )
    return np.concatenate(padded, axis=1)


def _roi_to_dict(roi: ROI) -> dict:
    return {"x": roi.x, "y": roi.y, "width": roi.width, "height": roi.height}


def _estimate_additional_rois(hero_left: ROI, hero_right: ROI) -> tuple[ROI, ROI]:
    leftmost_x = min(hero_left.x, hero_right.x)
    rightmost_x = max(hero_left.x + hero_left.width, hero_right.x + hero_right.width)
    combined_width = (rightmost_x - leftmost_x) + 30

    stack_roi = ROI(
        x=leftmost_x,
        y=max(hero_left.y + hero_left.height, hero_right.y + hero_right.height) + 10,
        width=combined_width,
        height=30,
    )
    dealer_button_roi = ROI(
        x=hero_left.x + hero_left.width // 2,
        y=max(hero_left.y - 50, 0),
        width=40,
        height=40,
    )
    return stack_roi, dealer_button_roi


def _apply_detected_layouts(
    vision: MultiTableVision, tables: List[dict], config_path: Path
) -> None:
    hero_left_roi = ROI(**tables[0]["hero_left"])
    hero_right_roi = ROI(**tables[0]["hero_right"])
    stack_roi, dealer_button_roi = _estimate_additional_rois(hero_left_roi, hero_right_roi)

    vision.base_rois = TableROISet(
        hero_left=hero_left_roi,
        hero_right=hero_right_roi,
        stack=stack_roi,
        dealer_button=dealer_button_roi,
    )
    vision.manual_layouts = [table["layout"] for table in tables]

    config: dict = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}

    config.setdefault("roi", {})
    config["roi"]["hero_left"] = _roi_to_dict(hero_left_roi)
    config["roi"]["hero_right"] = _roi_to_dict(hero_right_roi)
    config["roi"]["stack"] = _roi_to_dict(stack_roi)
    config["roi"]["dealer_button"] = _roi_to_dict(dealer_button_roi)
    config.setdefault("multi_table", {})["layouts"] = vision.manual_layouts

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    logger.info(
        "Auto-detected ROIs updated",
        extra={
            "event": "roi_autodetect_update",
            "layouts": len(vision.manual_layouts),
        },
    )


def auto_extract_templates(
    vision: MultiTableVision,
    templates_dir: Path,
    force: bool = False,
    preview: bool = False,
) -> int:
    """
    Extract card templates from the current video frame if templates are missing.

    Args:
        vision: MultiTableVision instance with loaded video.
        templates_dir: Path to templates directory.
        force: Force re-extraction even if templates exist.
        preview: Generate a preview image of extracted templates.

    Returns:
        Number of templates written to disk.
    """
    existing_templates = list(templates_dir.glob("*.png"))
    need_templates = force or not existing_templates

    templates_dir.mkdir(parents=True, exist_ok=True)
    if not need_templates:
        logger.info("Found existing templates, skipping extraction")
        return 0

    logger.info("Templates folder empty - extracting from video...")

    detector = AutoCardDetector(vision.stream_url)
    config_path = Path("config.yaml")
    try:
        frame_for_detection = detector.load_frame()
        detected_tables = detector.detect_tables_and_cards(
            frame_for_detection, num_tables=vision.max_tables
        )
        if detected_tables:
            logger.info("Auto-detected %s tables with card positions", len(detected_tables))
            _apply_detected_layouts(vision, detected_tables, config_path)
        else:
            logger.warning("Auto-detection failed, using existing ROIs")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Automatic ROI detection failed: %s", exc)

    recognizer = CardRecognizer()
    frame = vision.capture_frame()
    layouts = vision.get_table_layouts(frame)
    if not layouts:
        logger.warning("No table layouts detected; cannot extract templates")
        return 0

    template_images: List[np.ndarray] = []
    template_count = 0
    unrecognized_count = 0
    recognized_count = 0

    for table_index, layout in enumerate(layouts, start=1):
        left_roi = vision.get_table_roi(layout, vision.base_rois.hero_left)
        right_roi = vision.get_table_roi(layout, vision.base_rois.hero_right)

        left_card = _crop(frame, left_roi)
        right_card = _crop(frame, right_roi)

        left_name = recognizer.recognize_card(left_card)
        right_name = recognizer.recognize_card(right_card)

        if left_name and len(left_name) == 2 and left_name[0] in "AKQJT98765432" and left_name[1] in "hdcs":
            left_filename = f"{left_name}.png"
            recognized_count += 1
            logger.info("✅ Recognized: %s (table %s left)", left_name, table_index)
        else:
            left_filename = f"card_table_{table_index}_left.png"
            unrecognized_count += 1
            logger.warning("❌ Could not recognize card from table %s left, using generic name", table_index)

        if right_name and len(right_name) == 2 and right_name[0] in "AKQJT98765432" and right_name[1] in "hdcs":
            right_filename = f"{right_name}.png"
            recognized_count += 1
            logger.info("✅ Recognized: %s (table %s right)", right_name, table_index)
        else:
            right_filename = f"card_table_{table_index}_right.png"
            unrecognized_count += 1
            logger.warning("❌ Could not recognize card from table %s right, using generic name", table_index)

        left_path = templates_dir / left_filename
        right_path = templates_dir / right_filename

        cv2.imwrite(str(left_path), left_card)
        cv2.imwrite(str(right_path), right_card)

        template_images.extend([left_card, right_card])
        template_count += 2

        logger.info(
            "Extracted templates for table %s: %s, %s", table_index, left_path.name, right_path.name
        )

    if preview and template_images:
        preview_path = templates_dir / "preview.png"
        cv2.imwrite(str(preview_path), _build_preview(template_images))
        logger.info("Template preview saved to %s", preview_path)

    vision.reload_templates()
    if unrecognized_count:
        logger.warning(
            "Failed to recognize %s cards. Please manually rename files in %s using format: Ah.png, Kc.png, etc.",
            unrecognized_count,
            templates_dir,
        )
    logger.info(
        "Extracted %s card templates from video (%s recognized)",
        template_count,
        recognized_count,
    )
    return template_count
