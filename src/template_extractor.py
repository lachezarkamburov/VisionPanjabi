import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

from multi_table import MultiTableVision
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
    if existing_templates and not force:
        logger.info("Found existing templates, skipping extraction")
        return 0

    templates_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Templates folder empty - extracting from video...")

    frame = vision.capture_frame()
    layouts = vision.get_table_layouts(frame)
    if not layouts:
        logger.warning("No table layouts detected; cannot extract templates")
        return 0

    template_images: List[np.ndarray] = []
    template_count = 0

    for table_index, layout in enumerate(layouts, start=1):
        left_roi = vision.get_table_roi(layout, vision.base_rois.hero_left)
        right_roi = vision.get_table_roi(layout, vision.base_rois.hero_right)

        left_card = _crop(frame, left_roi)
        right_card = _crop(frame, right_roi)

        left_path = templates_dir / f"card_table_{table_index}_left.png"
        right_path = templates_dir / f"card_table_{table_index}_right.png"

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
    logger.info("Extracted %s card templates from video", template_count)
    return template_count
