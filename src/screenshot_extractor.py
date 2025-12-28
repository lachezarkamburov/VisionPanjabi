"""
Screenshot-based template extraction utilities.

This module provides a semi-automated flow for building playing card templates from
static screenshots. It performs light-weight contour detection to locate card regions,
extracts the rank/suit corner, and attempts to auto-identify cards using existing
templates. Any cards that cannot be confidently identified fall back to an interactive
labeling prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


ImageDict = Dict[str, np.ndarray]


def _iter_image_files(screenshots_dir: Path) -> List[Path]:
    """Return a sorted list of supported image files in a directory."""
    supported = {".png", ".jpg", ".jpeg"}
    return sorted(
        path
        for path in screenshots_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


def _load_reference_templates(templates_dir: Path) -> ImageDict:
    """Load any existing templates to use for pattern matching."""
    templates: ImageDict = {}
    if not templates_dir.exists():
        return templates

    for template_path in templates_dir.glob("*.png"):
        image = cv2.imread(str(template_path))
        if image is not None:
            templates[template_path.stem] = image
    return templates


def extract_from_screenshots(screenshots_dir: Path, config: dict) -> None:
    """
    Extract card templates from a folder of screenshot images.

    Args:
        screenshots_dir: Path to folder containing screenshot images.
        config: Application configuration (used for output directory overrides).

    Example:
        python src/main.py --extract-templates screenshots/
    """
    templates_dir = Path(config.get("templates_dir", "templates"))
    templates_dir.mkdir(parents=True, exist_ok=True)

    image_files = _iter_image_files(screenshots_dir)
    if not image_files:
        print(f"❌ No images found in {screenshots_dir}")
        return

    print(f"📸 Found {len(image_files)} screenshot(s)")
    print("=" * 70)

    reference_templates = _load_reference_templates(templates_dir)
    all_cards: ImageDict = {}
    unidentified: List[np.ndarray] = []

    for index, img_path in enumerate(image_files, start=1):
        print(f"\n📷 Processing: {img_path.name} ({index}/{len(image_files)})")
        image = cv2.imread(str(img_path))
        if image is None:
            print("   ⚠️  Could not load image, skipping")
            continue

        identified, unknown = detect_and_extract_cards(image, reference_templates)
        print(f"   Found {len(identified) + len(unknown)} cards")

        for card_name, template in identified.items():
            if card_name not in all_cards and card_name not in reference_templates:
                all_cards[card_name] = template
                print(f"   ✅ {card_name}")

        unidentified.extend(unknown)

    if unidentified:
        print("\n🎛️  Interactive labeling required for remaining cards...")
        labeled = interactive_label_cards(unidentified)
        for card_name, template in labeled.items():
            if card_name not in all_cards and card_name not in reference_templates:
                all_cards[card_name] = template

    save_templates(all_cards, templates_dir)

    print("\n" + "=" * 70)
    print("✅ Extraction Complete!")
    print(f"   Screenshots processed: {len(image_files)}")
    print(f"   Unique cards found: {len(all_cards)}")
    print(f"   Templates saved to: {templates_dir.resolve()}")
    print("=" * 70)
    print("\nNext step: Run validation")
    print("  python src/validate_templates.py")
    print("=" * 70)


def detect_and_extract_cards(
    image: np.ndarray, reference_templates: ImageDict | None = None
) -> Tuple[ImageDict, List[np.ndarray]]:
    """
    Detect cards in an image and extract their corner templates.

    Args:
        image: Screenshot image (BGR format).
        reference_templates: Existing templates to match against for auto-labeling.

    Returns:
        Tuple of (identified_cards, unidentified_corners) where:
            identified_cards: {card_name: corner_template_image}
            unidentified_corners: list of corner images needing manual labeling
    """
    reference_templates = reference_templates or {}
    cards: ImageDict = {}
    unknown: List[np.ndarray] = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / h if h else 0
        if aspect < 0.5 or aspect > 0.9:
            continue

        corner = _extract_corner(image, x, y, w, h)
        if corner is None:
            continue

        card_name = identify_card(corner, reference_templates)
        if card_name:
            cards.setdefault(card_name, corner)
        else:
            unknown.append(corner)

    return cards, unknown


def _extract_corner(
    image: np.ndarray, x: int, y: int, w: int, h: int
) -> np.ndarray | None:
    """Extract the top-left corner of a detected card."""
    corner_h = min(int(h * 0.30), 120)
    corner_w = min(int(w * 0.35), 100)
    if corner_h < 40 or corner_w < 35:
        return None

    y_end = min(y + corner_h, image.shape[0])
    x_end = min(x + corner_w, image.shape[1])
    return image[max(y, 0) : y_end, max(x, 0) : x_end]


def identify_card(corner: np.ndarray, reference_templates: ImageDict) -> str | None:
    """
    Identify card rank and suit from a corner image using template matching.

    Args:
        corner: Card corner image showing rank and suit.
        reference_templates: Already-labeled templates to match against.

    Returns:
        Card name string (e.g., "9s") or None if confidence is low.
    """
    if not reference_templates:
        return None

    target_is_red = is_red_card(corner)
    corner_gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)

    best_match = None
    best_score = -1.0
    for name, template in reference_templates.items():
        template_is_red = is_red_card(template)
        if template_is_red != target_is_red:
            continue

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        resized_corner = cv2.resize(
            corner_gray, (template_gray.shape[1], template_gray.shape[0]), interpolation=cv2.INTER_AREA
        )
        result = cv2.matchTemplate(resized_corner, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_match = name

    if best_match and best_score >= 0.8:
        return best_match

    return None


def is_red_card(corner: np.ndarray) -> bool:
    """Detect if a card corner is predominantly red (hearts/diamonds)."""
    hsv = cv2.cvtColor(corner, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = corner.shape[0] * corner.shape[1]
    return (red_pixels / max(total_pixels, 1)) > 0.05


def interactive_label_cards(cards: Iterable[np.ndarray]) -> ImageDict:
    """
    Show unidentified cards to the user for manual labeling.

    Args:
        cards: Iterable of card corner images.

    Returns:
        Dictionary mapping card_name -> image for manually labeled templates.
    """
    labeled: ImageDict = {}
    cards = list(cards)
    display_available = True

    for index, corner in enumerate(cards, start=1):
        if display_available:
            try:
                display = cv2.resize(corner, (150, 225))
                cv2.imshow("Card Template", display)
                cv2.waitKey(100)
            except cv2.error:
                display_available = False
                cv2.destroyAllWindows()

        color_hint = "RED (♥♦)" if is_red_card(corner) else "BLACK (♣♠)"
        print(f"\nCard {index}/{len(cards)}")
        print(f"Color: {color_hint}")

        while True:
            label = input("Enter card (e.g., 9s, Qc, Kh) or 'skip': ").strip()
            if label.lower() == "skip":
                break

            if len(label) >= 2:
                rank = label[0].upper()
                suit = label[1].lower()
                if rank in list("AKQJT98765432") and suit in ["c", "s", "h", "d"]:
                    labeled[f"{rank}{suit}"] = corner
                    print(f"✅ Saved: {rank}{suit}")
                    break

            print("❌ Invalid format! Use: 9s, Qc, Kh, etc.")

    if display_available:
        cv2.destroyAllWindows()

    return labeled


def save_templates(cards: ImageDict, templates_dir: Path) -> None:
    """Save card templates to the templates directory."""
    templates_dir.mkdir(parents=True, exist_ok=True)
    for card_name, template in cards.items():
        filepath = templates_dir / f"{card_name}.png"
        cv2.imwrite(str(filepath), template)


__all__ = [
    "detect_and_extract_cards",
    "extract_from_screenshots",
    "interactive_label_cards",
    "is_red_card",
    "save_templates",
]
