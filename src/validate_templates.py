"""Validate card template images and report gaps."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

RANKS: List[str] = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS: List[str] = ["c", "s", "h", "d"]
VALID_PATTERN = re.compile(r"^[AKQJT98765432][cshd]\.png$", re.IGNORECASE)


def _expected_cards() -> List[str]:
    return [f"{rank}{suit}" for rank in RANKS for suit in SUITS]


def _load_image(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return img


def _quality_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def validate_templates(templates_dir: Path | str = "templates") -> Dict[str, Tuple[bool, str]]:
    """Validate template files for size, naming, and quality.

    Args:
        templates_dir: Directory containing template PNGs.

    Returns:
        Mapping of filename to a tuple of (is_valid, message).
    """

    templates_dir = Path(templates_dir)
    results: Dict[str, Tuple[bool, str]] = {}

    if not templates_dir.exists():
        print(f"❌ Templates directory not found: {templates_dir}")
        return results

    for path in sorted(templates_dir.glob("*.png")):
        filename = path.name
        if not VALID_PATTERN.match(filename):
            results[filename] = (False, "Invalid name (expected rank+suit.png)")
            continue

        img = _load_image(path)
        if img is None:
            results[filename] = (False, "Unable to read image")
            continue

        h, w = img.shape[:2]
        if not (30 <= w <= 200 and 30 <= h <= 200):
            results[filename] = (False, f"Size out of range: {w}x{h}px")
            continue

        score = _quality_score(img)
        results[filename] = (True, f"Size: {w}x{h}px | Quality: {score:.1f}")

    return results


def print_report(results: Dict[str, Tuple[bool, str]]) -> None:
    """Pretty-print validation results and missing cards summary."""

    if not results:
        print("⚠️  No templates found to validate.")
        return

    valid_count = 0
    for filename, (ok, message) in results.items():
        prefix = "✅" if ok else "⚠️"
        if ok:
            valid_count += 1
        print(f"{prefix} {filename} - {message}")

    existing_cards = {filename[:2].lower() for filename in results.keys() if results[filename][0]}
    expected = set(_expected_cards())
    missing = sorted(expected - existing_cards)

    if missing:
        print(f"\n❌ Missing {len(missing)} cards")
        high_priority = [c for c in missing if c[0] in ["A", "K", "Q", "J", "T"]]
        print(f"   Priority: {', '.join(high_priority[:20])}")
    else:
        print("\n🎉 COMPLETE! All 52 templates present.")

    print(f"\nSummary: {valid_count}/{len(expected)} valid templates")


def main() -> None:
    """Run template validation from the command line."""

    results = validate_templates()
    print_report(results)


if __name__ == "__main__":
    main()
