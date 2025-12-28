"""Automated card template extraction utilities.

This module scans a poker video for card corners, clusters similar cards,
and provides an interactive labeling flow to create high-quality templates
quickly. Run it directly:

    python src/smart_extract.py "video/poker-game.mp4"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CardSample(TypedDict):
    """Represents a detected card corner sample."""

    corner: np.ndarray
    is_red: bool


RANKS: List[str] = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"]
SUITS: List[str] = ["c", "s", "h", "d"]


def is_red_card(corner_img: np.ndarray) -> bool:
    """Determine if a card corner is red based on pixel analysis.

    Args:
        corner_img: BGR image of the card corner.

    Returns:
        True if more than 5% of the pixels fall within a red HSV range, False otherwise.

    Examples:
        >>> import numpy as np
        >>> sample = np.zeros((10, 10, 3), dtype=np.uint8)
        >>> is_red_card(sample)
        False
    """

    hsv = cv2.cvtColor(corner_img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = max(corner_img.shape[0] * corner_img.shape[1], 1)

    return (red_pixels / total_pixels) > 0.05


def scan_video_for_cards(video_path: str | Path, max_frames: int = 200) -> List[CardSample]:
    """Scan a poker video and extract unique card corners.

    Args:
        video_path: Path to the video file to analyze.
        max_frames: Maximum number of frames to sample across the video.

    Returns:
        A list of detected card samples with corner images and color metadata.

    Raises:
        FileNotFoundError: If the provided video path does not exist.
        RuntimeError: If the video cannot be opened.

    Examples:
        >>> scan_video_for_cards("video/poker-game.mp4", max_frames=50)
        [{'corner': array(...), 'is_red': True}, ...]
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        logger.warning("Video reports zero frames: %s", video_path)

    interval = max(1, total_frames // max_frames) if total_frames > 0 else 1

    all_cards: List[CardSample] = []

    for frame_num in range(0, total_frames or max_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 2000:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / h if h > 0 else 0

            if aspect < 0.5 or aspect > 0.9:
                continue

            card = frame[y : y + h, x : x + w]

            corner_h = min(int(h * 0.30), 120)
            corner_w = min(int(w * 0.35), 100)

            if corner_h < 50 or corner_w < 40:
                continue

            corner = card[5:corner_h, 5:corner_w]
            is_red = is_red_card(corner)

            all_cards.append({"corner": corner, "is_red": is_red})

    cap.release()
    logger.info("Collected %s card corner samples", len(all_cards))
    return all_cards


def _laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def cluster_unique_cards(cards: List[CardSample], threshold: float = 0.15) -> List[CardSample]:
    """Cluster similar cards and return unique templates.

    Args:
        cards: List of card samples with corner images.
        threshold: Normalized similarity threshold (0-1) based on mean absolute difference.

    Returns:
        A reduced list of representative card samples.
    """

    if not cards:
        return []

    standard_size = (60, 90)
    normalized: List[Dict[str, object]] = []

    for card in cards:
        resized = cv2.resize(card["corner"], standard_size)
        normalized.append({"normalized": resized, "original": card["corner"], "is_red": card["is_red"]})

    unique_cards: List[CardSample] = []
    used = [False] * len(normalized)

    for i, card1 in enumerate(normalized):
        if used[i]:
            continue

        cluster: List[Dict[str, object]] = [card1]
        used[i] = True

        for j, card2 in enumerate(normalized):
            if used[j] or i == j:
                continue

            diff = cv2.absdiff(card1["normalized"], card2["normalized"])
            mse = float(np.mean(diff))

            if mse < threshold * 255:
                cluster.append(card2)
                used[j] = True

        best = max(cluster, key=lambda c: _laplacian_variance(c["original"]))
        unique_cards.append({"corner": best["original"], "is_red": bool(best["is_red"])})

    logger.info("Clustered %s samples into %s unique cards", len(cards), len(unique_cards))
    return unique_cards


def interactive_label_and_save(unique_cards: List[CardSample]) -> Dict[str, bool]:
    """Interactive labeling interface for card templates.

    Args:
        unique_cards: List of unique card samples for labeling.

    Returns:
        Mapping of saved card names to True for bookkeeping.
    """

    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("🎯 INTERACTIVE CARD LABELING")
    print("=" * 70)
    print("\nInstructions:")
    print("  • Type rank and suit: Kc, As, 9h, Td")
    print("  • Type 'skip' to skip this card")
    print("  • Type 'done' when finished")
    print("  • Color is auto-detected to help you\n")

    saved_cards: Dict[str, bool] = {}
    saved_count = 0

    for i, card_data in enumerate(unique_cards):
        img = card_data["corner"]
        is_red = card_data["is_red"]

        print(f"\n--- Card {i + 1}/{len(unique_cards)} ---")
        color_hint = "RED (♥ hearts / ♦ diamonds)" if is_red else "BLACK (♣ clubs / ♠ spades)"
        print(f"   Color: {color_hint}")

        display = cv2.resize(img, (150, 225))
        try:
            cv2.imshow("Card Template", display)
            cv2.waitKey(100)
        except cv2.error:
            logger.debug("cv2.imshow unavailable; skipping preview window")

        while True:
            label = input("   Label (e.g., Kc, As, 9h) or 'skip'/'done': ").strip()

            if label.lower() == "done":
                cv2.destroyAllWindows()
                print(f"\n✅ Saved {saved_count} templates!")
                return saved_cards

            if label.lower() == "skip" or not label:
                break

            if len(label) < 2:
                print("   ❌ Invalid! Use: Kc, As, 9h, Td")
                continue

            rank = label[0].upper()
            suit = label[1].lower()

            if rank not in RANKS:
                print(f"   ❌ Invalid rank! Use: {', '.join(RANKS)}")
                continue

            if suit not in SUITS:
                print("   ❌ Invalid suit! Use: c, s, h, d")
                continue

            expected_red = suit in ["h", "d"]
            if expected_red != is_red:
                print(
                    f"   ⚠️  WARNING: {suit} should be {'RED' if expected_red else 'BLACK'} "
                    f"but card appears {color_hint}"
                )
                confirm = input("   Continue anyway? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            card_name = f"{rank}{suit}"
            filepath = templates_dir / f"{card_name}.png"
            if filepath.exists():
                overwrite = input(f"   ⚠️  {card_name}.png exists. Overwrite? (y/n): ").strip().lower()
                if overwrite != "y":
                    continue

            cv2.imwrite(str(filepath), img)
            saved_cards[card_name] = True
            saved_count += 1
            print(f"   ✅ Saved: {filepath.name}")
            break

    cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"✅ Saved {saved_count} templates")
    print("=" * 70)

    all_expected = [f"{r}{s}" for r in RANKS for s in SUITS]
    missing = [c for c in all_expected if c not in saved_cards]

    if missing:
        print(f"\n⚠️  Missing {len(missing)} cards:")
        high_priority = [c for c in missing if c[0] in ["A", "K", "Q", "J", "T"]]
        print(f"   High priority: {', '.join(high_priority[:15])}")
    else:
        print("\n🎉 COMPLETE! All 52 templates created!")

    return saved_cards


def main() -> None:
    """Run the smart extraction pipeline from the command line."""

    if len(sys.argv) < 2:
        print("Usage: python src/smart_extract.py <video_file>")
        print("Example: python src/smart_extract.py video/poker.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    print("=" * 70)
    print("🎴 SMART CARD TEMPLATE EXTRACTOR")
    print("=" * 70)

    try:
        print("\n📹 Step 1: Scanning video for cards...")
        all_cards = scan_video_for_cards(video_path, max_frames=200)
        print(f"   Found {len(all_cards)} card samples")

        if not all_cards:
            print("❌ No cards detected in video!")
            return

        print("\n🔍 Step 2: Identifying unique cards...")
        unique_cards = cluster_unique_cards(all_cards, threshold=0.15)
        print(f"   Identified {len(unique_cards)} unique cards")

        if not unique_cards:
            print("❌ No unique cards identified!")
            return

        print("\n🏷️  Step 3: Label the cards...")
        interactive_label_and_save(unique_cards)
    except FileNotFoundError as exc:
        print(f"❌ {exc}")
        sys.exit(1)
    except RuntimeError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Run: python src/validate_templates.py")
    print("  2. Create missing templates (if any)")
    print("  3. Test: python src/main.py --video 'your-video.mp4'")
    print("=" * 70)


if __name__ == "__main__":
    main()
