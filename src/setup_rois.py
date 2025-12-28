"""Interactive ROI setup tool."""

from pathlib import Path

import cv2
import yaml


def add_instruction_overlay(frame):
    """Add helpful instruction overlay to frame."""
    overlay = frame.copy()

    # Semi-transparent background box
    cv2.rectangle(overlay, (10, 10), (650, 140), (0, 0, 0), -1)
    frame_with_overlay = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

    # Add instruction text
    instructions = [
        "IMPORTANT: Click ONLY on the rank symbol!",
        "1. Find hero cards (bottom-center area)",
        "2. Click TOP-LEFT of rank letter (K, Q, T, etc.)",
        "3. Click BOTTOM-RIGHT of rank letter",
        "Target: ~40-80 pixels (NOT whole card!)",
    ]

    y = 30
    for i, text in enumerate(instructions):
        color = (0, 255, 255) if i == 0 else (255, 255, 255)
        thickness = 2 if i == 0 else 1

        cv2.putText(
            frame_with_overlay,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
        )
        y += 25

    return frame_with_overlay


def setup_rois() -> None:
    """Interactive ROI setup - click on one card to auto-configure."""

    # Load video
    config_path = Path("config.yaml")
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    video_path = config["stream"]["url"]
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()

    if not success:
        print("Failed to load video!")
        return

    print("\n" + "=" * 60)
    print("ROI SETUP - Manual Configuration")
    print("=" * 60)
    print("\nInstructions:")
    print("1. The video frame will appear")
    print("2. Find a CLEAR hero card (the two cards at bottom-center)")
    print("3. Look at the RANK SYMBOL in the card corner (the 'K', 'Q', 'T', etc.)")
    print("4. Click the TOP-LEFT corner of ONLY the rank symbol")
    print("5. Click the BOTTOM-RIGHT corner of ONLY the rank symbol")
    print("")
    print("⚠️  IMPORTANT:")
    print("   • Click ONLY on the rank letter/number, NOT the full card!")
    print("   • Example: For King of clubs 'K♣', click only on the 'K' part")
    print("   • Target ROI size: ~40-80 pixels (whole card is WRONG!)")
    print("")
    print("\nPress any key to start...")
    print("=" * 60)

    input()

    # Show frame and let user click
    clicks: list[tuple[int, int]] = []

    display_frame = add_instruction_overlay(frame.copy())

    def mouse_callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

            # Draw click marker
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(display_frame, (x, y), 8, (0, 255, 0), 2)

            if len(clicks) == 1:
                cv2.putText(
                    display_frame,
                    "Now click BOTTOM-RIGHT of rank symbol only!",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            elif len(clicks) == 2:
                x1, y1 = clicks[0]
                x2, y2 = clicks[1]

                # Draw ROI rectangle
                cv2.rectangle(
                    display_frame,
                    (min(x1, x2), min(y1, y2)),
                    (max(x1, x2), max(y1, y2)),
                    (0, 255, 0),
                    2,
                )

                width = abs(x2 - x1)
                height = abs(y2 - y1)

                if width > 200 or height > 200:
                    cv2.putText(
                        display_frame,
                        f"TOO LARGE! {width}x{height}px - You selected the whole card!",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        "Click ONLY on rank symbol! Press 'r' to retry",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                elif width < 30 or height < 30:
                    cv2.putText(
                        display_frame,
                        f"TOO SMALL! {width}x{height}px - Make larger selection",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        "Press 'r' to retry",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        display_frame,
                        f"GOOD! {width}x{height}px - Press ENTER to confirm",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Setup ROIs", display_frame)

    cv2.namedWindow("Setup ROIs")
    cv2.setMouseCallback("Setup ROIs", mouse_callback)
    cv2.imshow("Setup ROIs", display_frame)

    print("\nClick on ONE playing card's rank symbol top-left corner, then bottom-right corner...")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicks.clear()
            display_frame = add_instruction_overlay(frame.copy())
            cv2.imshow("Setup ROIs", display_frame)
            print("Reset! Click again...")

        elif key == ord("q"):
            print("Setup cancelled by user")
            cv2.destroyAllWindows()
            return

        elif key == 13:  # ENTER key
            if len(clicks) == 2:
                break

        if len(clicks) < 2:
            continue

    cv2.waitKey(2000)  # Show result for 2 seconds
    cv2.destroyAllWindows()

    # Calculate ROI from clicks
    x1, y1 = clicks[0]
    x2, y2 = clicks[1]

    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    print(f"\n📏 ROI Size: {width}×{height} pixels")

    MIN_RANK_SIZE = 30
    MAX_RANK_SIZE = 200
    MAX_ASPECT_RATIO = 2.5

    if width < MIN_RANK_SIZE or height < MIN_RANK_SIZE:
        print("❌ ERROR: ROI too small!")
        print(f"   Minimum size: {MIN_RANK_SIZE}×{MIN_RANK_SIZE} pixels")
        print("   The rank symbol needs to be clearly visible")
        print("\n⚠️  Please run the script again and click more carefully")
        return

    if width > MAX_RANK_SIZE or height > MAX_RANK_SIZE:
        print("❌ ERROR: ROI too large - you selected the whole card!")
        print(f"   Your selection: {width}×{height} pixels")
        print(f"   Maximum size: {MAX_RANK_SIZE}×{MAX_RANK_SIZE} pixels")
        print("\n⚠️  You should click ONLY on the rank symbol (K, Q, T, etc.)")
        print("   NOT the entire card!")
        print("   Please run the script again and select ONLY the rank letter")
        return

    aspect_ratio = max(width, height) / min(width, height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        print(f"⚠️  WARNING: Unusual ROI shape (aspect ratio: {aspect_ratio:.2f})")
        print(f"   Size: {width}×{height} pixels")
        print("   Rank symbols are usually roughly square")
        print("   Continue anyway? (y/n): ", end="")

        response = input().strip().lower()
        if response != "y":
            print("Setup cancelled. Please run the script again.")
            return

    print(f"✅ ROI size validated: {width}×{height} pixels")
    print("   This is appropriate for a rank symbol!\n")
    print(f"\n✅ Card detected at: x={x}, y={y}, width={width}, height={height}")

    # Determine which table the click is in (2x2 grid)
    frame_h, frame_w = frame.shape[:2]
    table_w = frame_w // 2
    table_h = frame_h // 2

    table_col = 0 if x < table_w else 1
    table_row = 0 if y < table_h else 1

    table_x = table_col * table_w
    table_y = table_row * table_h

    rel_x = x - table_x
    rel_y = y - table_y

    print(
        "\n✅ Detected click in table: "
        f"row {table_row}, col {table_col} (origin x={table_x}, y={table_y})"
    )
    print(f"   Relative position: x={rel_x}, y={rel_y}")

    # Assume cards are side-by-side with small gap
    gap = 15  # Typical gap between hero cards

    # Update config with RELATIVE coordinates
    config["roi"] = {
        "hero_left": {
            "x": rel_x,
            "y": rel_y,
            "width": width,
            "height": height,
        },
        "hero_right": {
            "x": rel_x + width + gap,
            "y": rel_y,
            "width": width,
            "height": height,
        },
        "stack": {
            "x": rel_x,
            "y": rel_y + height + 10,
            "width": width * 2 + gap,
            "height": 30,
        },
        "dealer_button": {
            "x": rel_x + width // 2,
            "y": rel_y - 50,
            "width": 40,
            "height": 40,
        },
    }

    # Update multi-table layouts to a 2x2 grid based on frame size
    config.setdefault("multi_table", {})["layouts"] = [
        {"x": 0, "y": 0, "width": table_w, "height": table_h},
        {"x": table_w, "y": 0, "width": table_w, "height": table_h},
        {"x": 0, "y": table_h, "width": table_w, "height": table_h},
        {"x": table_w, "y": table_h, "width": table_w, "height": table_h},
    ]

    # Save config
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    print(f"\n✅ ROI configuration saved to {config_path}")
    print("\nNow run: python src/main.py --extract-templates")
    print("This will extract fresh templates using the correct coordinates.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_rois()
