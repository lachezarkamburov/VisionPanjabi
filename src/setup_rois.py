"""Interactive ROI setup tool."""

from pathlib import Path

import cv2
import yaml


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
    print("2. Click on the TOP-LEFT corner of a hero card")
    print("3. Then click on the BOTTOM-RIGHT corner of the same card")
    print("4. The script will save the ROI coordinates")
    print("\nPress any key to start...")
    print("=" * 60)

    input()

    # Show frame and let user click
    clicks: list[tuple[int, int]] = []

    def mouse_callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(frame_copy, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Setup ROIs", frame_copy)

            if len(clicks) == 2:
                # Draw rectangle
                cv2.rectangle(frame_copy, clicks[0], clicks[1], (0, 255, 0), 2)
                cv2.imshow("Setup ROIs", frame_copy)

    frame_copy = frame.copy()
    cv2.namedWindow("Setup ROIs")
    cv2.setMouseCallback("Setup ROIs", mouse_callback)
    cv2.imshow("Setup ROIs", frame_copy)

    print("\nClick on ONE playing card's top-left corner, then bottom-right corner...")

    while len(clicks) < 2:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    cv2.waitKey(2000)  # Show result for 2 seconds
    cv2.destroyAllWindows()

    # Calculate ROI from clicks
    x1, y1 = clicks[0]
    x2, y2 = clicks[1]

    x = min(x1, x2)
    y = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    print(f"\n✅ Card detected at: x={x}, y={y}, width={width}, height={height}")

    # Assume cards are side-by-side with small gap
    gap = 15  # Typical gap between hero cards

    # Update config
    config["roi"] = {
        "hero_left": {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        },
        "hero_right": {
            "x": x + width + gap,
            "y": y,
            "width": width,
            "height": height,
        },
        "stack": {
            "x": x,
            "y": y + height + 10,
            "width": width * 2 + gap,
            "height": 30,
        },
        "dealer_button": {
            "x": x + width // 2,
            "y": y - 50,
            "width": 40,
            "height": 40,
        },
    }

    # Save config
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, default_flow_style=False, sort_keys=False)

    print(f"\n✅ ROI configuration saved to {config_path}")
    print("\nNow run: python src/main.py --extract-templates")
    print("This will extract fresh templates using the correct coordinates.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    setup_rois()
