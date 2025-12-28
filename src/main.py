import argparse
import json
import logging
import time
from pathlib import Path

import yaml

from card_detector import AutoCardDetector
from multi_table import MultiTableVision, TableROISet
from strategy import StrategyEngine
from template_extractor import auto_extract_templates
from vision_agent import ROI
from screenshot_extractor import extract_from_screenshots


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def print_banner() -> None:
    """Print a nice banner."""
    banner = """
╔════════════════════════════════════════════════════╗
║         VisionPanjabi - Poker Vision Engine        ║
║              Local Video Demo Mode                 ║
╚════════════════════════════════════════════════════╝
"""
    print(banner)


def print_table_result(table_id: str, table_data: dict, strategy_result=None) -> None:
    """Print formatted table results."""
    print(f"\n{'='*60}")
    print(f"📊 {table_id.upper()}")
    print(f"{'='*60}")

    cards = table_data.get("cards", [None, None])
    print(f"🎴 Hero Cards: {cards[0] or '?'} | {cards[1] or '?'}")
    print(f"💰 Stack Size: {table_data.get('stack_size', 'Unknown')}")
    print(f"🔘 Dealer Button: {'Yes ✓' if table_data.get('dealer_button') else 'No ✗'}")

    if strategy_result:
        print(f"\n🎯 STRATEGY RECOMMENDATION:")
        print(f"   Hand: {strategy_result.hand}")
        print(f"   Zone: {strategy_result.zone}")
        print(f"   Action: {strategy_result.action}")
    else:
        print(f"\n⚠️  No strategy available (cards not detected)")

    print(f"{'='*60}\n")


def analyze_tables(vision: MultiTableVision, strategy: StrategyEngine) -> None:
    """Run a single analysis pass and print results."""
    logger = logging.getLogger(__name__)
    print("\n🔍 Analyzing poker tables...\n")
    tables_data = vision.read_all_tables()
    if not tables_data:
        logger.warning("No tables detected in current frame")
        return

    # Build complete output with strategy
    output = {"tables": {}}

    for table_id, table_info in tables_data.items():
        cards = table_info["cards"]
        table_output = {
            "cards": cards,
            "stack_size": table_info["stack_size"],
            "dealer_button": table_info["dealer_button"],
        }

        strategy_result = None
        if cards[0] and cards[1]:
            rank_left = cards[0][0].upper()
            rank_right = cards[1][0].upper()
            strategy_result = strategy.lookup(rank_left, rank_right)

            table_output["strategy"] = {
                "hand": strategy_result.hand,
                "zone": strategy_result.zone,
                "action": strategy_result.action,
            }

        output["tables"][table_id] = table_output

        # Print formatted output for each table
        print_table_result(table_id, table_output, strategy_result)

    # Print JSON output
    print("\n" + "=" * 60)
    print("📄 JSON OUTPUT:")
    print("=" * 60)
    print(json.dumps(output, indent=2))
    print("=" * 60 + "\n")

    logger.info("VisionPanjabi engine completed successfully")
    print("✅ Analysis complete!\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VisionPanjabi - Poker Vision Engine")
    parser.add_argument(
        "--extract-templates",
        type=str,
        metavar="PATH",
        help="Extract templates from screenshots folder or video file",
    )
    parser.add_argument(
        "--setup-rois",
        action="store_true",
        help="Interactive ROI setup tool to manually select card positions.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Seconds between repeated analyses. Use 0 for a single run.",
    )
    parser.add_argument(
        "--preview-templates",
        action="store_true",
        help="Save a preview image when extracting templates.",
    )
    parser.add_argument(
        "--auto-detect-roi",
        action="store_true",
        help="Run automatic ROI detection before analysis to update config.yaml.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug visualization images during detection.",
    )
    return parser


def _estimate_additional_rois(hero_left: dict, hero_right: dict) -> tuple[dict, dict]:
    leftmost_x = min(hero_left["x"], hero_right["x"])
    rightmost_x = max(hero_left["x"] + hero_left["width"], hero_right["x"] + hero_right["width"])
    combined_width = (rightmost_x - leftmost_x) + 30

    stack_roi = {
        "x": leftmost_x,
        "y": max(hero_left["y"] + hero_left["height"], hero_right["y"] + hero_right["height"]) + 10,
        "width": combined_width,
        "height": 30,
    }
    dealer_button_roi = {
        "x": hero_left["x"] + hero_left["width"] // 2,
        "y": max(hero_left["y"] - 50, 0),
        "width": 40,
        "height": 40,
    }
    return stack_roi, dealer_button_roi


def _run_auto_roi_detection(
    config_path: Path, config: dict, video_path: Path, max_tables: int, debug: bool = False
) -> dict | None:
    logger = logging.getLogger(__name__)
    detector = AutoCardDetector(str(video_path))
    try:
        frame = detector.load_frame()
        tables = detector.detect_tables_and_cards(frame, num_tables=max_tables, debug=debug)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Automatic ROI detection failed: %s", exc)
        return None

    if not tables:
        logger.warning("Automatic ROI detection found no tables/cards")
        return None

    hero_left = tables[0]["hero_left"]
    hero_right = tables[0]["hero_right"]
    stack_roi, dealer_button_roi = _estimate_additional_rois(hero_left, hero_right)

    config = dict(config)
    config.setdefault("roi", {})
    config["roi"]["hero_left"] = hero_left
    config["roi"]["hero_right"] = hero_right
    config["roi"]["stack"] = stack_roi
    config["roi"]["dealer_button"] = dealer_button_roi
    config.setdefault("multi_table", {})["layouts"] = [table["layout"] for table in tables]

    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    logger.info("Automatic ROI detection updated config.yaml with %s tables", len(tables))
    return config


def main() -> None:
    """Main entry point for vision engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    args = build_arg_parser().parse_args()

    print_banner()
    logger.info("VisionPanjabi engine starting...")

    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("config.yaml not found!")
        print("\n❌ ERROR: config.yaml not found in current directory")
        print("Please create config.yaml with your settings.")
        return

    config = load_config(config_path)
    logger.info("Configuration loaded successfully")

    templates_dir = Path("templates")
    extract_path = Path(args.extract_templates) if args.extract_templates else None

    if args.setup_rois:
        from setup_rois import setup_rois

        setup_rois()
        return

    if extract_path:
        if extract_path.is_dir():
            print("📸 Extracting templates from screenshots...")
            extract_from_screenshots(extract_path, config)
            return
        if not extract_path.exists():
            logger.error("Path not found: %s", extract_path)
            print(f"\n❌ Path not found: {extract_path}")
            return

    # Check video file exists
    video_path = Path(config["stream"]["url"])
    force_template_extraction = False
    if extract_path and extract_path.is_file():
        video_path = extract_path
        force_template_extraction = True

    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print(f"Please place your video file at: {video_path}")
        return

    print(f"📹 Video file: {video_path}")

    if force_template_extraction:
        logger.info("Forcing template extraction...")
        if templates_dir.exists():
            import shutil

            shutil.rmtree(templates_dir)
        templates_dir.mkdir()

    if args.auto_detect_roi:
        detected_config = _run_auto_roi_detection(
            config_path,
            config,
            video_path,
            config.get("multi_table", {}).get("max_tables", 4),
            debug=args.debug,
        )
        if detected_config:
            config = detected_config
            logger.info("Using auto-detected ROIs from config.yaml")
        else:
            logger.warning("Proceeding with existing ROIs due to auto-detection failure")

    # Build ROI set from config
    roi_config = config["roi"]
    base_rois = TableROISet(
        hero_left=ROI(**roi_config["hero_left"]),
        hero_right=ROI(**roi_config["hero_right"]),
        stack=ROI(**roi_config["stack"]),
        dealer_button=ROI(**roi_config["dealer_button"]),
    )

    # Initialize multi-table vision
    logger.info("Initializing vision system...")
    vision = MultiTableVision(
        stream_url=str(video_path),
        templates_dir=templates_dir,
        base_rois=base_rois,
        auto_detect=config["multi_table"]["auto_detect"],
        max_tables=config["multi_table"]["max_tables"],
        manual_layouts=config["multi_table"].get("layouts", []),
    )

    # Auto extract templates if needed
    auto_extract_templates(
        vision,
        templates_dir,
        force=force_template_extraction,
        preview=args.preview_templates,
    )

    # Initialize strategy engine
    strategy = StrategyEngine(Path(config["strategy"]["matrix_path"]))
    logger.info("Strategy engine initialized")

    interval = max(args.interval, 0.0)
    if interval == 0:
        analyze_tables(vision, strategy)
        return

    logger.info("Running continuous analysis every %.2f seconds", interval)
    try:
        while True:
            analyze_tables(vision, strategy)
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("Continuous analysis stopped by user")
        print("\n🛑 Stopped continuous analysis.\n")


if __name__ == "__main__":
    main()
