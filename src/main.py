import json
import logging
from pathlib import Path

import yaml

from multi_table import MultiTableVision, TableROISet
from strategy import StrategyEngine
from vision_agent import ROI


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


def main() -> None:
    """Main entry point for vision engine."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

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

    # Check video file exists
    video_path = Path(config["stream"]["url"])
    if not video_path.exists():
        logger.error("Video file not found: %s", video_path)
        print(f"\n❌ ERROR: Video file not found: {video_path}")
        print(f"Please place your video file at: {video_path}")
        return

    print(f"📹 Video file: {video_path}")

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
        templates_dir=Path("templates"),
        base_rois=base_rois,
        auto_detect=config["multi_table"]["auto_detect"],
        max_tables=config["multi_table"]["max_tables"],
        manual_layouts=config["multi_table"].get("layouts", []),
    )

    # Initialize strategy engine
    strategy = StrategyEngine(Path(config["strategy"]["matrix_path"]))
    logger.info("Strategy engine initialized")

    # Read all tables
    print("\n🔍 Analyzing poker tables...\n")
    tables_data = vision.read_all_tables()

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


if __name__ == "__main__":
    main()
