import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from multi_table import MultiTableVision, TableROISet
from strategy import StrategyEngine
from vision_agent import ROI


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def rank_from_template(card_name: Optional[str]) -> Optional[str]:
    if not card_name:
        return None
    return card_name[0].upper()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config = load_config(Path("/app/config.yaml"))

    roi_config = config["roi"]
    base_rois = TableROISet(
        hero_left=ROI(**roi_config["hero_left"]),
        hero_right=ROI(**roi_config["hero_right"]),
        stack=ROI(**roi_config["stack"]),
        dealer_button=ROI(**roi_config["dealer_button"]),
    )

    multi_config = config["multi_table"]
    vision = MultiTableVision(
        stream_url=config["stream"]["url"],
        templates_dir=Path("/app/templates"),
        base_rois=base_rois,
        auto_detect=multi_config.get("auto_detect", True),
        max_tables=multi_config.get("max_tables", 6),
        manual_layouts=multi_config.get("layouts", []),
    )

    strategy_engine = StrategyEngine(Path(config["strategy"]["matrix_path"]))
    table_data = vision.read_all_tables()

    for table_id, payload in table_data.items():
        cards = payload["cards"]
        result = strategy_engine.lookup(
            rank_from_template(cards[0]), rank_from_template(cards[1])
        )
        payload["strategy"] = {
            "hand": result.hand,
            "zone": result.zone,
            "action": result.action,
        }
        logging.info(
            "%s -> Hand %s in zone %s => %s",
            table_id,
            result.hand,
            result.zone,
            result.action,
        )

    print(json.dumps(table_data, indent=2))


if __name__ == "__main__":
    main()
