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
import logging
import os
from pathlib import Path

from strategy import StrategyEngine
from vision_agent import ROI, VisionAgent


def build_agent() -> VisionAgent:
    return VisionAgent(
        stream_url=os.getenv("STREAM_URL", "https://twitch.tv/ggpoker"),
        templates_dir=Path("/app/templates"),
        roi_hero_left=ROI(x=980, y=880, width=80, height=80),
        roi_hero_right=ROI(x=1070, y=880, width=80, height=80),
        roi_stack=ROI(x=990, y=960, width=140, height=40),
        roi_button=ROI(x=1140, y=760, width=40, height=40),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Project started")
    agent = build_agent()
    strategy_engine = StrategyEngine(Path("/app/charts/strategy_matrix.json"))

    game_state = agent.read_game_state()
    result = strategy_engine.lookup(game_state.hero_left, game_state.hero_right)

    logging.info(
        "Hand %s in zone %s -> %s", result.hand, result.zone, result.action
    )


if __name__ == "__main__":
    main()
