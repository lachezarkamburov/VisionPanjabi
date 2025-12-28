import logging
import os
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

import yaml

from strategy import StrategyEngine
from vision_agent import ROI, VisionAgent

CONFIG_ENV_VAR = "CONFIG_PATH"
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_TEMPLATES_DIR = "/app/templates"
DEFAULT_MATRIX_PATH = "/app/charts/strategy_matrix.json"


def load_config(config_path: Path) -> Dict[str, object]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML mapping")

def rank_from_template(card_name: Optional[str]) -> Optional[str]:
    if not card_name:
        return None
    return card_name[0].upper()

def validate_config(config: Dict[str, object], config_path: Path) -> Dict[str, object]:
    stream_cfg = config.get("stream") or {}
    stream_url = stream_cfg.get("url")
    if not isinstance(stream_url, str):
        raise ValueError("stream.url must be a string")
    parsed_url = urlparse(stream_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("stream.url must include a scheme and hostname")

    templates_dir = Path(config.get("templates_dir") or DEFAULT_TEMPLATES_DIR)
    if not templates_dir.exists():
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    if not templates_dir.is_dir():
        raise NotADirectoryError(f"Templates path is not a directory: {templates_dir}")

def build_multi_table_agent(config: dict) -> MultiTableVision:
    roi_config = config["roi"]
    base_rois = TableROISet(
        hero_left=ROI(**roi_config["hero_left"]),
        hero_right=ROI(**roi_config["hero_right"]),
        stack=ROI(**roi_config["stack"]),
        dealer_button=ROI(**roi_config["dealer_button"]),
    )
    multi_table_config = config.get("multi_table", {})
    return MultiTableVision(
        stream_url=config["stream"]["url"],
        templates_dir=Path(config.get("templates_dir", "/app/templates")),
        base_rois=base_rois,
        auto_detect=multi_table_config.get("auto_detect", True),
        max_tables=multi_table_config.get("max_tables", 6),
        manual_layouts=multi_table_config.get("layouts", []),
    )


def apply_strategy(
    strategy_engine: StrategyEngine, table_states: dict[str, dict[str, object]]
) -> dict[str, dict[str, object]]:
    for table_name, state in table_states.items():
        cards = state.get("cards", [])
        rank_left = rank_from_template(cards[0]) if len(cards) > 0 else None
        rank_right = rank_from_template(cards[1]) if len(cards) > 1 else None
        result = strategy_engine.lookup(rank_left, rank_right)
        state["strategy"] = {
            "hand": result.hand,
            "zone": result.zone,
            "action": result.action,
        }
        table_states[table_name] = state
    return table_states


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    logging.info("Loading configuration from %s", config_path)

    config = load_config(config_path)
    strategy_engine = StrategyEngine(Path(config["strategy"]["matrix_path"]))
    multi_table_agent = build_multi_table_agent(config)

    table_states = multi_table_agent.read_all_tables()
    evaluated_tables = apply_strategy(strategy_engine, table_states)

    output = multi_table_agent.to_json(evaluated_tables)
    logging.info("Evaluation result: %s", output)
    print(output)


if __name__ == "__main__":
    main()
