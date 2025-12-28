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

    return validate_config(config, config_path)


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

    roi_cfg = config.get("roi")
    if not isinstance(roi_cfg, dict):
        raise ValueError("roi must be a mapping of ROI definitions")

    rois: Dict[str, ROI] = {}
    for name in ("hero_left", "hero_right", "stack", "dealer_button"):
        roi_values = roi_cfg.get(name)
        if not isinstance(roi_values, dict):
            raise ValueError(f"roi.{name} must be a mapping")
        try:
            roi = ROI(
                x=int(roi_values["x"]),
                y=int(roi_values["y"]),
                width=int(roi_values["width"]),
                height=int(roi_values["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"roi.{name} must include integer x, y, width, height") from exc

        if roi.width <= 0 or roi.height <= 0:
            raise ValueError(f"roi.{name} width and height must be positive")
        if roi.x < 0 or roi.y < 0:
            raise ValueError(f"roi.{name} x and y must be non-negative")
        rois[name] = roi

    strategy_cfg = config.get("strategy") or {}
    matrix_path = Path(strategy_cfg.get("matrix_path") or DEFAULT_MATRIX_PATH)
    if not matrix_path.exists():
        raise FileNotFoundError(f"Strategy matrix file not found: {matrix_path}")

    return {
        "stream_url": stream_url,
        "templates_dir": templates_dir,
        "roi": rois,
        "matrix_path": matrix_path,
        "config_path": config_path,
    }


def build_agent(config: Dict[str, object]) -> VisionAgent:
    roi_cfg: Dict[str, ROI] = config["roi"]  # type: ignore[assignment]
    return VisionAgent(
        stream_url=config["stream_url"],  # type: ignore[arg-type]
        templates_dir=config["templates_dir"],  # type: ignore[arg-type]
        roi_hero_left=roi_cfg["hero_left"],
        roi_hero_right=roi_cfg["hero_right"],
        roi_stack=roi_cfg["stack"],
        roi_button=roi_cfg["dealer_button"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Project started")

    config_path = Path(os.getenv(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH))
    config = load_config(config_path)

    agent = build_agent(config)
    strategy_engine = StrategyEngine(config["matrix_path"])  # type: ignore[arg-type]

    game_state = agent.read_game_state()
    result = strategy_engine.lookup(game_state.hero_left, game_state.hero_right)

    logging.info("Hand %s in zone %s -> %s", result.hand, result.zone, result.action)


if __name__ == "__main__":
    main()
