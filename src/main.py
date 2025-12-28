import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from strategy import StrategyEngine
from vision_agent import ROI, VisionAgent


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "stacklevel",
        }
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


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
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    logger.info("Project started", extra={"event": "app_start"})
    agent = build_agent()
    strategy_engine = StrategyEngine(Path("/app/charts/strategy_matrix.json"))

    start_time = time.perf_counter()
    game_state = agent.read_game_state()
    result = strategy_engine.lookup(game_state.hero_left, game_state.hero_right)
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "Hand decision computed",
        extra={
            "event": "strategy_lookup",
            "hand": result.hand,
            "zone": result.zone,
            "action": result.action,
            "duration_ms": round(duration_ms, 2),
            "hero_left": game_state.hero_left,
            "hero_right": game_state.hero_right,
        },
    )


if __name__ == "__main__":
    main()
