import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StrategyResult:
    hand: str
    zone: str
    action: str


class StrategyEngine:
    def __init__(self, matrix_path: Path) -> None:
        self.matrix_path = matrix_path
        self.ranks, self.matrix = self._load_matrix()

    def _load_matrix(self) -> tuple[list[str], list[list[str]]]:
        with self.matrix_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload["ranks"], payload["matrix"]

    def _hand_to_indices(self, rank_left: str, rank_right: str) -> Optional[tuple[int, int]]:
        if rank_left not in self.ranks or rank_right not in self.ranks:
            return None
        row = self.ranks.index(rank_left)
        col = self.ranks.index(rank_right)
        return row, col

    def lookup(self, rank_left: Optional[str], rank_right: Optional[str]) -> StrategyResult:
        if not rank_left or not rank_right:
            return StrategyResult(hand="Unknown", zone="Unknown", action="Hold")
        indices = self._hand_to_indices(rank_left, rank_right)
        if not indices:
            return StrategyResult(
                hand=f"{rank_left}{rank_right}", zone="Unknown", action="Hold"
            )
        row, col = indices
        zone = self.matrix[row][col]
        action = "4-Bet Bluff" if zone.lower() == "red" else "Check"
        return StrategyResult(hand=f"{rank_left}{rank_right}", zone=zone, action=action)
