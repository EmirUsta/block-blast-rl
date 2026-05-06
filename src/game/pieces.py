"""Parça tanımları (immutable bool matrix).

Block Blast'ta dynamic rotation yok; her rotation ayrı parça olarak YAML'da listelenir.
`type_idx` parça setindeki sıraya göre verilir; one-hot encoding'de kullanılır.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class Piece:
    id: str
    type_idx: int
    cells: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if self.cells.dtype != np.bool_:
            object.__setattr__(self, "cells", self.cells.astype(bool))
        if self.cells.ndim != 2:
            raise ValueError(f"Piece {self.id} cells must be 2D, got {self.cells.shape}")
        if not self.cells.any():
            raise ValueError(f"Piece {self.id} has no filled cells")

    @property
    def height(self) -> int:
        return int(self.cells.shape[0])

    @property
    def width(self) -> int:
        return int(self.cells.shape[1])

    @property
    def n_cells(self) -> int:
        return int(self.cells.sum())


def load_piece_set(yaml_path: str | Path) -> list[Piece]:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    pieces: list[Piece] = []
    seen: set[str] = set()
    for idx, item in enumerate(data["pieces"]):
        pid = item["id"]
        if pid in seen:
            raise ValueError(f"Duplicate piece id: {pid}")
        seen.add(pid)
        cells = np.array(item["cells"], dtype=bool)
        pieces.append(Piece(id=pid, type_idx=idx, cells=cells))
    return pieces
