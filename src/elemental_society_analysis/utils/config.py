"""Load YAML configuration into a typed dataclass."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class CorrectionConfig:
    method: str
    zero_strategy: str
    tiny_value: float


@dataclass
class AnomalyConfig:
    method: str
    n_neighbors: int
    contamination: float


@dataclass
class IOConfig:
    raw_path: str
    sample_path: str
    encoding: str
    separator: str
    decimal: str
    output_dir: str
    id_columns: list[str]
    coord_columns: list[str]
    min_coverage: float
    core_features: list[str]


@dataclass
class LoggingConfig:
    level: str
    format: str


@dataclass
class Config:
    correction: CorrectionConfig
    anomaly: AnomalyConfig
    io: IOConfig
    logging: LoggingConfig


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    """Load the YAML config into typed dataclasses."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(
        correction=CorrectionConfig(**data["correction"]),
        anomaly=AnomalyConfig(**data["anomaly"]),
        io=IOConfig(**data["io"]),
        logging=LoggingConfig(**data["logging"]),
    )
