"""parallel-decisions: typed, locally-run LLM decisions.

    from parallel_decisions import Decider, Schema

    decider = Decider()
    schema = Schema({"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "..."}})
    result = decider.decide(context, schema)
    result["risk"].value        # "HIGH"
    result["risk"].probability  # 0.91

With a fitted calibrator, `probability` becomes a number you can threshold on:

    from parallel_decisions import Calibrator
    decider = Decider(calibration="calibration.json")
    decider.decide(context, schema)["risk"].probability  # calibrated
"""

from .calibration import (
    CalibrationError,
    CalibrationFit,
    CalibrationRecord,
    Calibrator,
    adaptive_ece,
    auroc,
    confidence,
    ece,
    filter_records,
    fit_calibration,
    load_records,
    reliability_table,
    slices,
    risk_coverage,
    wilson_interval,
)
from .config import Config, ConfigError, load_config
from .engine import (
    DEFAULT_MODEL,
    ConcurrencyError,
    Decider,
    DecisionResult,
    FieldValue,
    UnsupportedPlatformError,
)
from .schema import MAX_CHOICES, Field, Schema, SchemaError

__all__ = [
    "DEFAULT_MODEL",
    "Decider",
    "DecisionResult",
    "FieldValue",
    "Schema",
    "Field",
    "SchemaError",
    "MAX_CHOICES",
    "Calibrator",
    "CalibrationRecord",
    "CalibrationFit",
    "CalibrationError",
    "fit_calibration",
    "load_records",
    "filter_records",
    "slices",
    "reliability_table",
    "risk_coverage",
    "wilson_interval",
    "ece",
    "adaptive_ece",
    "auroc",
    "confidence",
    "Config",
    "ConfigError",
    "load_config",
    "ConcurrencyError",
    "UnsupportedPlatformError",
]

__version__ = "0.3.0"
