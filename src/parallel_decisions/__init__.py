"""parallel-decisions: typed, locally-run LLM decisions.

    from parallel_decisions import Decider, Schema

    decider = Decider()
    schema = Schema({"risk": {"type": "enum", "choices": ["LOW", "HIGH"], "description": "..."}})
    result = decider.decide(context, schema)
    result["risk"].value        # "HIGH"
    result["risk"].probability  # 0.91
"""

from .engine import DEFAULT_MODEL, Decider, DecisionResult, FieldValue
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
]

__version__ = "0.1.0"
