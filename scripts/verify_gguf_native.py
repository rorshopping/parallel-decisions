"""Opt-in synthetic GGUF contract check through Decider; NOT proof of parallelism.

Help and pytest contract tests never load a model. Real execution requires both
--run-native and --exclusive-model-use after coordinating with the engine owner.
See NOTES_GGUF_ACCEPTANCE.md for the separate native architecture evidence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = {
    "a": "Synthetic parcel record: fragile yes; color RED; parcel red; tags dry and light.",
    "b": "Synthetic parcel record: fragile no; color BLUE; parcel blue; tags heavy.",
    "long": ("Synthetic inventory note: shelf seven contains sealed empty boxes. " * 24
             + "Parcel record: fragile no; color GREEN; other; tags dry and heavy."),
}
COLLISION_POOLS = (
    ["parcel red", "parcel blue", "other"],
    ["same first alpha", "same first beta", "different"],
    ["123456 red", "123456 blue", "none"],
)


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def probability(value: Any, where: str) -> float:
    require(isinstance(value, (float, int)) and not isinstance(value, bool),
            f"{where}: expected numeric probability")
    require(math.isfinite(value) and 0 <= value <= 1, f"{where}: invalid probability {value}")
    return float(value)


def schema_spec(collision_choices: list[str]) -> dict:
    return {
        "fragile": {"type": "boolean", "description": "Is the parcel fragile?"},
        "color": {"type": "enum", "choices": ["RED", "BLUE", "GREEN"],
                  "description": "Recorded parcel color"},
        "tags": {"type": "multi", "choices": ["dry", "light", "heavy"],
                 "description": "Recorded parcel tags"},
        "long_named_collision_field": {"type": "enum", "choices": collision_choices,
                                       "description": "Recorded parcel phrase"},
    }


def select_schema(Schema: Any, tokenizer: Any) -> tuple[Any, list]:
    # Global common-prefix removal is part of Schema.compile: merely using
    # ["parcel red", "parcel blue"] would NOT guarantee a collision.
    for choices in COLLISION_POOLS:
        schema = Schema(schema_spec(choices))
        compiled = schema.compile(tokenizer)
        collision = compiled[-1]
        sequences = [tuple(s) for s in collision.sequences]
        if (collision.collision and all(sequences)
                and len(set(sequences)) == len(sequences)
                and max(map(len, sequences)) > 1):
            return schema, compiled
    raise ContractError("No exact, distinct multi-token collision in bounded fixture pools; "
                        "add a reviewed tokenizer-specific fixture, do not skip the gate")


def validate_result(result: Any, schema: Any, rows: int, *, calibrated: bool = False) -> dict:
    require(set(result) == set(schema.fields), "unexpected/missing result fields")
    require(result.calibrated is calibrated, "result calibration flag mismatch")
    require(result.fields_evaluated == rows, "fields_evaluated must count expanded multi rows")
    require(isinstance(result.chunks, int) and result.chunks > 0, "missing chunk count")
    for key in ("latency_ms", "prefill_ms", "pass_ms"):
        value = getattr(result, key)
        require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
                f"invalid {key}")
    require(result.telemetry.get("backend") == "llamacpp", "not the requested backend")
    values = {}
    for name, field in schema.fields.items():
        fv = result[name]
        require(fv.name == name, f"{name}: field name mismatch")
        require(fv.approximate is False, f"{name}: approximate collision result")
        require(fv.calibrated is calibrated, f"{name}: calibration flag mismatch")
        p = probability(fv.probability, name)
        raw = probability(fv.raw_probability, name + ".raw")
        if not calibrated:
            require(abs(p - raw) <= 1e-7, f"{name}: raw confidence changed without calibration")
        allowed = field.choices if field.is_multi else field.answers
        require(set(fv.distribution) == set(allowed), f"{name}: distribution keys")
        dist = {k: probability(v, f"{name}.{k}") for k, v in fv.distribution.items()}
        if field.is_multi:
            require(type(fv.value) is list and all(type(v) is str for v in fv.value),
                    f"{name}: multi must be list[str]")
            require(len(set(fv.value)) == len(fv.value), f"{name}: duplicate multi value")
            require(fv.value == [c for c in allowed if dist[c] >= 0.5],
                    f"{name}: multi inclusion must follow independent marginals")
            expected = (min(dist[c] for c in fv.value) if fv.value else 1 - max(dist.values()))
            # Multi marginals are NOT a categorical distribution; never sum to one here.
        else:
            require(abs(sum(dist.values()) - 1) <= 1e-5, f"{name}: distribution not normalized")
            if field.is_boolean:
                require(type(fv.value) is bool, f"{name}: boolean is not bool")
                selected = str(fv.value).lower()
            else:
                require(type(fv.value) is str and fv.value in allowed, f"{name}: invalid enum")
                selected = fv.value
            expected = dist[selected]
            require(expected >= max(dist.values()) - 1e-7, f"{name}: selected non-max candidate")
        require(abs(p - expected) <= 1e-6, f"{name}: confidence does not match distribution")
        for alternative, score in fv.alternatives:
            require(alternative in dist, f"{name}: unknown alternative")
            require(abs(probability(score, name) - dist[alternative]) <= 1e-6,
                    f"{name}: alternative score mismatch")
        value = list(fv.value) if field.is_multi else fv.value
        values[name] = {"value": value, "probability": p, "raw_probability": raw,
                        "distribution": dist}
    require(result.json() == {k: v["value"] for k, v in values.items()}, "json values differ")
    require(set(result.full_json()) == set(values), "full_json fields differ")
    json.dumps(result.json(), allow_nan=False)
    json.dumps(result.full_json(), allow_nan=False)
    return {"fields": values, "latency_ms": result.latency_ms, "prefill_ms": result.prefill_ms,
            "pass_ms": result.pass_ms, "chunks": result.chunks,
            "fields_evaluated": result.fields_evaluated, "telemetry": result.telemetry}


def compare(left: dict, right: dict, atol: float) -> dict:
    require(set(left["fields"]) == set(right["fields"]), "comparison schema differs")
    delta = 0.0
    changed = []
    for name, old in left["fields"].items():
        new = right["fields"][name]
        require(set(old["distribution"]) == set(new["distribution"]), "comparison labels differ")
        if old["value"] != new["value"]:
            changed.append(name)
        delta = max(delta, *(abs(p - new["distribution"][k])
                             for k, p in old["distribution"].items()))
    return {"passed": not changed and delta <= atol, "changed_values": changed,
            "max_abs_distribution_delta": delta, "atol": atol}


def run_suite(decider: Any, Schema: Any, Calibrator: Any, *, repeats: int,
              atol: float, report: dict) -> None:
    schema, compiled = select_schema(Schema, decider.tokenizer)
    report["schema"] = schema.to_list()
    report["compiled_rows"] = [
        {"name": cf.row_name, "suffix_tokens": cf.suffix_tokens,
         "candidate_ids": cf.candidate_ids, "sequences": cf.sequences,
         "collision": cf.collision} for cf in compiled]
    report["runs"] = []
    report["comparisons"] = []
    baselines = {}
    retained = []
    # One loaded Decider, same FULL schema across row limits. A singleton-schema
    # prompt changes the schema block and is not an equivalence oracle.
    for repetition in range(repeats):
        for limit in ((8, 2, 1) if repetition % 2 == 0 else (1, 2, 8)):
            decider.max_fields_per_batch = limit
            decider.max_collision_rows = limit
            for context_id in ("a", "b", "long", "a"):
                start = time.perf_counter()
                result = decider.decide(CONTEXTS[context_id], schema)
                wall_ms = (time.perf_counter() - start) * 1000
                snapshot = validate_result(result, schema, len(compiled))
                require(snapshot["chunks"] == math.ceil(len(compiled) / limit),
                        "chunk limit was not applied (including singleton-row mode)")
                snapshot.update(context_id=context_id, row_limit=limit, repetition=repetition,
                                wall_ms=wall_ms, cold_request=not report["runs"])
                report["runs"].append(snapshot)
                retained.append((result, snapshot))
                if context_id in baselines:
                    check = compare(baselines[context_id], snapshot, atol)
                    check.update(context_id=context_id, row_limit=limit, repetition=repetition)
                    report["comparisons"].append(check)
                else:
                    baselines[context_id] = snapshot
    # Retained outputs must not alias native logits buffers overwritten by later calls.
    for result, snapshot in retained:
        current = validate_result(result, schema, len(compiled))
        require(compare(snapshot, current, 0)["passed"], "retained result mutated after later calls")

    # A single-field smoke check, deliberately NOT compared to the full-schema prompt.
    singleton = Schema({"only": {"type": "boolean", "description": "Is the parcel fragile?"}})
    report["singleton_schema"] = validate_result(
        decider.decide(CONTEXTS["a"], singleton), singleton, 1)

    # Public calibrator attribute, as exposed by the existing facade. This is a
    # synthetic transform contract, not calibration fitted on representative data.
    decider.max_fields_per_batch = 8
    decider.calibrator = Calibrator(kind="temperature", temperature=2.0,
                                   meta={"synthetic_contract_only": True})
    calibrated = validate_result(decider.decide(CONTEXTS["a"], schema), schema,
                                 len(compiled), calibrated=True)
    report["calibration"] = calibrated
    for name, field in schema.fields.items():
        old, new = baselines["a"]["fields"][name], calibrated["fields"][name]
        require(old["value"] == new["value"], "calibration changed a selected value")
        require(abs(old["raw_probability"] - new["raw_probability"]) <= atol,
                "calibration did not preserve raw confidence")
        expected = ({k: decider.calibrator.transform_confidence(p)
                     for k, p in old["distribution"].items()} if field.is_multi
                    else decider.calibrator.transform(old["distribution"]))
        require(all(abs(expected[k] - new["distribution"][k]) <= atol for k in expected),
                "calibration distribution differs from public Calibrator transform")
    decider.calibrator = None
    report["context_distribution_delta"] = compare(baselines["a"], baselines["b"], atol)[
        "max_abs_distribution_delta"]
    report["context_sensitivity_observed"] = report["context_distribution_delta"] > atol
    require(all(c["passed"] for c in report["comparisons"]),
            "singleton/chunk/repeated-context equivalence failed; inspect raw comparisons")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-native", action="store_true", help="Explicitly opt in to model inference")
    p.add_argument("--exclusive-model-use", action="store_true",
                   help="Acknowledge coordination: no engine implementer/model inference running")
    p.add_argument("--model", type=Path, help="Existing local .gguf only; never downloaded")
    checksums = p.add_mutually_exclusive_group()
    checksums.add_argument("--sha256", help="Expected 64-hex model checksum")
    checksums.add_argument("--sha256-file", type=Path, help="Explicit checksum file (first word is hash)")
    p.add_argument("--output", type=Path, help="New JSON report path; existing files are not overwritten")
    p.add_argument("--repeats", type=int, default=2, help="Suite repetitions, 1..10 (default: 2)")
    p.add_argument("--atol", type=float, default=1e-4, help="Absolute distribution tolerance (default: 1e-4)")
    return p


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if not args.run_native or not args.exclusive_model_use:
        p.error("inference disabled: requires --run-native AND --exclusive-model-use")
    if args.model is None or args.output is None:
        p.error("--model and --output are required for native execution")
    if not 1 <= args.repeats <= 10 or not math.isfinite(args.atol) or not 0 <= args.atol <= 0.01:
        p.error("require repeats in 1..10 and finite atol in 0..0.01")
    model = args.model.resolve()
    if not model.is_file() or model.suffix.lower() != ".gguf":
        p.error("--model must be an existing local .gguf file")
    if args.output.exists() or not args.output.parent.is_dir():
        p.error("--output must be a new file in an existing directory")
    expected = args.sha256
    if args.sha256_file:
        words = args.sha256_file.read_text(encoding="utf-8-sig").split()
        expected = words[0] if words else ""
    if expected is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", expected):
        p.error("expected SHA-256 must be exactly 64 hex digits")
    report = {"status": "not_run", "synthetic_only": True, "accuracy_evaluated": False,
              "native_parallelism": "NOT_VERIFIED: architecture review and native trace required",
              "model": str(model), "python": sys.version, "platform": platform.platform(),
              "repeats": args.repeats, "atol": args.atol, "contexts": CONTEXTS,
              "warmup": False, "limitations": [
                  "Functional agreement is not proof of native KV branching or batched suffixes.",
                  "Reported phase timings require independent synchronization review.",
                  "No GPU/offload or accuracy claim follows from a passing synthetic contract."]}
    try:
        start = time.perf_counter()
        report["model_sha256"] = sha256(model)
        report["checksum_ms"] = (time.perf_counter() - start) * 1000
        report["checksum_verified_against_expected"] = expected is not None
        require(expected is None or report["model_sha256"] == expected.lower(), "model checksum mismatch")
        # Pin THIS checkout, never an environment's potentially unrelated editable install.
        sys.path.insert(0, str(ROOT / "src"))
        import parallel_decisions
        from parallel_decisions import Calibrator, Decider, Schema
        from parallel_decisions.config import Config
        require(Path(parallel_decisions.__file__).resolve().is_relative_to(ROOT / "src"),
                "import provenance is outside the selected checkout")
        report["package_path"] = parallel_decisions.__file__
        report["source_sha256"] = {f.name: sha256(f) for f in sorted(
            (ROOT / "src" / "parallel_decisions").glob("*.py"))}
        report["llama_cpp_python_version"] = importlib.metadata.version("llama-cpp-python")
        start = time.perf_counter()
        decider = Decider(model_id=str(model), backend="llamacpp", max_fields_per_batch=8,
                          max_collision_rows=8, warmup=False, config=Config(log="off"))
        decider.load()  # Explicit: constructor is lazy in the existing public facade.
        report["load_ms"] = (time.perf_counter() - start) * 1000
        run_suite(decider, Schema, Calibrator, repeats=args.repeats, atol=args.atol, report=report)
        report["status"] = "FUNCTIONAL_CONTRACT_PASS_NATIVE_PROOF_PENDING"
        code = 0
    except Exception as exc:
        report["status"] = "FAILED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        code = 1
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2, allow_nan=False)
        target.write("\n")
    print(f"{report['status']}: {args.output}")
    print(report["native_parallelism"])
    if code:
        print(report["error"], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
