"""Helpers to validate and apply classifier endpoint traffic splits via Databricks bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TrafficSplit:
    champion: int
    challenger: int
    catboost: int
    lightgbm: int = 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.champion, self.challenger, self.catboost, self.lightgbm)


def validate_traffic_split(
    champion: int,
    challenger: int,
    catboost: int,
    lightgbm: int = 0,
) -> TrafficSplit:
    for name, value in (
        ("champion", champion),
        ("challenger", challenger),
        ("catboost", catboost),
        ("lightgbm", lightgbm),
    ):
        if value < 0 or value > 100:
            msg = f"{name} traffic must be between 0 and 100, got {value}"
            raise ValueError(msg)
    total = champion + challenger + catboost + lightgbm
    if total != 100:
        msg = f"traffic percentages must sum to 100, got {total}"
        raise ValueError(msg)
    return TrafficSplit(
        champion=champion,
        challenger=challenger,
        catboost=catboost,
        lightgbm=lightgbm,
    )


def format_job_params(split: TrafficSplit) -> str:
    parts = [
        f"traffic_a={split.champion}",
        f"traffic_b={split.challenger}",
        f"traffic_c={split.catboost}",
    ]
    if split.lightgbm:
        parts.append(f"traffic_d={split.lightgbm}")
    return ",".join(parts)


def build_validate_command(*, target: str, profile: str | None) -> list[str]:
    cmd = ["databricks", "bundle", "validate", "-t", target]
    if profile:
        cmd.extend(["-p", profile])
    return cmd


def build_deploy_command(*, target: str, profile: str | None) -> list[str]:
    cmd = ["databricks", "bundle", "deploy", "--auto-approve", "-t", target]
    if profile:
        cmd.extend(["-p", profile])
    return cmd


def build_run_command(
    *,
    target: str,
    profile: str | None,
    split: TrafficSplit,
) -> list[str]:
    cmd = [
        "databricks",
        "bundle",
        "run",
        "-t",
        target,
        "--params",
        format_job_params(split),
        "deploy_classifier_endpoint",
    ]
    if profile:
        cmd.extend(["-p", profile])
    return cmd


def build_bundle_pipeline_commands(
    *,
    target: str,
    profile: str | None,
    split: TrafficSplit,
) -> list[list[str]]:
    return [
        build_validate_command(target=target, profile=profile),
        build_deploy_command(target=target, profile=profile),
        build_run_command(target=target, profile=profile, split=split),
    ]


def resolve_split_from_args(
    champion: int | None,
    challenger: int | None,
    catboost: int | None,
    lightgbm: int | None = None,
) -> TrafficSplit | None:
    core = (champion, challenger, catboost)
    if all(v is None for v in core) and lightgbm is None:
        return None
    if any(v is None for v in core):
        msg = "provide all of --traffic-a, --traffic-b, and --traffic-c"
        raise ValueError(msg)
    return validate_traffic_split(champion, challenger, catboost, 0 if lightgbm is None else lightgbm)


def resolve_split_from_env() -> TrafficSplit | None:
    keys = (
        "FLIP_FLOPPER_TRAFFIC_A",
        "FLIP_FLOPPER_TRAFFIC_B",
        "FLIP_FLOPPER_TRAFFIC_C",
        "FLIP_FLOPPER_TRAFFIC_D",
    )
    raw = [os.environ.get(key) for key in keys]
    if all(v is None for v in raw[:3]):
        return None
    if any(v is None for v in raw[:3]):
        return None
    return validate_traffic_split(
        int(raw[0]),  # type: ignore[arg-type]
        int(raw[1]),  # type: ignore[arg-type]
        int(raw[2]),  # type: ignore[arg-type]
        int(raw[3] or "0"),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update classifier endpoint traffic via bundle deploy+run.")
    p.add_argument("-t", "--target", default="dev")
    p.add_argument("-p", "--profile", default=None)
    p.add_argument("--traffic-a", type=int, default=None)
    p.add_argument("--traffic-b", type=int, default=None)
    p.add_argument("--traffic-c", type=int, default=None)
    p.add_argument("--traffic-d", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    ns = _parse_args(argv)
    split = resolve_split_from_args(ns.traffic_a, ns.traffic_b, ns.traffic_c, ns.traffic_d)
    if split is None:
        split = resolve_split_from_env()
    if split is None:
        split = validate_traffic_split(25, 25, 25, 25)

    pipeline = build_bundle_pipeline_commands(target=ns.target, profile=ns.profile, split=split)
    for cmd in pipeline:
        print("$", " ".join(cmd))
        if not ns.dry_run:
            subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
