from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ArmMetrics:
    support: int
    label_rate: float
    error_rate: float
    avg_execution_duration_ms: float


@dataclass(frozen=True)
class CurrentSplit:
    control: int
    challenger: int


@dataclass(frozen=True)
class PolicyDecision:
    control: int
    challenger: int
    action: Literal["hold", "apply", "rollback", "shadow"]
    reason: str


@dataclass(frozen=True)
class GuardrailConfig:
    min_challenger_percent: int
    max_challenger_percent: int
    max_step_percent: int
    min_support_per_arm: int
    max_error_rate: float
    max_avg_execution_ms: int


def propose_split(
    *,
    metrics_by_arm: dict[str, ArmMetrics],
    current_split: CurrentSplit,
    guardrails: GuardrailConfig,
    last_good_split: CurrentSplit,
    shadow_mode: bool,
) -> PolicyDecision:
    control = metrics_by_arm["control"]
    challenger = metrics_by_arm["challenger"]

    if control.support < guardrails.min_support_per_arm or challenger.support < guardrails.min_support_per_arm:
        return PolicyDecision(
            control=current_split.control,
            challenger=current_split.challenger,
            action="hold",
            reason="minimum_support_not_met",
        )

    if (
        challenger.error_rate > guardrails.max_error_rate
        or challenger.avg_execution_duration_ms > guardrails.max_avg_execution_ms
    ):
        return PolicyDecision(
            control=last_good_split.control,
            challenger=last_good_split.challenger,
            action="rollback",
            reason="operational_guardrail_breached",
        )

    if challenger.label_rate <= control.label_rate:
        return PolicyDecision(
            control=current_split.control,
            challenger=current_split.challenger,
            action="hold",
            reason="challenger_not_better",
        )

    next_challenger = min(
        current_split.challenger + guardrails.max_step_percent,
        guardrails.max_challenger_percent,
    )
    next_challenger = max(next_challenger, guardrails.min_challenger_percent)
    return PolicyDecision(
        control=100 - next_challenger,
        challenger=next_challenger,
        action="shadow" if shadow_mode else "apply",
        reason="challenger_outperforming",
    )
