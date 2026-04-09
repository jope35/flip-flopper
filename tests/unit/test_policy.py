from flip_flopper_ab_test.policy import (
    ArmMetrics,
    CurrentSplit,
    GuardrailConfig,
    PolicyDecision,
    propose_split,
)

GUARDRAILS = GuardrailConfig(
    min_challenger_percent=10,
    max_challenger_percent=50,
    max_step_percent=10,
    min_support_per_arm=200,
    max_error_rate=0.05,
    max_avg_execution_ms=1200,
)


def test_policy_holds_when_support_is_too_low() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=120, label_rate=0.31, error_rate=0.01, avg_execution_duration_ms=390),
        },
        current_split=CurrentSplit(control=90, challenger=10),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision == PolicyDecision(
        control=90,
        challenger=10,
        action="hold",
        reason="minimum_support_not_met",
    )


def test_policy_steps_up_challenger_when_it_wins() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=500, label_rate=0.28, error_rate=0.01, avg_execution_duration_ms=380),
        },
        current_split=CurrentSplit(control=90, challenger=10),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision.challenger == 20
    assert decision.action == "apply"
    assert decision.reason == "challenger_outperforming"


def test_policy_rolls_back_when_operational_guardrail_breaks() -> None:
    decision = propose_split(
        metrics_by_arm={
            "control": ArmMetrics(support=500, label_rate=0.20, error_rate=0.01, avg_execution_duration_ms=400),
            "challenger": ArmMetrics(support=500, label_rate=0.28, error_rate=0.09, avg_execution_duration_ms=1600),
        },
        current_split=CurrentSplit(control=70, challenger=30),
        guardrails=GUARDRAILS,
        last_good_split=CurrentSplit(control=90, challenger=10),
        shadow_mode=False,
    )

    assert decision == PolicyDecision(
        control=90,
        challenger=10,
        action="rollback",
        reason="operational_guardrail_breached",
    )
