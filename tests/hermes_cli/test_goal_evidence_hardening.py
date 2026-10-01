import json

import hermes_cli.goals as goals


def _manager_with_state(monkeypatch, state):
    monkeypatch.setattr(goals, "load_goal", lambda _sid: None)
    monkeypatch.setattr(goals, "save_goal", lambda _sid, st: st)
    mgr = goals.GoalManager("hardening-test")
    mgr._state = state
    return mgr


def test_restored_goal_helper_dataclasses_are_constructible():
    control = goals.DecompositionScopeControl(
        scope="simple", min_items=2, max_items=5, guidance="compact"
    )
    assert control.scope == "simple"
    assert control.max_items == 5

    ref = goals.GoalReference(kind="file", reference="spec.md", status="resolved")
    context = goals.GoalReferenceContext(references=[ref])
    assert context.has_content() is False
    assert context.to_audit_dict()["reference_count"] == 1


def test_decomposition_scope_control_constructs_expected_shape():
    simple = goals.decomposition_scope_control("say hello")
    assert simple.scope == "simple"
    assert simple.min_items == 2
    assert simple.max_items == 5

    complex_goal = goals.decomposition_scope_control(
        "implement and deploy a tested API service with reliability verification"
    )
    assert complex_goal.scope == "complex"
    assert complex_goal.min_items >= 8


def test_goal_state_roundtrips_evidence_checklist_and_assumptions():
    state = goals.GoalState(
        goal="verify deployment",
        checklist=[goals.ChecklistItem(text="run tests")],
        assumption_ledger=[{"kind": "hypothesis", "statement": "cache is stale", "source": "agent_response"}],
    )
    goals._add_ledger_entry(
        state,
        evidence_type=goals.EVIDENCE_TYPE_TEST,
        source=goals.EVIDENCE_SOURCE_TOOL,
        summary="pytest: 12 passed",
        status="passed",
    )
    restored = goals.GoalState.from_json(state.to_json())
    assert restored.checklist[0].text == "run tests"
    assert restored.evidence_ledger[0].source == goals.EVIDENCE_SOURCE_TOOL
    assert restored.evidence_ledger[0].summary == "pytest: 12 passed"
    assert restored.assumption_ledger[0]["kind"] == "hypothesis"


def test_completion_evidence_population_has_provenance_and_no_nameerror():
    state = goals.GoalState(goal="x")
    evidence = goals.CompletionEvidence(raw_present=True, verification_performed=["pytest -q: 3 passed"])
    goals._populate_ledger_from_evidence(state, evidence)
    assert len(state.evidence_ledger) == 1
    assert state.evidence_ledger[0].evidence_type == goals.EVIDENCE_TYPE_TEST
    assert state.evidence_ledger[0].source == goals.EVIDENCE_SOURCE_AGENT


def test_assumption_ledger_only_parses_explicit_typed_block():
    state = goals.GoalState(goal="debug USB", turns_used=2)
    response = """Investigation update.\n\n## ASSUMPTION LEDGER\n- observed: usb.list returned 0 devices\n- hypothesis: USB role negotiation changed\n- unknown: why physical connection differs from kernel state\n- disproved: watch was physically unplugged\n"""
    goals._populate_assumption_ledger_from_response(state, response)
    assert [e["kind"] for e in state.assumption_ledger] == ["observed", "hypothesis", "unknown", "disproved"]
    assert all(e["source"] == "agent_response" for e in state.assumption_ledger)
    # Ordinary prose must not be guessed into the ledger.
    before = list(state.assumption_ledger)
    goals._populate_assumption_ledger_from_response(state, "Maybe the cable is bad.")
    assert state.assumption_ledger == before


def test_contract_completion_guard_rejects_bare_claim():
    state = goals.GoalState(
        goal="ship fix",
        contract=goals.GoalContract(verification="pytest -q passes"),
    )
    ok, reason = goals._contract_completion_evidence_ok(state, goals.CompletionEvidence())
    assert ok is False
    assert "structured verification" in reason


def test_contract_completion_guard_accepts_structured_verification():
    state = goals.GoalState(
        goal="ship fix",
        contract=goals.GoalContract(verification="pytest -q passes"),
    )
    evidence = goals.CompletionEvidence(
        raw_present=True,
        verification_performed=["pytest -q: 12 passed"],
        declares_no_known_gaps=True,
        declares_no_blockers=True,
    )
    ok, reason = goals._contract_completion_evidence_ok(state, evidence)
    assert ok is True
    assert "structured verification" in reason


def test_contract_completion_guard_accepts_passed_deterministic_gate():
    state = goals.GoalState(
        goal="ship fix",
        contract=goals.GoalContract(verification="pytest -q passes"),
        gates=[goals.GoalGate(command="pytest -q", last_exit_code=0)],
    )
    ok, reason = goals._contract_completion_evidence_ok(state, goals.CompletionEvidence())
    assert ok is True
    assert "quality gates" in reason


def test_goalmanager_withholds_done_when_contract_verification_missing(monkeypatch):
    state = goals.GoalState(
        goal="ship fix",
        contract=goals.GoalContract(verification="pytest -q passes"),
    )
    mgr = _manager_with_state(monkeypatch, state)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: ("done", "looks complete", False, None, False),
    )
    decision = mgr.evaluate_after_turn("Implemented the change.")
    assert decision["status"] == "active"
    assert decision["verdict"] == "continue"
    assert mgr.state.status == "active"
    assert "completion evidence was insufficient" in mgr.state.last_reason


def test_goalmanager_can_finish_with_structured_verification(monkeypatch):
    state = goals.GoalState(
        goal="ship fix",
        contract=goals.GoalContract(verification="pytest -q passes"),
    )
    mgr = _manager_with_state(monkeypatch, state)
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda *a, **k: ("done", "verified", False, None, False),
    )
    response = """COMPLETION EVIDENCE
- Verification performed: pytest -q: 12 passed
- Counts or reconciliations: 12 passed, 0 failed
- Known gaps: none
- Blockers: none
"""
    decision = mgr.evaluate_after_turn(response)
    assert decision["status"] == "done"
    assert mgr.state.status == "done"
    assert mgr.state.last_completion_evidence["raw_present"] is True
    assert mgr.state.evidence_ledger
