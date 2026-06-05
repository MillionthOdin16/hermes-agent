"""Tests for hermes_cli/goals.py — persistent cross-turn goals."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Bust the goal-module's DB cache for each test so it re-resolves HERMES_HOME.
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


# ──────────────────────────────────────────────────────────────────────
# _parse_judge_response
# ──────────────────────────────────────────────────────────────────────


class TestParseJudgeResponse:
    def test_clean_json_done(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response('{"done": true, "reason": "all good"}')
        assert done is True
        assert reason == "all good"

    def test_clean_json_continue(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response('{"done": false, "reason": "more work needed"}')
        assert done is False
        assert reason == "more work needed"

    def test_json_in_markdown_fence(self):
        from hermes_cli.goals import _parse_judge_response

        raw = '```json\n{"done": true, "reason": "done"}\n```'
        done, reason, _ = _parse_judge_response(raw)
        assert done is True
        assert "done" in reason

    def test_json_embedded_in_prose(self):
        """Some models prefix reasoning before emitting JSON — we extract it."""
        from hermes_cli.goals import _parse_judge_response

        raw = 'Looking at this... the agent says X. Verdict: {"done": false, "reason": "partial"}'
        done, reason, _ = _parse_judge_response(raw)
        assert done is False
        assert reason == "partial"

    def test_string_done_values(self):
        from hermes_cli.goals import _parse_judge_response

        for s in ("true", "yes", "done", "1"):
            done, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is True
        for s in ("false", "no", "not yet"):
            done, _, _ = _parse_judge_response(f'{{"done": "{s}", "reason": "r"}}')
            assert done is False

    def test_malformed_json_fails_open(self):
        """Non-JSON → not done, with error-ish reason (so judge_goal can map to continue)."""
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response("this is not json at all")
        assert done is False
        assert reason  # non-empty

    def test_empty_response(self):
        from hermes_cli.goals import _parse_judge_response

        done, reason, _ = _parse_judge_response("")
        assert done is False
        assert reason


# ──────────────────────────────────────────────────────────────────────
# judge_goal — fail-open semantics
# ──────────────────────────────────────────────────────────────────────


class TestJudgeGoal:
    def test_empty_goal_skipped(self):
        from hermes_cli.goals import judge_goal

        verdict, _, _ = judge_goal("", "some response")
        assert verdict == "skipped"

    def test_empty_response_continues(self):
        from hermes_cli.goals import judge_goal

        verdict, _, _ = judge_goal("ship the thing", "")
        assert verdict == "continue"

    def test_no_aux_client_continues(self):
        """Fail-open: if no aux client, we must return continue, not skipped/done."""
        from hermes_cli import goals

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(None, None),
        ):
            verdict, _, _ = goals.judge_goal("my goal", "my response")
        assert verdict == "continue"

    def test_api_error_continues(self):
        """Judge exception → fail-open continue (don't wedge progress on judge bugs)."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("boom")
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert "judge error" in reason.lower()

    def test_judge_says_done(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": true, "reason": "achieved"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "agent response")
        assert verdict == "done"
        assert reason == "achieved"

    def test_judge_says_continue(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content='{"done": false, "reason": "not yet"}')
                )
            ]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, reason, _ = goals.judge_goal("goal", "agent response")
        assert verdict == "continue"
        assert reason == "not yet"


# ──────────────────────────────────────────────────────────────────────
# GoalManager lifecycle + persistence
# ──────────────────────────────────────────────────────────────────────


class TestGoalManager:
    def test_no_goal_initial(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-1")
        assert mgr.state is None
        assert not mgr.is_active()
        assert not mgr.has_goal()
        assert "No active goal" in mgr.status_line()

    def test_set_then_status(self, hermes_home):
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="test-sid-2", default_max_turns=5)
        state = mgr.set("port the thing")
        assert state.goal == "port the thing"
        assert state.status == "active"
        assert "active" in mgr.status_line()

    def test_goal_persists_across_reloads(self, hermes_home):
        from hermes_cli.goals import GoalManager, load_goal

        mgr = GoalManager(session_id="test-sid-3")
        mgr.set("keep it")

        reloaded = load_goal("test-sid-3")
        assert reloaded is not None
        assert reloaded.goal == "keep it"
        assert reloaded.status == "active"

    def test_pause_and_resume(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalStatus

        mgr = GoalManager(session_id="test-sid-4")
        mgr.set("do something")
        mgr.pause("user-request")
        assert mgr.state.status == GoalStatus.PAUSED.value
        assert mgr.state.paused_reason == "user-request"

        mgr.resume()
        assert mgr.state.status == GoalStatus.ACTIVE.value
        assert mgr.state.paused_reason is None

    def test_clear(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalStatus, load_goal

        mgr = GoalManager(session_id="test-sid-5")
        mgr.set("finish me")
        mgr.clear()
        assert mgr.state is None
        reloaded = load_goal("test-sid-5")
        assert reloaded.status == GoalStatus.CLEARED.value


# ──────────────────────────────────────────────────────────────────────
# Goal decomposition
# ──────────────────────────────────────────────────────────────────────


class TestGoalDecomposition:
    def test_parse_decomposition_text(self):
        from hermes_cli.goals import _parse_decomposition_text

        raw = """
        1. Gather logs
        2. Inspect error
        3. Fix bug
        """
        checklist, notes = _parse_decomposition_text(raw)
        assert len(checklist) == 3
        assert notes == ""

    def test_parse_decomposition_text_ignores_notes_block(self):
        from hermes_cli.goals import _parse_decomposition_text

        raw = """
        1. Gather logs
        2. Inspect error
        Notes: here are some notes that should be ignored
        """
        checklist, notes = _parse_decomposition_text(raw)
        assert len(checklist) == 2
        assert notes.startswith("Notes:")

    def test_simple_goal_decomposition_is_capped_and_deduplicated(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content=json.dumps({
                        "checklist": [
                            {"text": "Say hello to the user"},
                            {"text": "Say hello to the user."},
                            {"text": "Ensure the final answer is visible"},
                            {"text": "Document known gaps"},
                            {"text": "Provide completion evidence"},
                            {"text": "Mention there are no blockers"},
                            {"text": "Avoid unrelated content"},
                            {"text": "Use a friendly tone"},
                        ]
                    }))
                )
            ]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            items, err = goals.decompose_goal("say hello")

        assert err is None
        assert len(items) <= 5
        assert [i["text"] for i in items].count("Say hello to the user") == 1

    def test_decompose_prompt_includes_scope_control_without_contract_change(self):
        from hermes_cli.goals import build_decompose_system_prompt

        prompt = build_decompose_system_prompt("say hello")

        assert "SCOPE CONTROL" in prompt
        assert "simple" in prompt.lower()
        assert '{"checklist": [{"text": "<item>"}' in prompt

    def test_complex_goal_decomposition_keeps_larger_checklist_budget(self):
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content=json.dumps({
                        "checklist": [
                            {"text": f"Implement and verify complex requirement {i}"}
                            for i in range(1, 15)
                        ]
                    }))
                )
            ]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            items, err = goals.decompose_goal(
                "Audit the goal system, implement fixes, add tests, and verify gateway integration"
            )

        assert err is None
        assert len(items) == 14

    def test_decompose_goal_inlines_referenced_file_context(self, tmp_path, monkeypatch):
        from hermes_cli import goals

        spec = tmp_path / "goal-spec.md"
        spec.write_text(
            "# Spec\n\n"
            "- Add OAuth login.\n"
            "- Preserve existing password login.\n"
            "- Add regression tests for both flows.\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content=json.dumps({
                        "checklist": [
                            {"text": "OAuth login is implemented"},
                            {"text": "Existing password login still works"},
                            {"text": "Regression tests cover both login flows"},
                        ]
                    }))
                )
            ]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            items, err = goals.decompose_goal("implement the following spec in [goal-spec.md]")

        assert err is None
        assert len(items) == 3
        user_prompt = fake_client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "Resolved goal reference context" in user_prompt
        assert "Add OAuth login" in user_prompt
        assert "Preserve existing password login" in user_prompt
        assert '{"checklist": [{"text": "<item>"}' not in user_prompt

    def test_goal_manager_persists_decomposition_reference_audit(self, hermes_home, tmp_path, monkeypatch):
        from hermes_cli.goals import GoalManager, GoalVerdict

        spec = tmp_path / "task.md"
        spec.write_text("Build the export command and test CSV output.\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        mgr = GoalManager(session_id="reference-audit")
        state = mgr.set("fully implement the spec in task.md")

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content=json.dumps({
                        "checklist": [
                            {"text": "Export command is implemented"},
                            {"text": "CSV output is tested"},
                        ]
                    }))
                )
            ]
        )

        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            decision = mgr.evaluate_after_turn("starting")

        assert decision.verdict == GoalVerdict.DECOMPOSE
        assert state.decomposition_reference_context["reference_count"] == 1
        assert state.decomposition_reference_context["resolved_count"] == 1
        assert state.decomposition_reference_context["references"][0]["kind"] == "file"
        event = state.goal_event_log[-1]
        assert event["reference_count"] == 1
        assert event["resolved_reference_count"] == 1

    def test_goal_reference_context_blocks_sensitive_files(self, tmp_path, monkeypatch):
        from hermes_cli.goals import build_goal_reference_context

        secret = tmp_path / ".env"
        secret.write_text("TOKEN=secret\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        context = build_goal_reference_context("use the config in [.env]")

        assert context.references
        assert context.references[0].status == "blocked"
        assert "TOKEN=secret" not in context.render_for_decompose_prompt()

    def test_goal_reference_context_keeps_named_task_discovery_requirements(self):
        from hermes_cli.goals import build_goal_reference_context

        context = build_goal_reference_context("fully implement the spec for [billing-export-task]")

        assert context.references
        assert context.references[0].kind == "named_task"
        assert context.references[0].status == "discovery_required"
        prompt_block = context.render_for_decompose_prompt()
        assert "billing-export-task" in prompt_block
        assert "authoritative source of truth" in prompt_block

    def test_goal_manager_records_decomposition_scope_audit(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalVerdict

        mgr = GoalManager(session_id="scope-audit")
        state = mgr.set("say hello")

        with patch(
            "hermes_cli.goals.decompose_goal",
            return_value=([{"text": "Say hello"}, {"text": "Show final answer"}], None),
        ):
            decision = mgr.evaluate_after_turn("starting")

        assert decision.verdict == GoalVerdict.DECOMPOSE
        assert state.decomposition_scope == "simple"
        assert state.decomposition_item_bounds == {"min_items": 2, "max_items": 5}
        assert "scope: simple" in mgr.status_line()
        event = state.goal_event_log[-1]
        assert event["type"] == "goal_decomposed"
        assert event["scope"] == "simple"
        assert event["item_count"] == 2


class TestGoalVerifierPolicy:
    def test_file_investigation_goal_gets_safe_project_root(self, tmp_path, monkeypatch):
        from hermes_cli.goals import GoalState, build_verifier_policy

        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / ".git").mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "feature.py").write_text("print('ok')\n", encoding="utf-8")
        monkeypatch.chdir(repo / "src")

        state = GoalState(
            goal="audit the repository implementation and tests",
            goal_facets=["code_modification"],
        )

        policy = build_verifier_policy(state, "I changed src/feature.py and ran tests")

        assert policy.allowed_file_roots == [str(repo)]
        assert "file root discovered from project context" in policy.reason

    def test_file_verifier_blocks_sensitive_paths_inside_allowed_root(self, tmp_path):
        from hermes_cli.goals import _judge_read_text_file

        secret = tmp_path / ".env"
        secret.write_text("TOKEN=secret\n", encoding="utf-8")

        result = json.loads(_judge_read_text_file(str(secret), [str(tmp_path)]))

        assert result["ok"] is False
        assert "sensitive path access is blocked" in result["error"]
        assert "TOKEN=secret" not in json.dumps(result)

    def test_decomposition_scope_survives_goalstate_roundtrip(self):
        from hermes_cli.goals import GoalState

        state = GoalState(
            goal="say hello",
            decomposition_scope="simple",
            decomposition_item_bounds={"min_items": 2, "max_items": 5},
        )

        reloaded = GoalState.from_json(state.to_json())

        assert reloaded.decomposition_scope == "simple"
        assert reloaded.decomposition_item_bounds == {"min_items": 2, "max_items": 5}

    def test_goalstate_from_json_normalizes_legacy_invalid_counters_and_reference_audit(self):
        from hermes_cli.goals import DEFAULT_MAX_TURNS, GoalState

        state = GoalState.from_json(json.dumps({
            "goal": "legacy",
            "turns_used": -4,
            "max_turns": -1,
            "redecompose_count": -2,
            "max_redecompositions": 0,
            "judge_calls_made": -9,
            "decomposition_reference_context": {
                "reference_count": 99,
                "resolved_count": 99,
                "references": [
                    {
                        "kind": "file",
                        "reference": ".env",
                        "status": "resolved",
                        "summary": "read TOKEN=secret",
                        "content": "TOKEN=secret",
                        "metadata": {"path": "/home/me/.env", "bytes": 12},
                    }
                ],
            },
        }))

        assert state.turns_used == 0
        assert state.max_turns == DEFAULT_MAX_TURNS
        assert state.redecompose_count == 0
        assert state.max_redecompositions == 3
        assert state.judge_calls_made == 0
        audit = state.decomposition_reference_context
        assert audit["reference_count"] == 1
        assert audit["resolved_count"] == 1
        encoded = json.dumps(audit)
        assert "content" not in encoded
        assert "TOKEN=secret" not in encoded
        assert ".env" not in encoded
        assert "[redacted sensitive path]" in encoded


class TestSplitBullets:
    def test_split_bullets_numeric(self):
        from hermes_cli.goals import _split_bullets

        raw = """
        1. one
        2. two
        3. three
        """
        bullets = _split_bullets(raw)
        assert bullets == ["one", "two", "three"]

    def test_split_bullets_dash(self):
        from hermes_cli.goals import _split_bullets

        raw = """
        - one
        - two
        - three
        """
        bullets = _split_bullets(raw)
        assert bullets == ["one", "two", "three"]


# ──────────────────────────────────────────────────────────────────────
# Judge evidence parsing
# ──────────────────────────────────────────────────────────────────────


class TestEvidenceParsing:
    def test_parse_completion_evidence(self):
        from hermes_cli.goals import parse_completion_evidence

        raw = """
        ## COMPLETION EVIDENCE

        **Checklist items addressed:**
        - [0] thing 1
        - [1] thing 2

        **Artifacts/files/URLs created or changed:**
        - foo.txt

        **Verification performed:**
        - tests passed

        **Known gaps, blockers, or exclusions:**
        - none
        """
        evidence = parse_completion_evidence(raw)
        assert evidence.raw_present is True
        assert len(evidence.checklist_items_addressed) == 2
        assert evidence.declares_no_known_gaps is True


class TestEvidenceParsingErrors:
    def test_parse_completion_evidence_markdown(self):
        from hermes_cli.goals import _parse_completion_evidence_markdown

        raw = "not evidence"
        evidence = _parse_completion_evidence_markdown(raw)
        assert evidence.raw_present is False
        assert evidence.parse_warnings


# ──────────────────────────────────────────────────────────────────────
# Evidence packet building
# ──────────────────────────────────────────────────────────────────────


class TestEvidencePacketBuilding:
    def test_evidence_packet_missing_tool_output(self):
        from hermes_cli.goals import build_judge_evidence_packet

        packet = build_judge_evidence_packet(messages=[], last_response="")
        assert "no tool output" in packet.lower()

    def test_repeated_evidence_fingerprint_ignores_ledger_growth(self):
        from hermes_cli.goals import (
            ADDED_BY_JUDGE,
            ITEM_PENDING,
            ChecklistItem,
            GoalState,
            _add_ledger_entry,
            _stable_evidence_fingerprint,
        )

        state = GoalState(goal="test", decomposed=True)
        state.checklist = [
            ChecklistItem(text="Run tests", status=ITEM_PENDING, added_by=ADDED_BY_JUDGE, added_at=time.time())
        ]
        messages = [
            {
                "role": "tool",
                "name": "terminal",
                "content": "pytest -q\n1 passed in 0.11s\nexit_code: 0",
            }
        ]
        response = "Same evidence every time"

        first = _stable_evidence_fingerprint(response, state=state, messages=messages)
        _add_ledger_entry(
            state,
            evidence_type="test_result",
            source="tool_output",
            summary="pytest -q\n1 passed in 0.11s\nexit_code: 0",
            result_summary="1 passed",
        )
        second = _stable_evidence_fingerprint(response, state=state, messages=messages)

        assert first == second


# ──────────────────────────────────────────────────────────────────────
# Goal evaluation routing
# ──────────────────────────────────────────────────────────────────────


class TestEvaluationRouting:
    def test_route_decision(self, hermes_home):
        from hermes_cli.goals import GoalManager, GoalStatus

        mgr = GoalManager(session_id="route-test")
        mgr.set("perform task")
        assert mgr.state.status == GoalStatus.ACTIVE.value


# ──────────────────────────────────────────────────────────────────────
# Parse failure counter
# ──────────────────────────────────────────────────────────────────────


class TestParseFailureCounter:
    def test_empty_judge_reply_flagged_as_parse_failure(self):
        """End-to-end: judge returns empty content → parse_failed=True."""
        from hermes_cli import goals

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        with patch(
            "agent.auxiliary_client.get_text_auxiliary_client",
            return_value=(fake_client, "judge-model"),
        ):
            verdict, _, parse_failed = goals.judge_goal("goal", "response")
        assert verdict == "continue"
        assert parse_failed is True

    def test_auto_pause_after_three_consecutive_parse_failures(self, hermes_home):
        """N=3 consecutive parse failures → auto-pause with config pointer."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES

        assert DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES == 3
        mgr = GoalManager(session_id="parse-fail-sid-1", default_max_turns=20)
        mgr.set("do a thing")
        mgr.state.decomposed = True

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge returned empty response", True)
        ):
            d1 = mgr.evaluate_after_turn("step 1")
            assert d1["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 1

            d2 = mgr.evaluate_after_turn("step 2")
            assert d2["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 2

            d3 = mgr.evaluate_after_turn("step 3")
            assert d3["should_continue"] is False
            assert d3["status"] == "paused"
            assert mgr.state.consecutive_parse_failures == 3
            # Message points at the config surface so the user can fix it.
            assert "auxiliary" in d3["message"]
            assert "goal_judge" in d3["message"]
            assert "config.yaml" in d3["message"]

    def test_parse_failure_counter_resets_on_good_reply(self, hermes_home):
        """A single good judge reply resets the counter — transient flakes don't pause."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-2", default_max_turns=20)
        mgr.set("another goal")
        mgr.state.decomposed = True

        # Two parse failures…
        with patch.object(
            goals, "judge_goal", return_value=("continue", "not json", True)
        ):
            mgr.evaluate_after_turn("step 1")
            mgr.evaluate_after_turn("step 2")
            assert mgr.state.consecutive_parse_failures == 2

        # …then one clean reply resets the counter.
        with patch.object(
            goals, "judge_goal", return_value=("continue", "making progress", False)
        ):
            d = mgr.evaluate_after_turn("step 3")
            assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0

    def test_parse_failure_counter_not_incremented_by_api_errors(self, hermes_home):
        """API/transport errors must NOT count toward the auto-pause threshold."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id="parse-fail-sid-3", default_max_turns=20)
        mgr.set("goal")
        mgr.state.decomposed = True

        with patch.object(
            goals, "judge_goal", return_value=("continue", "judge error: RuntimeError", False)
        ):
            for _ in range(5):
                d = mgr.evaluate_after_turn("still going")
                assert d["should_continue"] is True
            assert mgr.state.consecutive_parse_failures == 0
            assert mgr.state.status == "active"

    def test_consecutive_parse_failures_persists_across_goalmanager_reloads(
        self, hermes_home
    ):
        """The counter must be durable so cross-session resumes see it."""
        from hermes_cli import goals
        from hermes_cli.goals import GoalManager, load_goal

        mgr = GoalManager(session_id="parse-fail-sid-4", default_max_turns=20)
        mgr.set("persistent goal")
        mgr.state.decomposed = True

        with patch.object(
            goals, "judge_goal", return_value=("continue", "empty", True)
        ):
            mgr.evaluate_after_turn("r")
            mgr.evaluate_after_turn("r")

        reloaded = load_goal("parse-fail-sid-4")
        assert reloaded is not None
        assert reloaded.consecutive_parse_failures == 2


# ──────────────────────────────────────────────────────────────────────
# /subgoal — user-added criteria
# ──────────────────────────────────────────────────────────────────────


class TestGoalStateSubgoalsBackcompat:
    def test_old_state_meta_row_loads_without_subgoals(self):
        """A goal serialized BEFORE the subgoals field existed must
        round-trip with an empty list, not crash."""
        from hermes_cli.goals import GoalState

        legacy = json.dumps({
            "goal": "do a thing",
            "status": "active",
            "turns_used": 2,
            "max_turns": 20,
            "created_at": 1.0,
            "last_turn_at": 2.0,
            "consecutive_parse_failures": 0,
        })
        state = GoalState.from_json(legacy)
        assert state.goal == "do a thing"
        assert state.subgoals == []

    def test_subgoals_round_trip(self):
        from hermes_cli.goals import GoalState
        state = GoalState(goal="g", subgoals=["a", "b", "c"])
        rt = GoalState.from_json(state.to_json())
        assert rt.subgoals == ["a", "b", "c"]


class TestGoalManagerSubgoals:
    def test_add_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-add")
        mgr.set("main goal")
        text = mgr.add_subgoal("  use bullet points  ")
        assert text == "use bullet points"
        assert mgr.state.subgoals == ["use bullet points"]

    def test_add_subgoal_requires_active_goal(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-noactive")
        with pytest.raises(RuntimeError):
            mgr.add_subgoal("oops")

    def test_add_empty_subgoal_rejected(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-empty")
        mgr.set("g")
        with pytest.raises(ValueError):
            mgr.add_subgoal("   ")

    def test_remove_subgoal(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-remove")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")
        mgr.add_subgoal("third")
        removed = mgr.remove_subgoal(2)
        assert removed == "second"
        assert mgr.state.subgoals == ["first", "third"]

    def test_remove_subgoal_out_of_range(self, hermes_home):
        import pytest
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-oob")
        mgr.set("g")
        mgr.add_subgoal("only")
        with pytest.raises(IndexError):
            mgr.remove_subgoal(5)
        with pytest.raises(IndexError):
            mgr.remove_subgoal(0)

    def test_clear_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-clear")
        mgr.set("g")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        prev = mgr.clear_subgoals()
        assert prev == 2
        assert mgr.state.subgoals == []

    def test_subgoals_persist_across_reloads(self, hermes_home):
        """Subgoals stored in SessionDB survive a fresh GoalManager."""
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sub-persist")
        mgr.set("g")
        mgr.add_subgoal("first")
        mgr.add_subgoal("second")

        mgr2 = GoalManager(session_id="sub-persist")
        assert mgr2.state.subgoals == ["first", "second"]


class TestContinuationPromptWithSubgoals:
    def test_empty_subgoals_uses_original_template(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-empty")
        mgr.set("ship the feature")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" not in prompt

    def test_with_subgoals_includes_them(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="cp-with")
        mgr.set("ship the feature")
        mgr.add_subgoal("write tests")
        mgr.add_subgoal("update docs")
        prompt = mgr.next_continuation_prompt()
        assert prompt is not None
        assert "ship the feature" in prompt
        assert "Additional criteria" in prompt
        assert "1. write tests" in prompt
        assert "2. update docs" in prompt


class TestJudgeGoalWithSubgoals:
    def test_judge_uses_subgoals_template_when_provided(self, hermes_home):
        """judge_goal switches templates when subgoals is non-empty.

        We don't actually call the model — we patch the aux client to
        capture the prompt that would be sent.
        """
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "all done"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _FakeResp()

        with patch.object(goals, "get_text_auxiliary_client",
                          return_value=(_FakeClient, "fake-model"), create=True), \
             patch.object(goals, "get_auxiliary_extra_body",
                          return_value=None, create=True), \
             patch("agent.auxiliary_client.get_text_auxiliary_client",
                   return_value=(_FakeClient, "fake-model")), \
             patch("agent.auxiliary_client.get_auxiliary_extra_body",
                   return_value=None):
            verdict, reason, parse_failed = goals.judge_goal(
                "ship the feature",
                "ok shipped",
                subgoals=["write tests", "update docs"],
            )

        # The aux client was called with a prompt that includes the subgoals.
        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" in user_msg
        assert "1. write tests" in user_msg
        assert "2. update docs" in user_msg
        assert "every additional criterion" in user_msg
        assert verdict == "done"

    def test_judge_uses_original_template_when_no_subgoals(self, hermes_home):
        from unittest.mock import patch
        from hermes_cli import goals

        captured = {}

        class _FakeMsg:
            content = '{"done": true, "reason": "ok"}'
        class _FakeChoice:
            message = _FakeMsg()
        class _FakeResp:
            choices = [_FakeChoice()]
        class _FakeClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        captured.update(kwargs)
                        return _FakeResp()

        with patch("agent.auxiliary_client.get_text_auxiliary_client",
                   return_value=(_FakeClient, "fake-model")), \
             patch("agent.auxiliary_client.get_auxiliary_extra_body",
                   return_value=None):
            goals.judge_goal("ship it", "done", subgoals=None)

        sent_messages = captured.get("messages") or []
        user_msg = next((m["content"] for m in sent_messages if m["role"] == "user"), "")
        assert "Additional criteria" not in user_msg
        assert "ship it" in user_msg


class TestStatusLineSubgoalCount:
    def test_status_line_no_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-empty")
        mgr.set("ship it")
        line = mgr.status_line()
        assert "ship it" in line
        assert "subgoal" not in line.lower()

    def test_status_line_with_subgoals(self, hermes_home):
        from hermes_cli.goals import GoalManager
        mgr = GoalManager(session_id="sl-with")
        mgr.set("ship it")
        mgr.add_subgoal("a")
        mgr.add_subgoal("b")
        line = mgr.status_line()
        assert "2 subgoals" in line
