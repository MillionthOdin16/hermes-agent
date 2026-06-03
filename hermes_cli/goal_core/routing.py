"""Deterministic pre-judge routing and work-phase inference."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hermes_cli.goal_core.types import GoalStatus


ROUTE_CALL_JUDGE = "call_judge"
ROUTE_SKIP_NO_EVIDENCE = "skip_no_actionable_evidence"
ROUTE_SKIP_INTENT_ONLY = "skip_intent_only"
ROUTE_SKIP_BLOCKED_USER = "skip_blocked_user_input"
ROUTE_SKIP_BLOCKED_TOOLING = "skip_blocked_tooling"
ROUTE_SKIP_GOAL_INACTIVE = "skip_goal_inactive"

PHASE_EXPLORE = "explore"
PHASE_PLAN = "plan"
PHASE_IMPLEMENT = "implement"
PHASE_VERIFY = "verify"
PHASE_REVIEW = "review"
PHASE_BLOCKED = "blocked"
PHASE_DONE = "done"

_EVIDENCE_VERBS = re.compile(
    r"\b(?:created|wrote|modified|updated|implemented|ran|tested|verified|"
    r"deployed|generated|saved|fixed|produced|built|installed|configured|"
    r"refactored|migrated|completed|delivered|shipped|launched|published|"
    r"compiled|executed|passed|merged|resolved|patched|applied|applied)\b",
    re.IGNORECASE,
)

_INTENT_PATTERNS = [
    re.compile(r"\bI (?:will|shall|am going to|plan to|intend to|need to)\b", re.IGNORECASE),
    re.compile(r"\b(?:next|then|after that|going forward|moving on)\b.*\bI(?:'ll| will)\b", re.IGNORECASE),
    re.compile(r"\blet me (?:now |first )?(?:inspect|check|examine|review|look at)\b", re.IGNORECASE),
]

_NOOP_PATTERNS = [
    re.compile(r"^(?:ok|okay|got it|thanks|thank you|acknowledged|noted|i understand|understood)$", re.IGNORECASE),
    re.compile(r"^no (?:changes|action|progress) (?:yet|made|taken)(?:\s*\.?\s*)$", re.IGNORECASE),
    re.compile(
        r"^i (?:have )?(?:not|haven't) (?:started|begun|done|made|taken) (?:yet|anything)(?:\s*\.?\s*)$",
        re.IGNORECASE,
    ),
]

_BLOCKED_USER_PATTERNS = [
    re.compile(r"\bI need (?:the|your|a) .*(?:before|to|in order to)\b", re.IGNORECASE),
    re.compile(r"\b(?:please|can you) (?:provide|share|give|send|tell)\b", re.IGNORECASE),
    re.compile(r"\bwhat (?:is|are) the (?:url|path|repo|address|credentials?)\b", re.IGNORECASE),
    re.compile(r"\bI (?:don't|do not) (?:have|know) (?:the|your|where)\b", re.IGNORECASE),
]

_BLOCKED_TOOLING_PATTERNS = [
    re.compile(r"\b(?:tool|access|permission|command) (?:failed|denied|unavailable|not available)\b", re.IGNORECASE),
    re.compile(r"\bI cannot (?:access|run|execute|use)\b", re.IGNORECASE),
    re.compile(r"\b(?:missing|no) (?:access|permission|tool|credentials)\b", re.IGNORECASE),
]

_EXPLORE_PATTERNS = [
    re.compile(
        r"\b(?:inspect(?:ed|ing)?|examin(?:ed|ing)|read(?:ing)?|review(?:ed|ing)?|analyz(?:ed|ing)|"
        r"look(?:ed|ing)? (?:at|into)|check(?:ed|ing)?|investigat(?:ed|ing)|explor(?:ed|ing)|"
        r"scann(?:ed|ing)|list(?:ed|ing)?|search(?:ed|ing)?|found|discover(?:ed|ing)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:file(?:s)?|director(?:y|ies)|code(?:base)?|structur(?:e|ed)|architecture|config(?:uration)?)\b", re.IGNORECASE),
]

_PLAN_PATTERNS = [
    re.compile(r"\b(?:will|plan(?:ning)?|going to|next step|intend|should|need to|must|let me)\b", re.IGNORECASE),
]

_VERIFY_PATTERNS = [
    re.compile(
        r"\b(?:test(?:s|ed|ing)?|verif(?:y|ied|ying)|pass(?:ed|ing)?|fail(?:ed|ing)?|assert|"
        r"check(?:ed|ing)?|validat(?:ed|ing)|confirm(?:ed|ing)?|lint(?:ed|ing)?|lint)\b",
        re.IGNORECASE,
    ),
]

_COMPLETION_CLAIM_PATTERNS = [
    re.compile(r"\bthe goal is complete\b", re.IGNORECASE),
    re.compile(r"\bthis is complete\b", re.IGNORECASE),
    re.compile(r"\ball items (?:are|is) complete\b", re.IGNORECASE),
    re.compile(r"\bi have completed\b", re.IGNORECASE),
    re.compile(r"\bcompleted the task\b", re.IGNORECASE),
    re.compile(r"\bnothing remains\b", re.IGNORECASE),
    re.compile(r"\bready for final review\b", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*(?:done|complete)\s*[.!]?\s*$", re.IGNORECASE),
]

_COMPLETION_NEGATION_PATTERNS = [
    re.compile(r"\bnot\s+(?:complete|done)\b", re.IGNORECASE),
    re.compile(r"\bincomplete\b", re.IGNORECASE),
    re.compile(r"\bremaining work\b", re.IGNORECASE),
    re.compile(r"\bnot done\b", re.IGNORECASE),
    re.compile(r"\bi am not done\b", re.IGNORECASE),
    re.compile(r"\bthis still needs\b", re.IGNORECASE),
    re.compile(r"\bblocked\b", re.IGNORECASE),
]


@dataclass
class GoalEvaluationRoute:
    """Deterministic pre-judge routing decision."""

    route: str = ROUTE_CALL_JUDGE
    should_call_judge: bool = True
    should_continue: bool = True
    reason: str = ""
    evidence_present: bool = False
    completion_claim_present: bool = False
    blocked: bool = False
    phase: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "route": self.route,
            "should_call_judge": self.should_call_judge,
            "should_continue": self.should_continue,
            "reason": self.reason,
            "evidence_present": self.evidence_present,
            "completion_claim_present": self.completion_claim_present,
            "blocked": self.blocked,
        }
        if self.phase:
            d["phase"] = self.phase
        return d


def infer_goal_work_phase(
    state: Any,
    last_response: str,
    evidence: Optional[Any] = None,
) -> str:
    """Infer the current work phase from goal state and agent response."""
    if not state or getattr(state, "status", None) != GoalStatus.ACTIVE.value:
        return PHASE_DONE

    checklist = getattr(state, "checklist", []) or []
    if checklist and all(getattr(it, "status", "") in ("completed", "impossible") for it in checklist):
        return PHASE_DONE

    response = (last_response or "").strip()
    if not response:
        return PHASE_EXPLORE

    for pat in _BLOCKED_USER_PATTERNS:
        if pat.search(response):
            return PHASE_BLOCKED
    for pat in _BLOCKED_TOOLING_PATTERNS:
        if pat.search(response):
            return PHASE_BLOCKED

    if evidence is not None and getattr(evidence, "raw_present", False) and getattr(evidence, "declares_completion", False):
        return PHASE_REVIEW

    if any(p.search(response) for p in _VERIFY_PATTERNS):
        return PHASE_VERIFY
    if _EVIDENCE_VERBS.search(response):
        return PHASE_IMPLEMENT

    explore_hits = sum(1 for p in _EXPLORE_PATTERNS if p.search(response))
    if explore_hits >= 2:
        return PHASE_EXPLORE

    for pat in _PLAN_PATTERNS:
        if pat.search(response):
            return PHASE_PLAN

    if re.search(r"https?://", response) or re.search(r"(?:^|\s)(?:/|~/|[\w]+/)[\w][\w./@-]*\.\w+", response):
        return PHASE_EXPLORE

    return PHASE_EXPLORE


def route_goal_evaluation(
    state: Any,
    last_response: str,
    evidence: Optional[Any] = None,
    candidates: Optional[Dict[str, List[str]]] = None,
) -> GoalEvaluationRoute:
    """Deterministic pre-judge routing: decide if a judge call is needed."""
    phase = infer_goal_work_phase(state, last_response, evidence=evidence)

    if getattr(state, "status", None) != GoalStatus.ACTIVE.value:
        return GoalEvaluationRoute(
            route=ROUTE_SKIP_GOAL_INACTIVE,
            should_call_judge=False,
            should_continue=False,
            reason="goal is not active",
            phase=PHASE_DONE,
        )

    if not last_response or not last_response.strip():
        return GoalEvaluationRoute(
            route=ROUTE_SKIP_NO_EVIDENCE,
            should_call_judge=False,
            should_continue=True,
            reason="empty response — nothing to evaluate",
            phase=phase,
        )

    response = last_response.strip()

    if len(response) < 80:
        for pat in _NOOP_PATTERNS:
            if pat.search(response):
                has_evidence_signals = bool(
                    _EVIDENCE_VERBS.search(response)
                    or re.search(r"https?://", response)
                    or re.search(r"(?:^|\s)(?:/|~/|[\w]+/)[\w][\w./@-]*\.\w+", response)
                    or re.search(
                        r"\b\d+\s+(?:items?|files?|tests?|lines?|entries?|records?|results?)\b",
                        response,
                        re.IGNORECASE,
                    )
                )
                if not has_evidence_signals:
                    return GoalEvaluationRoute(
                        route=ROUTE_SKIP_NO_EVIDENCE,
                        should_call_judge=False,
                        should_continue=True,
                        reason="non-evidentiary acknowledgement — no concrete action reported",
                        phase=phase,
                    )

    if evidence is not None and getattr(evidence, "raw_present", False):
        has_content = bool(
            getattr(evidence, "checklist_items_addressed", None)
            or getattr(evidence, "artifacts", None)
            or getattr(evidence, "urls", None)
            or getattr(evidence, "files", None)
            or getattr(evidence, "verification_performed", None)
            or getattr(evidence, "counts_or_reconciliations", None)
        )
        if has_content or getattr(evidence, "declares_completion", False):
            return GoalEvaluationRoute(
                route=ROUTE_CALL_JUDGE,
                should_call_judge=True,
                reason="structured COMPLETION EVIDENCE present",
                evidence_present=True,
                completion_claim_present=bool(getattr(evidence, "declares_completion", False)),
                phase=phase,
            )

    if candidates:
        has_candidates = any(candidates.get(k) for k in ("urls", "files", "counts", "artifacts"))
        if has_candidates:
            return GoalEvaluationRoute(
                route=ROUTE_CALL_JUDGE,
                should_call_judge=True,
                reason="verifier candidates extracted from evidence",
                evidence_present=True,
                phase=phase,
            )

    for pat in _COMPLETION_CLAIM_PATTERNS:
        if pat.search(response):
            return GoalEvaluationRoute(
                route=ROUTE_CALL_JUDGE,
                should_call_judge=True,
                reason="completion claim detected in response",
                completion_claim_present=True,
                phase=phase,
            )

    if _EVIDENCE_VERBS.search(response):
        return GoalEvaluationRoute(
            route=ROUTE_CALL_JUDGE,
            should_call_judge=True,
            reason="response contains evidence verbs (concrete action reported)",
            evidence_present=True,
            phase=phase,
        )

    for pat in _BLOCKED_USER_PATTERNS:
        if pat.search(response):
            return GoalEvaluationRoute(
                route=ROUTE_SKIP_BLOCKED_USER,
                should_call_judge=False,
                should_continue=False,
                reason="agent appears to need user input to proceed",
                blocked=True,
                phase=PHASE_BLOCKED,
            )

    for pat in _BLOCKED_TOOLING_PATTERNS:
        if pat.search(response):
            return GoalEvaluationRoute(
                route=ROUTE_SKIP_BLOCKED_TOOLING,
                should_call_judge=False,
                should_continue=True,
                reason="agent reports tooling/access failure",
                blocked=True,
                phase=PHASE_BLOCKED,
            )

    if re.search(r"https?://", response):
        return GoalEvaluationRoute(
            route=ROUTE_CALL_JUDGE,
            should_call_judge=True,
            reason="response contains URL",
            evidence_present=True,
            phase=phase,
        )
    if re.search(r"(?:^|\s)(?:/|~/|[\w]+/)[\w][\w./@-]*\.[\w]+", response):
        return GoalEvaluationRoute(
            route=ROUTE_CALL_JUDGE,
            should_call_judge=True,
            reason="response contains file path",
            evidence_present=True,
            phase=phase,
        )
    if re.search(r"\b\d+\s+(?:items?|files?|tests?|lines?|entries?|records?|results?)\b", response, re.IGNORECASE):
        return GoalEvaluationRoute(
            route=ROUTE_CALL_JUDGE,
            should_call_judge=True,
            reason="response contains count/reconciliation",
            evidence_present=True,
            phase=phase,
        )

    for pat in _INTENT_PATTERNS:
        if pat.search(response):
            return GoalEvaluationRoute(
                route=ROUTE_SKIP_INTENT_ONLY,
                should_call_judge=False,
                should_continue=True,
                reason="response describes intent, not completed action",
                phase=phase,
            )

    return GoalEvaluationRoute(
        route=ROUTE_CALL_JUDGE,
        should_call_judge=True,
        reason="no safe skip condition matched — calling judge",
        phase=phase,
    )

