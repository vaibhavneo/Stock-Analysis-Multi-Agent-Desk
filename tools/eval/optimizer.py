"""
Vendored copy of ai-ops/skills/prompt-optimizer-loop/optimizer.py (from
the sibling `ai-ops` repo, vaibhavneo/ai-ops) - copied rather than
imported across repos since stock_agent deploys standalone via
gunicorn/Railway and may not have ai-ops checked out alongside it. This
is a copy-once snapshot, not synced automatically; re-copy manually if
ai-ops's version changes and you want the update here too. Dev-only
tooling (tools/eval/), not imported by production code.

Lightweight prompt-optimization loop.

Pattern adapted from Comet Opik Agent Optimizer's MetaPromptOptimizer
(https://github.com/comet-ml/opik, sdks/opik_optimizer/, Apache License
2.0, Copyright (c) Comet ML, Inc.): evaluate a prompt against real
examples, ask an LLM to propose improved variants that target the
specific failures, evaluate the variants, keep whichever scores best,
repeat. Code below is written from scratch for this repo - no Opik
source was copied - and drops the upstream package's LiteLLM / DEAP /
GEPA / HF-datasets dependencies in favor of caller-supplied plain
functions, since this is sized for one developer's four apps rather
than a hosted optimization service. See docs/sources.md for the full
attribution.

This proposes; it never applies anything to production code by itself.
Review optimize_prompt()'s returned history and pick a winner yourself -
the same "scanning is automatic, applying is always manual" rule this
repo already follows for agent-improvement-loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# (candidate_prompt, eval_case) -> score, any consistent scale (e.g. 0..1 or 0..10)
ScoreFn = Callable[[str, Any], float]

# (current_prompt, [(eval_case, score), ...] worst-scoring first, n_variants) -> candidate prompt strings
ProposeFn = Callable[[str, list, int], list]


@dataclass
class Trial:
    trial_num: int
    prompt: str
    avg_score: float
    per_case_scores: list


@dataclass
class OptimizationResult:
    best_prompt: str
    best_score: float
    starting_score: float
    history: list = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.best_score > self.starting_score


def _evaluate(prompt: str, eval_cases: Sequence[Any], score_fn: ScoreFn) -> list:
    return [score_fn(prompt, case) for case in eval_cases]


def optimize_prompt(
    base_prompt: str,
    eval_cases: Sequence[Any],
    score_fn: ScoreFn,
    propose_fn: ProposeFn,
    max_trials: int = 5,
    variants_per_trial: int = 3,
) -> OptimizationResult:
    """Iteratively improve base_prompt against eval_cases.

    score_fn scores one (prompt, eval_case) pair - wrap llm_judge.run_judge
    around it (see the sibling llm-judge skill), or use exact-match / any
    other scorer; this loop doesn't care what "score" means as long as
    higher is better and the scale is consistent across calls.

    propose_fn receives the current best prompt plus (eval_case, score)
    pairs sorted worst-first, and must return `variants_per_trial`
    candidate prompt strings - typically by asking an LLM to rewrite the
    prompt to fix whatever the worst-scoring cases got wrong. This is the
    one call site where you plug in your own LLM client; nothing here
    calls an LLM directly.

    Returns every trial's prompt and score, not just the winner, so you
    can see *why* the loop landed where it did before deciding whether to
    adopt the result.
    """
    if not eval_cases:
        raise ValueError("optimize_prompt needs at least one eval case")

    starting_scores = _evaluate(base_prompt, eval_cases, score_fn)
    starting_avg = sum(starting_scores) / len(starting_scores)

    best_prompt, best_avg = base_prompt, starting_avg
    history = [Trial(trial_num=0, prompt=base_prompt, avg_score=starting_avg, per_case_scores=starting_scores)]

    current_prompt, current_scores = base_prompt, starting_scores

    for trial_num in range(1, max_trials + 1):
        ranked = sorted(zip(eval_cases, current_scores), key=lambda pair: pair[1])
        candidates = propose_fn(current_prompt, ranked, variants_per_trial)

        trial_best_prompt = current_prompt
        trial_best_avg = sum(current_scores) / len(current_scores)
        trial_best_scores = current_scores

        for candidate in candidates:
            scores = _evaluate(candidate, eval_cases, score_fn)
            avg = sum(scores) / len(scores)
            history.append(Trial(trial_num=trial_num, prompt=candidate, avg_score=avg, per_case_scores=scores))
            if avg > trial_best_avg:
                trial_best_prompt, trial_best_avg, trial_best_scores = candidate, avg, scores

        current_prompt, current_scores = trial_best_prompt, trial_best_scores
        if trial_best_avg > best_avg:
            best_prompt, best_avg = trial_best_prompt, trial_best_avg

    return OptimizationResult(best_prompt=best_prompt, best_score=best_avg, starting_score=starting_avg, history=history)
