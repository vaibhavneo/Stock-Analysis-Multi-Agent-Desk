#!/usr/bin/env python3
"""
Standalone prompt-evaluation/optimization CLI for the 5 stock_agent LLM
agents, built on the vendored tools/eval/judges.py and
tools/eval/optimizer.py (adapted from Comet Opik - see
ai-ops/docs/sources.md for full attribution).

Run by hand, never from the live request path:

    python3 tools/eval_agents.py --agent technical --trials 5 --variants 3

Requires DEEPSEEK_API_KEY (same resolution as the production pipeline -
this stays DeepSeek-only, no new provider). Every run costs real API
calls: roughly `trials * variants` scoring calls per eval case, plus
`trials` proposal calls (see the cost note in ai-ops's
prompt-optimizer-loop SKILL.md) - use --max-cases and small --trials for
a cheap smoke test before a full run.

This tool only PROPOSES improved system prompts and prints the full
trial history for human review. It never writes to agents/stock_agents.py
- adopting a winning prompt is always a separate, manual edit.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.stock_agents import (
    _build_algo_prompt,
    _build_fundamentals_prompt,
    _build_prediction_prompt,
    _build_social_prompt,
    _build_technical_prompt,
    _call,
    _get_client,
)
from tools.eval.eval_cases import EVAL_CASES
from tools.eval.judges import HALLUCINATION, Verdict, build_custom_judge, run_judge
from tools.eval.optimizer import OptimizationResult, optimize_prompt

BUILDERS = {
    "fundamentals": _build_fundamentals_prompt,
    "technical":    _build_technical_prompt,
    "social":       _build_social_prompt,
    "algo":         _build_algo_prompt,
    "prediction":   _build_prediction_prompt,
}

JUDGES = {
    "fundamentals": build_custom_judge(
        "fundamentals_quality",
        "The analysis must cite or use the ACTUAL numbers given (PE, margins, "
        "growth, ROE, etc.) rather than generic boilerplate that could apply to "
        "any stock, and its stated 1-10 score must be directionally consistent "
        "with its own prose (a bullish-reading analysis should not score itself "
        "2/10, and vice versa).",
    ),
    "technical": build_custom_judge(
        "technical_consistency",
        "Any specific price levels named (entry, target, stop, support, "
        "resistance) must be consistent with the given current price and "
        "indicators - e.g. an entry price should not sit far above the current "
        "price on a 'buy the dip' read, and a bullish call should not pair a "
        "stop-loss above the entry. Trend/momentum claims must match the given "
        "RSI/MACD/ADX values, not contradict them.",
    ),
    "social": HALLUCINATION,  # answer must not invent posts/numbers absent from the given data
    "algo": build_custom_judge(
        "algo_consistency",
        "The interpretation must match the sign and rough magnitude of the "
        "given quantitative signals - e.g. it must not call momentum 'bullish' "
        "when the given momentum figures are negative, or call volatility "
        "'low risk' when the given vol_regime is HIGH and expanding.",
    ),
    "prediction": build_custom_judge(
        "prediction_consistency",
        "The output JSON's action (BUY/SELL/HOLD) and overall tone must be "
        "consistent with what the given FUNDAMENTAL, TECHNICAL, SOCIAL, and "
        "ALGO analyses actually argue - flag it if the action contradicts a "
        "clear majority of the source analyses without any explanation for "
        "the disagreement.",
    ),
}


def _judge_kwargs(agent: str, case: dict, output: str) -> dict:
    """Build the **kwargs run_judge needs for this agent's judge template."""
    if agent == "social":
        context = (
            f"REDDIT: {case['reddit_data']}\nSTOCKTWITS: {case['stocktwits_data']}\n"
            f"WEB: {case['web_forum_data']}"
        )
        return {"context": context, "question": f"Social sentiment for {case['ticker']}", "answer": output}
    if agent == "prediction":
        source = (
            f"FUNDAMENTAL: {case['fundamentals_analysis']}\nTECHNICAL: {case['technical_analysis']}\n"
            f"SOCIAL: {case['social_analysis']}\nALGO: {case['algo_analysis']}"
        )
        return {"input": source, "answer": output}
    # fundamentals / technical / algo: generic build_custom_judge shape
    return {"input": f"ticker={case['ticker']}", "answer": output}


def make_score_fn(agent: str, llm_call_fn):
    builder = BUILDERS[agent]
    judge = JUDGES[agent]

    def score_fn(candidate_prompt: str, case: dict) -> float:
        kwargs = {k: v for k, v in case.items() if k != "notes"}
        _orig_system, user = builder(**kwargs)
        output = llm_call_fn(candidate_prompt, user)
        verdict: Verdict = run_judge(judge, llm_call_fn, **_judge_kwargs(agent, case, output))
        if not verdict.ok:
            return 0.0
        return verdict.score

    return score_fn


_VARIANT_RE = re.compile(r"===\s*VARIANT\s*\d+\s*===", re.IGNORECASE)


def make_propose_fn(llm_call_fn, variants_per_trial: int):
    def propose_fn(current_prompt: str, ranked, n: int):
        worst = ranked[: min(3, len(ranked))]
        failure_notes = "\n".join(
            f"- case {case.get('ticker', '?')}: scored {score:.2f} — {case.get('notes', '')}"
            for case, score in worst
        )
        instruction = (
            f"Current system prompt for a stock-analysis LLM agent:\n\n{current_prompt}\n\n"
            f"It scores worst on these cases:\n{failure_notes}\n\n"
            f"Write {n} improved rewrites of this system prompt that would address these "
            f"specific weaknesses, without changing the agent's core role or output format. "
            f"Separate each rewrite with a line reading exactly '=== VARIANT k ===' "
            f"(k = 1..{n}). Output only the {n} variants, nothing else."
        )
        raw = llm_call_fn("You are an expert prompt engineer.", instruction)
        parts = [p.strip() for p in _VARIANT_RE.split(raw) if p.strip()]
        if not parts:
            return [current_prompt] * n
        while len(parts) < n:
            parts.append(parts[-1])
        return parts[:n]

    return propose_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--max-cases", type=int, default=None, help="limit eval cases for a cheap smoke test")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("No API key found. Set DEEPSEEK_API_KEY in .env or environment.", file=sys.stderr)
        sys.exit(1)

    client = _get_client(api_key)

    def llm_call_fn(system_prompt: str, user_prompt: str) -> str:
        return _call(client, system_prompt, user_prompt)

    cases = EVAL_CASES[args.agent]
    if args.max_cases:
        cases = cases[: args.max_cases]

    first_kwargs = {k: v for k, v in cases[0].items() if k != "notes"}
    base_prompt = BUILDERS[args.agent](**first_kwargs)[0]

    print(f"Optimizing '{args.agent}' system prompt against {len(cases)} eval cases "
          f"({args.trials} trials x {args.variants} variants)...\n")

    result: OptimizationResult = optimize_prompt(
        base_prompt=base_prompt,
        eval_cases=cases,
        score_fn=make_score_fn(args.agent, llm_call_fn),
        propose_fn=make_propose_fn(llm_call_fn, args.variants),
        max_trials=args.trials,
        variants_per_trial=args.variants,
    )

    print("=" * 70)
    for trial in result.history:
        print(f"trial {trial.trial_num}  avg={trial.avg_score:.2f}  scores={[round(s, 2) for s in trial.per_case_scores]}")
    print("=" * 70)
    print(f"starting score: {result.starting_score:.2f}")
    print(f"best score:     {result.best_score:.2f}  (improved={result.improved})")
    print("\nBEST PROMPT FOUND (nothing has been changed in agents/stock_agents.py):\n")
    print(result.best_prompt)


if __name__ == "__main__":
    main()
