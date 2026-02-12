"""Eval harness for the Bluesky post explainer agent.

Runs each entry in evals/dataset.json through explain_post() and scores the result
using an LLM judge that compares the produced bullets against the gold_summary.

Output: per-sample verdict + aggregate pass rate printed to stdout.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx

# Resolve dataset path relative to this file
DATASET_PATH = Path(__file__).parent / "dataset.json"

# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "eval_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "score": {"type": "integer"},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "score", "reason"],
            "additionalProperties": False,
        },
    },
}

JUDGE_PROMPT = """You are evaluating a Bluesky post explainer agent.

Given:
- POST TEXT: the original Bluesky post
- GOLD SUMMARY: a reference description of what the post is about / what context is needed
- AGENT BULLETS: the agent's explanation bullets with citations

Score the agent bullets from 0-10 and give a verdict:
PASS (score >= 6) if:
- Bullets collectively cover the key background context indicated by the gold summary
- Bullets explain WHY concepts matter, not just what they are
- Citations are present

FAIL (score < 6) if:
- Bullets miss the main point of the gold summary
- Bullets merely summarize the post instead of explaining background
- No citations

Be strict. The agent should earn its pass."""


def _judge(
    post_text: str,
    gold_summary: str,
    bullets: List[str],
    settings: Any,
) -> Dict[str, Any]:
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))
    user_content = (
        f"POST TEXT:\n{post_text}\n\n"
        f"GOLD SUMMARY:\n{gold_summary}\n\n"
        f"AGENT BULLETS:\n{numbered}"
    )
    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{settings.openai_api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.openai_chat_model,
                "messages": [
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "response_format": JUDGE_SCHEMA,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
    return json.loads(resp.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval() -> None:
    # Import here so the harness can be run via run.sh
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agent.config import Settings
    from agent.main import explain_post

    settings = Settings.from_env()

    dataset: List[Dict[str, Any]] = json.loads(DATASET_PATH.read_text())
    total = len(dataset)
    passed = 0
    results = []

    print(f"Running eval on {total} samples...\n{'='*60}")

    for sample in dataset:
        sid = sample["id"]
        category = sample["category"]
        url = sample["url"]
        gold = sample["gold_summary"]

        print(f"\n[{sid}] category={category}")
        print(f"  url: {url}")

        try:
            t0 = time.time()
            result = explain_post(url)
            elapsed = time.time() - t0

            bullets = result["bullets"]
            verdict_data = _judge(result["post_text"], gold, bullets, settings)
            verdict = verdict_data["verdict"]
            score = verdict_data["score"]
            reason = verdict_data["reason"]

            if verdict == "pass":
                passed += 1
                status = "✓ PASS"
            else:
                status = "✗ FAIL"

            print(f"  {status} (score={score}/10, {elapsed:.1f}s)")
            print(f"  judge: {reason}")
            print(f"  bullets ({len(bullets)}):")
            for b in bullets:
                print(f"    - {b[:120]}")

            results.append({
                "id": sid,
                "category": category,
                "verdict": verdict,
                "score": score,
                "elapsed_s": round(elapsed, 1),
                "judge_reason": reason,
                "bullets": bullets,
            })

        except Exception as exc:
            print(f"  ERROR: {exc}")
            results.append({
                "id": sid,
                "category": category,
                "verdict": "error",
                "score": 0,
                "error": str(exc),
            })

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} passed ({100*passed//total}%)")
    print(f"{'='*60}")
    for r in results:
        icon = "✓" if r["verdict"] == "pass" else ("✗" if r["verdict"] == "fail" else "!")
        score_str = f"score={r.get('score', '?')}/10"
        print(f"  {icon} [{r['id']}] {r['category']:20s} {score_str}")

    # Write JSON results
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDetailed results written to {out_path}")


if __name__ == "__main__":
    run_eval()
