from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

import httpx

from .config import Settings
from .fetch import fetch_post, parse_bluesky_url, build_at_uri


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_bluesky",
            "description": "Search Bluesky for posts about a topic or entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (1-10)"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_post",
            "description": "Fetch the text of a Bluesky post by its bsky.app URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "bsky.app post URL"},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the final explanation bullets once enough context has been gathered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bullets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3-5 explanation bullets, each with at least one [N] citation.",
                    },
                    "sources": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "url": {"type": "string"},
                            },
                            "required": ["id", "url"],
                            "additionalProperties": False,
                        },
                        "description": "Sources cited in bullets.",
                    },
                },
                "required": ["bullets", "sources"],
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_search_bluesky(query: str, limit: int = 5) -> str:
    with httpx.Client(timeout=20) as client:
        resp = client.get(
            "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts",
            params={"q": query, "limit": min(limit, 10)},
        )
        resp.raise_for_status()
        posts = resp.json().get("posts", [])

    results = []
    for post in posts:
        uri = post.get("uri", "")
        parts = uri.removeprefix("at://").split("/")
        url = f"https://bsky.app/profile/{parts[0]}/post/{parts[2]}" if len(parts) == 3 else uri
        text = post.get("record", {}).get("text", "")
        handle = post.get("author", {}).get("handle", "")
        if text:
            results.append({"url": url, "handle": handle, "text": text})

    return json.dumps(results, ensure_ascii=False)


def _tool_fetch_post(url: str) -> str:
    try:
        profile, rkey = parse_bluesky_url(url)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    uri = build_at_uri(profile, rkey)
    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                "https://api.bsky.app/xrpc/app.bsky.feed.getPostThread",
                params={"uri": uri},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return json.dumps({"error": str(e)})

    text = data.get("thread", {}).get("post", {}).get("record", {}).get("text", "")
    return json.dumps({"url": url, "text": text}, ensure_ascii=False)


def _dispatch_tool(name: str, arguments: Dict[str, Any]) -> str:
    if name == "search_bluesky":
        return _tool_search_bluesky(**arguments)
    if name == "fetch_post":
        return _tool_fetch_post(**arguments)
    return json.dumps({"error": f"Unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Self-critique
# ---------------------------------------------------------------------------

CRITIQUE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "critique",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "fail"]},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
            "additionalProperties": False,
        },
    },
}

CRITIQUE_PROMPT = """You are a quality reviewer for Bluesky post explanations.

Given a post and explanation bullets, decide if the bullets pass or fail.

PASS if ALL of:
- 3 to 5 bullets
- Each bullet explains background context (not a summary of the post)
- Each bullet has at least one [N] citation
- Bullets are specific — not generic observations anyone could make

FAIL if ANY of:
- Fewer than 3 bullets or more than 5
- Any bullet summarises the post instead of explaining background
- Any bullet has no citation
- Bullets are vague or could apply to any post on the topic"""


def _critique_bullets(bullets: List[str], post_text: str, settings: Settings) -> Dict[str, str]:
    """Ask the LLM to evaluate bullet quality. Returns {verdict, reason}."""
    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))

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
                    {"role": "system", "content": CRITIQUE_PROMPT},
                    {"role": "user", "content": f"POST:\n{post_text}\n\nBULLETS:\n{numbered}"},
                ],
                "response_format": CRITIQUE_SCHEMA,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()

    return json.loads(resp.json()["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Bluesky post explainer. Your job is to explain the background
context of a Bluesky post for someone outside the author's bubble.

You have two tools:
- search_bluesky: search for posts about a topic or entity
- fetch_post: fetch a specific post's text

Strategy:
1. Read the post text provided by the user.
2. Identify concepts that need external context (jargon, people, projects, memes).
3. Search Bluesky for each concept to gather context.
4. Fetch specific posts if you need more detail.
5. Once you have enough context, call finish() with 3-5 bullets.

Each bullet must:
- Explain background, origin, or significance of a concept.
- Include at least one [N] citation referencing a source by its index (1-based, in the order you encountered them).
- Help a smart outsider understand the post.

Do NOT summarize the post. Explain what it assumes the reader already knows."""


def explain_post(url: str) -> Dict[str, object]:
    """Agentic loop: LLM drives search + fetch until critique passes."""
    settings = Settings.from_env()

    post = fetch_post(url, settings)
    if not post.text:
        raise ValueError("Fetched post has empty text; cannot explain.")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explain this Bluesky post:\n\n{post.text}"},
    ]

    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    for iteration in range(1, 11):
        print(f"\n[iter {iteration}]", file=sys.stderr)

        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{settings.openai_api_base}/chat/completions",
                headers=headers,
                json={
                    "model": settings.openai_chat_model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": "required",
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()

        response_message = resp.json()["choices"][0]["message"]
        messages.append(response_message)

        tool_calls = response_message.get("tool_calls") or []
        if not tool_calls:
            break

        finish_tc = next((tc for tc in tool_calls if tc["function"]["name"] == "finish"), None)

        if finish_tc:
            args = json.loads(finish_tc["function"]["arguments"])
            bullets, sources = args["bullets"], args["sources"]

            critique = _critique_bullets(bullets, post.text, settings)
            print(f"  → finish() critique: {critique['verdict']} — {critique['reason']}", file=sys.stderr)

            if critique["verdict"] == "pass":
                print(f"  ✓ done in {iteration} iteration(s)", file=sys.stderr)
                return {"bullets": bullets, "sources": sources, "post_text": post.text}

            # Provide required tool results for all calls in this batch
            for tc in tool_calls:
                content = (
                    json.dumps({"status": "rejected", "reason": critique["reason"]})
                    if tc["id"] == finish_tc["id"]
                    else _dispatch_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                )
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})

            messages.append({
                "role": "user",
                "content": f"Quality check failed: {critique['reason']}\n\nSearch for more specific context and try finish() again.",
            })
        else:
            # No finish call — dispatch all tools and continue
            for tc in tool_calls:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                display = {k: (f"{str(v)[:60]}…" if len(str(v)) > 60 else v) for k, v in args.items()}
                print(f"  → {name}({display})", file=sys.stderr)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _dispatch_tool(name, args),
                })

    raise RuntimeError("Agent did not produce passing bullets within 10 iterations.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_human_readable(result: Dict[str, object]) -> None:
    print("Explanation bullets:\n")
    for b in result.get("bullets") or []:
        print(f"- {b}")

    sources = result.get("sources") or []
    if sources:
        print("\nSources:")
        for src in sources:
            print(f"[{src.get('id')}] {src.get('url')}")


def main(argv: List[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("Usage: python -m agent.main <bluesky_post_url>")
        raise SystemExit(1)

    result = explain_post(argv[0])
    _print_human_readable(result)


if __name__ == "__main__":
    main()
