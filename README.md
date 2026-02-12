# Bluesky Post Explainer

An agentic AI tool that explains Bluesky posts for people outside the author's bubble. Given a post URL, it searches Bluesky for context, fetches relevant posts, and returns 3–5 explanation bullets with citations — iterating until a self-critique pass is achieved.

---

## Setup

**Requirements:** Python 3.11+, an OpenAI API key.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
# then edit .env:
OPENAI_API_KEY=sk-...
```

Optional overrides (all have defaults):

```
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_CHAT_MODEL=gpt-4.1-mini
BLUESKY_APPVIEW_BASE=https://api.bsky.app
```

---

## Usage

### Explain a post

```bash
./run.sh agent "https://bsky.app/profile/handle.bsky.social/post/rkey"
```

Progress is streamed to stderr (iteration count, tool calls, critique verdict); the final bullets and sources are printed to stdout.

### Run the eval harness

```bash
./run.sh eval
```

Runs all 10 posts in `evals/dataset.json` through the agent and scores them with an LLM judge. Results are written to `evals/results.json`.

---

## Architecture

### Agentic loop (`agent/main.py`)

The agent is a tool-calling loop directly against the OpenAI Chat Completions API (`gpt-4.1-mini`). No LangChain, no framework — just `httpx` and the raw API.

1. The target post is fetched via the Bluesky AT Protocol API (`app.bsky.feed.getPostThread`).
2. The LLM receives the post text and a system prompt explaining its task.
3. The LLM issues tool calls (`search_bluesky`, `fetch_post`) to gather context.
4. Results are returned as `role: tool` messages; the loop continues.
5. When the LLM has gathered enough context, it calls `finish()` with 3–5 bullets and sources.
6. A **self-critique** step (`_critique_bullets`) sends the bullets to a second LLM call for evaluation using `response_format: json_schema` (structured output).
7. If the critique **fails**, the reason is injected as a feedback message and the loop resumes.
8. If the critique **passes**, the result is returned. A 10-iteration safety cap prevents infinite loops.

```
fetch post
    │
    ▼
LLM ──► search_bluesky()  ──► results
    │
    ├──► fetch_post()      ──► post text
    │
    └──► finish()
              │
              ▼
         self-critique (separate LLM call, structured output)
              │
         pass? ──► return bullets + sources
              │
         fail? ──► inject feedback ──► LLM (next iteration)
```

### Tools

| Tool | Description |
|------|-------------|
| `search_bluesky` | `GET api.bsky.app/xrpc/app.bsky.feed.searchPosts` — returns up to 10 posts with url, handle, text |
| `fetch_post` | `GET api.bsky.app/xrpc/app.bsky.feed.getPostThread` — returns full text for a specific bsky.app URL |
| `finish` | Structured output: `bullets` (array of strings) + `sources` (array of `{id, url}`) |

`tool_choice: "required"` ensures the LLM always calls a tool, preventing free-form text responses mid-loop.

### Self-critique (`_critique_bullets`)

A separate LLM call at `temperature=0.0` with a strict JSON schema (`{verdict: "pass"|"fail", reason: string}`). Criteria:

- 3–5 bullets
- Each bullet explains **background context**, not a summary of the post
- Each bullet has at least one `[N]` citation
- Bullets are specific, not generic observations anyone could make

On failure, the critique's `reason` is injected back into the conversation as a user message and the loop continues.

### Post fetching (`agent/fetch.py`)

`fetch_post()` parses the bsky.app URL (`profile/<handle-or-did>/post/<rkey>`), builds an `at://` URI, and hits `getPostThread`. Returns a `Post` dataclass with `text`, `author_handle`, `created_at`, `external_links` (from richtext facets), and `images` (from embed views).

### Eval harness (`evals/run_eval.py`)

- Loads `evals/dataset.json` — 10 real Bluesky posts across 9 categories
- Runs each through `explain_post()`
- Scores each output with an LLM judge that compares bullets against a `gold_summary`
- Pass threshold: score ≥ 6/10
- Prints per-sample verdict + aggregate pass rate; writes full results to `evals/results.json`

### Eval dataset categories

| Category | # | Description |
|----------|---|-------------|
| `compression` | 2 | Dense technical/factual claims needing unpacking (quantum computing, vibe coding) |
| `slang_meme` | 1 | Internet culture references (Dril) |
| `thread_dependent` | 1 | Post that assumes prior thread context (monetary policy) |
| `hallucination_trap` | 1 | Crypto jargon the agent should explain accurately (rug pull) |
| `tech_jargon` | 1 | Platform decay vocabulary (enshittification) |
| `tech_context` | 1 | Broader tech ecosystem (Fediverse, ActivityPub) |
| `ideology_debate` | 1 | Economic movement context (degrowth) |
| `ai_jargon` | 1 | AI research terminology (RAG, hallucination, AUROC) |
| `ai_community` | 1 | Open-source AI landscape (DeepSeek vs Llama/Mistral) |

---

## Prompts

### Agent system prompt (`SYSTEM_PROMPT`)

```
You are a Bluesky post explainer. Your job is to explain the background
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

Do NOT summarize the post. Explain what it assumes the reader already knows.
```

### Self-critique prompt (`CRITIQUE_PROMPT`)

Used in a separate LLM call at `temperature=0.0` after each `finish()` to evaluate bullet quality. Returns `{verdict: "pass"|"fail", reason: string}`.

```
You are a quality reviewer for Bluesky post explanations.

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
- Bullets are vague or could apply to any post on the topic
```

### Eval judge prompt (`JUDGE_PROMPT`)

Used in `evals/run_eval.py` to score agent output against the gold summary. Returns `{verdict, score: 0-10, reason}`. Pass threshold: score ≥ 6.

```
You are evaluating a Bluesky post explainer agent.

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

Be strict. The agent should earn its pass.
```

---

## Design Decisions

**Why a raw tool-calling loop instead of a pipeline or framework?**
A hardcoded pipeline (extract entities → search → synthesise) can't adapt to what it finds. The LLM decides which concepts need context, how many searches to run, and whether to fetch specific posts for more detail. `tool_choice: "required"` keeps the loop deterministic — the LLM must always call a tool, so it can't drift into free-form responses. No framework dependency means the loop logic is plain Python and easy to follow.

**Why self-critique?**
The first `finish()` attempt frequently produces bullets that summarise the post rather than explaining background context. A cheap second LLM call at `temperature=0.0` with strict criteria catches this and forces iteration. The critique reason is fed back as a concrete instruction, not just a retry signal.

**Why `finish()` as a tool?**
Making `finish()` a tool (rather than relying on a natural stop) enforces a structured schema for bullets and sources via the tool parameter schema. It also makes the stop condition explicit: the loop only exits when `finish()` is called *and* the critique passes. If `finish()` appears alongside other tool calls in the same response, it takes priority.

**Why Bluesky search only (no web search)?**
The task is to explain posts *within the context of Bluesky discourse*. Searching Bluesky for how other users discuss the same concepts surfaces community understanding of terms — not just a Wikipedia definition. It also avoids external API dependencies, credentials, and rate limits.

**Source IDs**
Sources are 1-based integers assigned by the LLM itself in the `finish()` call, matching the `[N]` references in bullet text. The harness trusts the LLM's own citation numbering rather than re-indexing after the fact.

---

## File Layout

```
agent/
  config.py       # Settings dataclass — loads OPENAI_API_KEY + optional overrides from .env
  fetch.py        # Post dataclass, parse_bluesky_url(), build_at_uri(), fetch_post()
  main.py         # explain_post() — agentic loop, tool implementations, self-critique, CLI
evals/
  dataset.json    # 10 real Bluesky posts with gold summaries and category labels
  run_eval.py     # Eval harness: runs agent + LLM judge, writes results.json
run.sh            # ./run.sh agent "<url>"  |  ./run.sh eval
requirements.txt  # httpx>=0.27.0, python-dotenv>=1.0.1
.env.example      # Template — copy to .env and add your OPENAI_API_KEY
AGENTS.md         # Architecture notes and gotchas (Claude Code memory file)
```
