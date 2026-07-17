"""Rule-based prompt decomposer.

Given a user prompt (the raw text they typed into Claude Code), decide
whether it maps to a small DAG of homogeneous leaves that a local SLM
pool can handle in parallel. If so, return a plan dict ready to hand to
`slm_submit_plan`. Otherwise return None and let the frontier answer.

Precision-first. A false positive (auto-delegating a task the frontier
would have handled better) is worse than a false negative (looking like
today's behavior), so every rule requires a clear structural signal
(N-item list, "for each", "N variants of X", etc.) before firing.

The three patterns:

  1. Multi-variant drafts:  "give me 5 taglines for X"
  2. For-each over list:    "for each of these bullets, write a paragraph"
  3. Summarize N items:     "summarize each of the following 4 paragraphs"

Everything else -> None -> solo.
"""
from __future__ import annotations

import re
from typing import Any


DECOMPOSER_VERSION = "v2"

MIN_LEAVES = 3
MAX_LEAVES = 12  # cap so a bad rule can't fan out into 500 SLM calls

# multi_variant floor is stricter: short-output variant tasks (taglines,
# tweets, subject lines) lose tokens even under the stashed-plan design,
# because the SLM results still round-trip through the frontier for
# presentation. Empirically N>=6 is where parallelism starts beating
# solo on wall-clock and where token cost stops being worse than solo.
# See slm-experiments/autoroute_validate.py.
MIN_LEAVES_MULTI_VARIANT = 6

_NUMBER_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

# "give me 5 taglines for X", "write six variants of Y", "draft 4 subject lines"
_VARIANT_RE = re.compile(
    r"""
    \b(?:give\ me|write|draft|generate|produce|come\ up\ with|suggest)\s+
    (?P<n>\d+|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+
    (?:different\s+|distinct\s+|)
    (?P<noun>[a-z][\w\- ]{2,40}?)
    (?:\s+(?:for|about|on|of)\s+(?P<subject>.+))?
    $
    """,
    re.VERBOSE | re.IGNORECASE,
)

# "for each of the following", "for each of these", "for every ... below"
_FOR_EACH_TRIGGER_RE = re.compile(
    r"\b(?:for\s+each|for\s+every|one\s+\w+\s+per|per\s+each)\b",
    re.IGNORECASE,
)

# "summarize each of the following", "give me a tl;dr of each", "summary of each"
_SUMMARIZE_TRIGGER_RE = re.compile(
    r"""
    \b(?:
        summari[sz]e(?:\s+each|\s+every|\s+the\s+following|\s+these|\s+all)?
      | (?:give\s+(?:me\s+)?)?tl;?dr(?:\s+of\s+each|\s+of\s+the\s+following)?
      | one[- ]line\s+summary\s+(?:of\s+each|per)
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Bulleted or numbered list item at the start of a line.
_LIST_ITEM_RE = re.compile(
    r"""^\s*
        (?:
            [-*+]\s+           # -, *, +
          | \d{1,2}[.)]\s+     # 1. or 1)
          | [•·]\s+
        )
        (?P<body>.+?)\s*$
    """,
    re.VERBOSE,
)


def _parse_count(raw: str) -> int | None:
    raw = raw.strip().lower()
    if raw.isdigit():
        return int(raw)
    return _NUMBER_WORDS.get(raw)


def _extract_list_items(prompt: str) -> list[str]:
    """Pull out contiguous bulleted / numbered items from a prompt.

    Requires at least MIN_LEAVES contiguous list lines; otherwise returns [].
    This is deliberately strict — random bullets scattered through prose are
    not a list.
    """
    items: list[str] = []
    run: list[str] = []
    best_run: list[str] = []
    for line in prompt.splitlines():
        m = _LIST_ITEM_RE.match(line)
        if m:
            run.append(m.group("body").strip())
        else:
            if len(run) > len(best_run):
                best_run = run
            run = []
    if len(run) > len(best_run):
        best_run = run
    items = best_run
    if len(items) < MIN_LEAVES or len(items) > MAX_LEAVES:
        return []
    return items


def _plan(plan_id: str, description: str, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "description": description,
        "nodes": nodes,
    }


def _multi_variant(prompt: str) -> dict[str, Any] | None:
    """Match 'give me 5 taglines for X' style prompts."""
    text = prompt.strip()
    if len(text) > 500:  # variant asks are short by nature
        return None
    m = _VARIANT_RE.search(text)
    if m is None:
        return None
    n = _parse_count(m.group("n"))
    if n is None or n < MIN_LEAVES_MULTI_VARIANT or n > MAX_LEAVES:
        return None
    noun = m.group("noun").strip()
    subject = (m.group("subject") or "").strip().rstrip(".!?")
    if not noun:
        return None
    nodes: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        seed_hint = f" (variant {i}/{n}, make it distinct from the others)"
        leaf_prompt = (
            f"Write ONE {noun.rstrip('s')}"
            + (f" for {subject}" if subject else "")
            + f".{seed_hint} Output only the {noun.rstrip('s')} itself, no preamble."
        )
        nodes.append({"id": f"variant_{i}", "prompt": leaf_prompt, "depends_on": []})
    return _plan(
        plan_id=f"variants-{n}",
        description=f"{n} variants of: {noun}" + (f" ({subject})" if subject else ""),
        nodes=nodes,
    )


def _for_each(prompt: str) -> dict[str, Any] | None:
    """Match 'for each of the following ..., do X' with an explicit N-item list."""
    if not _FOR_EACH_TRIGGER_RE.search(prompt):
        return None
    items = _extract_list_items(prompt)
    if not items:
        return None
    # The instruction is whatever appears in the prompt *before* the list.
    # Best-effort: take the first non-list line that contains the "for each" phrase.
    instruction = ""
    for line in prompt.splitlines():
        if _LIST_ITEM_RE.match(line):
            break
        if _FOR_EACH_TRIGGER_RE.search(line):
            instruction = line.strip()
            break
    if not instruction:
        return None
    nodes: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        leaf_prompt = (
            f"{instruction}\n\n"
            f"Do this only for the following one item, and output only the result:\n"
            f"- {item}"
        )
        nodes.append({"id": f"item_{i}", "prompt": leaf_prompt, "depends_on": []})
    return _plan(
        plan_id=f"foreach-{len(items)}",
        description=f"per-item: {instruction[:80]}",
        nodes=nodes,
    )


def _summarize(prompt: str) -> dict[str, Any] | None:
    """Match 'summarize each of the following N paragraphs / bullets'."""
    if not _SUMMARIZE_TRIGGER_RE.search(prompt):
        return None
    items = _extract_list_items(prompt)
    if items:
        nodes = [
            {
                "id": f"sum_{i}",
                "prompt": f"Summarize this in 1-2 sentences. Output only the summary:\n\n{item}",
                "depends_on": [],
            }
            for i, item in enumerate(items, start=1)
        ]
        return _plan(
            plan_id=f"summarize-list-{len(items)}",
            description=f"per-item summary of {len(items)} items",
            nodes=nodes,
        )
    # Fallback: paragraph-split. Only fire if we get MIN_LEAVES..MAX_LEAVES
    # non-empty paragraphs of substantial length.
    paras = [p.strip() for p in re.split(r"\n\s*\n", prompt) if p.strip()]
    # Drop the first paragraph if it's the instruction ("summarize the following").
    if paras and _SUMMARIZE_TRIGGER_RE.search(paras[0]) and len(paras[0]) < 200:
        paras = paras[1:]
    substantive = [p for p in paras if len(p) >= 80]
    if not (MIN_LEAVES <= len(substantive) <= MAX_LEAVES):
        return None
    nodes = [
        {
            "id": f"sum_{i}",
            "prompt": f"Summarize this in 1-2 sentences. Output only the summary:\n\n{p}",
            "depends_on": [],
        }
        for i, p in enumerate(substantive, start=1)
    ]
    return _plan(
        plan_id=f"summarize-paras-{len(substantive)}",
        description=f"per-paragraph summary of {len(substantive)} paragraphs",
        nodes=nodes,
    )


_RULES = (
    ("multi_variant", _multi_variant),
    ("for_each",      _for_each),
    ("summarize",     _summarize),
)


def decompose(prompt: str) -> tuple[dict[str, Any] | None, str]:
    """Return (plan, rule_name). plan is None if no rule matches confidently.

    rule_name is 'none' when nothing matched — callers use it for logging.
    """
    if not prompt or not prompt.strip():
        return None, "empty"
    for name, fn in _RULES:
        try:
            plan = fn(prompt)
        except Exception:
            continue
        if plan is not None:
            return plan, name
    return None, "none"
