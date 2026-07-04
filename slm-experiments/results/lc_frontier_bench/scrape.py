"""Scrape 10 LeetCode problem statements via GraphQL (curl subprocess to bypass urllib 499)."""
import json, subprocess, re, html, pathlib, sys

SLUGS = [
    "valid-parentheses",
    "merge-two-sorted-lists",
    "maximum-subarray",
    "climbing-stairs",
    "best-time-to-buy-and-sell-stock",
    "contains-duplicate",
    "product-of-array-except-self",
    "single-number",
    "move-zeroes",
    "reverse-linked-list",
]

Q = ("query getQuestion($titleSlug: String!) { "
     "question(titleSlug: $titleSlug) { title titleSlug difficulty content } }")

HEADERS = [
    "-H", "Content-Type: application/json",
    "-H", "Origin: https://leetcode.com",
    "-H", "User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
]

def strip_html(s: str) -> str:
    s = re.sub(r"<sup>(.*?)</sup>", r"^\1", s, flags=re.S)
    s = re.sub(r"<sub>(.*?)</sub>", r"_\1", s, flags=re.S)
    s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
    s = re.sub(r"<strong>(.*?)</strong>", r"**\1**", s, flags=re.S)
    s = re.sub(r"<em>(.*?)</em>", r"*\1*", s, flags=re.S)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"</p>\s*<p>", "\n\n", s)
    s = re.sub(r"</?p>", "", s)
    s = re.sub(r"</?ul>", "", s)
    s = re.sub(r"<li>(.*?)</li>", r"- \1", s, flags=re.S)
    s = re.sub(r"</?ol>", "", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s

def fetch(slug: str) -> dict:
    body = json.dumps({"query": Q, "variables": {"titleSlug": slug}})
    cmd = ["curl", "-s", "-X", "POST", "https://leetcode.com/graphql/",
           "-H", f"Referer: https://leetcode.com/problems/{slug}/",
           *HEADERS, "--data", body]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    d = json.loads(out)["data"]["question"]
    return {
        "slug": d["titleSlug"],
        "title": d["title"],
        "difficulty": d["difficulty"],
        "statement": strip_html(d["content"]),
    }

def main():
    out_dir = pathlib.Path("/tmp/lc_bench")
    out_dir.mkdir(exist_ok=True)
    problems = []
    for s in SLUGS:
        p = fetch(s)
        problems.append(p)
        print(f"{p['difficulty']:8} {p['title']:45} {len(p['statement'])} chars", file=sys.stderr)
    (out_dir / "problems.json").write_text(json.dumps(problems, indent=2))
    print(f"\nWrote {len(problems)} problems to {out_dir/'problems.json'}", file=sys.stderr)

if __name__ == "__main__":
    main()
