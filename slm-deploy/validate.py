#!/usr/bin/env python3
"""Validate that an SLMDeployment set fits on a Node.

Usage:
    python3 slm-deploy/validate.py slm-deploy/node.yaml slm-deploy/slms.yaml

Sums (replicas * requests) per dimension across all SLMDeployment documents
and compares to the Node's `allocatable`. Prints a per-resource fit table and
exits 1 if any dimension is over-committed.

Memory units: K8s style (Ki/Mi/Gi/Ti/Pi/Ei = 1024^n, K/M/G/T/P/E = 1000^n).
CPU units: integer cores, or `<n>m` for millicores.
GPU: integer device count (Node may set `gpu.shareable: true` to skip summing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# -------- minimal YAML loader (block mappings + multi-doc) ------------------

def _parse_scalar(s: str) -> Any:
    s = s.strip()
    if not s:
        return None
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "~"):
        return None
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _strip_comment(line: str) -> str:
    in_str = None
    out = []
    for ch in line:
        if in_str:
            out.append(ch)
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'"):
            in_str = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _load_doc(lines: list[tuple[int, str]]) -> Any:
    """lines is list of (indent, content) with content non-empty."""
    if not lines:
        return None

    def parse_block(idx: int, indent: int) -> tuple[Any, int]:
        # All lines at this indent form one mapping (we don't support block lists).
        mapping: dict[str, Any] = {}
        while idx < len(lines):
            ind, content = lines[idx]
            if ind < indent:
                break
            if ind > indent:
                raise ValueError(f"unexpected indent at line content: {content!r}")
            if ":" not in content:
                raise ValueError(f"expected `key: value`, got: {content!r}")
            key, _, rest = content.partition(":")
            key = key.strip()
            rest = rest.strip()
            idx += 1
            if rest == "":
                # nested mapping on following lines
                if idx < len(lines) and lines[idx][0] > indent:
                    value, idx = parse_block(idx, lines[idx][0])
                else:
                    value = None
            else:
                value = _parse_scalar(rest)
            mapping[key] = value
        return mapping, idx

    value, _ = parse_block(0, lines[0][0])
    return value


def load_yaml_docs(path: Path) -> list[Any]:
    raw = path.read_text().splitlines()
    docs: list[list[tuple[int, str]]] = [[]]
    for raw_line in raw:
        stripped = _strip_comment(raw_line)
        if stripped.strip() == "---":
            docs.append([])
            continue
        if not stripped.strip():
            continue
        indent = len(stripped) - len(stripped.lstrip(" "))
        docs[-1].append((indent, stripped.strip()))
    return [_load_doc(d) for d in docs if d]


# -------- resource unit parsing --------------------------------------------

_BIN = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "Ti": 1024**4,
        "Pi": 1024**5, "Ei": 1024**6}
_DEC = {"K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,
        "P": 10**15, "E": 10**18}


def parse_memory(v: Any) -> int:
    if isinstance(v, int):
        return v
    s = str(v).strip()
    for suf, mult in _BIN.items():
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * mult)
    for suf, mult in _DEC.items():
        if s.endswith(suf):
            return int(float(s[:-len(suf)]) * mult)
    return int(s)


def parse_cpu(v: Any) -> int:
    """Return millicores."""
    if isinstance(v, int):
        return v * 1000
    s = str(v).strip()
    if s.endswith("m"):
        return int(s[:-1])
    return int(float(s) * 1000)


def parse_gpu(v: Any) -> int:
    return int(v) if v is not None else 0


def fmt_memory(n: int) -> str:
    if n >= 1024**3:
        return f"{n / 1024**3:.2f} Gi"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f} Mi"
    return f"{n} B"


def fmt_cpu(milli: int) -> str:
    return f"{milli} m" if milli < 1000 else f"{milli/1000:.2f} cores"


# -------- core logic --------------------------------------------------------

def collect_deployments(docs: list[Any]) -> list[dict]:
    out = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        if d.get("kind") != "SLMDeployment":
            continue
        out.append(d)
    return out


def find_node(docs: list[Any]) -> dict:
    for d in docs:
        if isinstance(d, dict) and d.get("kind") == "Node":
            return d
    raise SystemExit("no `kind: Node` document found in node file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("node_file", type=Path)
    ap.add_argument("deploy_files", type=Path, nargs="+")
    args = ap.parse_args()

    node = find_node(load_yaml_docs(args.node_file))
    deploys: list[dict] = []
    for p in args.deploy_files:
        deploys.extend(collect_deployments(load_yaml_docs(p)))

    alloc = node["spec"]["allocatable"]
    cap_mem = parse_memory(alloc["memory"])
    cap_cpu = parse_cpu(alloc["cpu"])
    cap_gpu = parse_gpu(alloc.get("gpu", 0))
    gpu_shareable = bool(((node["spec"].get("gpu") or {}).get("shareable")) or False)

    used_mem = 0
    used_cpu = 0
    used_gpu = 0
    max_per_replica_gpu = 0

    print(f"Node: {node['metadata']['name']}  "
          f"allocatable mem={fmt_memory(cap_mem)}  cpu={fmt_cpu(cap_cpu)}  "
          f"gpu={cap_gpu}{' (shareable)' if gpu_shareable else ''}\n")

    header = f"{'Deployment':<14} {'Model':<14} {'Repl':>4} {'mem/repl':>10} {'cpu/repl':>10} {'gpu/repl':>8}"
    print(header)
    print("-" * len(header))
    for d in deploys:
        spec = d["spec"]
        name = d["metadata"]["name"]
        model = spec["model"]
        replicas = int(spec.get("replicas", 1))
        req = spec["resources"]["requests"]
        m = parse_memory(req["memory"])
        c = parse_cpu(req["cpu"])
        g = parse_gpu(req.get("gpu", 0))
        used_mem += m * replicas
        used_cpu += c * replicas
        used_gpu += g * replicas
        if g > max_per_replica_gpu:
            max_per_replica_gpu = g
        print(f"{name:<14} {model:<14} {replicas:>4} {fmt_memory(m):>10} {fmt_cpu(c):>10} {g:>8}")

    print()
    rows = [
        ("memory", used_mem, cap_mem, fmt_memory),
        ("cpu",    used_cpu, cap_cpu, fmt_cpu),
    ]
    if gpu_shareable:
        rows.append(("gpu (shared, per-replica)", max_per_replica_gpu, cap_gpu, str))
    else:
        rows.append(("gpu", used_gpu, cap_gpu, str))

    print(f"{'Resource':<28} {'Requested':>14} {'Allocatable':>14}  Fit")
    print("-" * 72)
    overflow = False
    for name, used, cap, fmt in rows:
        ok = used <= cap
        if not ok:
            overflow = True
        print(f"{name:<28} {fmt(used):>14} {fmt(cap):>14}  {'OK' if ok else 'OVER'}")

    print()
    if overflow:
        print("RESULT: over-committed — adjust replicas or resources.")
        return 1
    print("RESULT: fits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
