#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from typing import Iterable, List, Dict, Any

from tqdm import tqdm

try:
    from openai import OpenAI, AsyncOpenAI
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
# NOTE: Replace this placeholder with your actual key if desired.
DEFAULT_OPENAI_API_KEY = "your-key-here"


def read_domains(csv_path: str) -> List[str]:
    domains: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            raw = (row[0] or "").strip()
            if not raw:
                continue
            # Normalize whitespace and lowercase
            domains.append(raw.lower())
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for d in domains:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return unique


DOMAIN_RE = re.compile(r"^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


def quick_prefilter(domains: Iterable[str]) -> List[str]:
    kept: List[str] = []
    for d in domains:
        if not DOMAIN_RE.match(d):
            continue
        # Reject excessive hyphens or length
        if d.count("-") > 2:
            continue
        # Reject very long SLDs (before first dot)
        sld = d.split(".")[0]
        if len(sld) > 20 or len(sld) < 2:
            continue
        kept.append(d)
    return kept


SYSTEM_PROMPT = (
    "You are a strict domain evaluator that decides if domains are meaningful, "
    "brandable, and likely valuable. You are concise and only output JSON."
)

USER_INSTRUCTIONS = (
    "Given a list of domain names, evaluate each domain and return a JSON array.\n"
    "For each domain, provide: domain (string), keep (boolean), score (0-10 integer), reason (string).\n\n"
    "Guidelines for KEEP=true (score>=7 usually):\n"
    "- Meaningful words or strong brandable compound.\n"
    "- Clear, memorable, easy to pronounce/spell.\n"
    "- Prefer short SLD (<=15 chars), avoid many hyphens/numbers unless meaningful.\n"
    "- Generic/commercial value (e.g., broad niches, common terms).\n"
    "- Avoid trademarks, adult/spammy terms, confusing strings.\n\n"
    "Only output a JSON array with the structure described. No extra commentary."
)


def chunked(seq: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def extract_json(text: str) -> Any:
    """Attempt to extract and parse the first JSON array/object in text."""
    start = None
    end = None
    for i, ch in enumerate(text):
        if ch in "[{":
            start = i
            break
    if start is None:
        raise ValueError("No JSON start found in model output")
    # find matching end by scanning from the end
    for j in range(len(text) - 1, -1, -1):
        if text[j] in "]}":
            end = j + 1
            break
    if end is None:
        raise ValueError("No JSON end found in model output")
    snippet = text[start:end]
    return json.loads(snippet)


def evaluate_with_openai(
    api_key: str,
    model: str,
    domains: List[str],
    batch_size: int = 50,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> List[Dict[str, Any]]:
    if OpenAI is None:
        raise RuntimeError(
            "openai package not available. Install dependencies with pip install -r requirements.txt"
        )

    client = OpenAI(api_key=api_key)
    results: List[Dict[str, Any]] = []

    for group in tqdm(list(chunked(domains, batch_size)), desc="Evaluating", unit="batch"):
        content = "\n".join(group)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTIONS + "\n\nDomains:\n" + content},
        ]

        for attempt in range(1, max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=1.0,
                )
                text = resp.choices[0].message.content or ""
                parsed = extract_json(text)
                if not isinstance(parsed, list):
                    raise ValueError("Model did not return a JSON array")
                # normalize entries
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    dom = str(item.get("domain", "")).lower()
                    keep = bool(item.get("keep", False))
                    score = int(item.get("score", 0))
                    reason = str(item.get("reason", "")).strip()
                    if dom:
                        results.append(
                            {"domain": dom, "keep": keep, "score": score, "reason": reason}
                        )
                break  # success → next batch
            except Exception as e:
                if attempt >= max_retries:
                    raise
                time.sleep(retry_backoff * attempt)
                continue

    return results


async def _eval_batch_async(
    client: "AsyncOpenAI",
    model: str,
    group: List[str],
    temperature: float,
    max_retries: int,
    retry_backoff: float,
) -> List[Dict[str, Any]]:
    content = "\n".join(group)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTIONS + "\n\nDomains:\n" + content},
    ]
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
            )
            text = resp.choices[0].message.content or ""
            parsed = extract_json(text)
            out: List[Dict[str, Any]] = []
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    dom = str(item.get("domain", "")).lower()
                    keep = bool(item.get("keep", False))
                    score = int(item.get("score", 0))
                    reason = str(item.get("reason", "")).strip()
                    if dom:
                        out.append({"domain": dom, "keep": keep, "score": score, "reason": reason})
            return out
        except Exception:
            if attempt >= max_retries:
                raise
            await asyncio.sleep(retry_backoff * attempt)


async def evaluate_with_openai_async(
    api_key: str,
    model: str,
    domains: List[str],
    batch_size: int = 50,
    concurrency: int = 5,
    temperature: float = 1.0,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
) -> List[Dict[str, Any]]:
    if AsyncOpenAI is None:
        raise RuntimeError(
            "openai package does not expose AsyncOpenAI. Ensure openai>=1.0 is installed."
        )
    client = AsyncOpenAI(api_key=api_key)
    groups = list(chunked(domains, batch_size))

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def runner(group: List[str]) -> List[Dict[str, Any]]:
        async with semaphore:
            return await _eval_batch_async(
                client=client,
                model=model,
                group=group,
                temperature=temperature,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )

    tasks = [asyncio.create_task(runner(g)) for g in groups]

    results: List[Dict[str, Any]] = []
    for _ in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating", unit="batch"):
        batch_res = await _
        results.extend(batch_res)

    await client.close()
    return results


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Filter expiring domains using OpenAI to keep meaningful, valuable ones."
    )
    p.add_argument("--input", required=True, help="Path to input CSV (first column = domain)")
    p.add_argument("--output", default="valuable_domains.csv", help="Output CSV path")
    p.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    p.add_argument("--api-key", default=None, help="OpenAI API key (optional)")
    p.add_argument(
        "--min-score",
        type=int,
        default=7,
        help="Minimum score (0-10) to include in output",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of domains per API call",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = deterministic; higher = more diverse)",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel API calls (use >1 for parallel mode)",
    )
    p.add_argument(
        "--no-prefilter",
        action="store_true",
        help="Disable quick regex-based prefilter",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output; just print summary",
    )

    args = p.parse_args(argv)

    api_key = (
        args.api_key
        or os.environ.get("OPENAI_API_KEY")
        or DEFAULT_OPENAI_API_KEY
    )

    all_domains = read_domains(args.input)
    domains = all_domains if args.no_prefilter else quick_prefilter(all_domains)

    if not domains:
        print("No valid domains after prefiltering.")
        return 1

    # Choose sequential vs parallel evaluation
    if args.concurrency and args.concurrency > 1:
        if AsyncOpenAI is None:
            print("Async client not available; falling back to sequential.")
            evaluated = evaluate_with_openai(
                api_key=api_key,
                model=args.model,
                domains=domains,
                batch_size=args.batch_size,
            )
        else:
            evaluated = asyncio.run(
                evaluate_with_openai_async(
                    api_key=api_key,
                    model=args.model,
                    domains=domains,
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    temperature=args.temperature,
                )
            )
    else:
        evaluated = evaluate_with_openai(
            api_key=api_key,
            model=args.model,
            domains=domains,
            batch_size=args.batch_size,
            temperature=args.temperature,
        )

    # Merge back to dedupe and preserve the best entry per domain
    best: Dict[str, Dict[str, Any]] = {}
    for item in evaluated:
        dom = item["domain"]
        prev = best.get(dom)
        if prev is None or item.get("score", 0) > prev.get("score", 0):
            best[dom] = item

    kept_rows = [r for r in best.values() if r.get("keep") and r.get("score", 0) >= args.min_score]
    kept_rows.sort(key=lambda x: (-int(x.get("score", 0)), x.get("domain", "")))

    if args.dry_run:
        print(f"Input domains: {len(all_domains)}")
        print(f"After prefilter: {len(domains)}")
        print(f"Evaluated: {len(evaluated)}")
        print(f"Kept (score >= {args.min_score}): {len(kept_rows)}")
        for r in kept_rows[:20]:
            print(f"{r['domain']}, score={r['score']}, reason={r['reason']}")
        return 0

    write_csv(
        args.output,
        kept_rows,
        fieldnames=["domain", "score", "reason"],
    )

    print(f"Wrote {len(kept_rows)} domains to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
