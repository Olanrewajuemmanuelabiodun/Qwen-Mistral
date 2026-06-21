"""
Quality Evaluation Runner — InferTutor Arena
Sends all 100 questions to both Qwen and Mistral, collects full responses,
and saves results to eval_responses.json for human scoring.

Usage:
  python run_eval.py --qwen-url <url> --mistral-url <url>

Both models must already be deployed and healthy on Modal before running.
Deploy each with --deploy-only flag first:

  python run_infertutor_experiment.py \
    --label eval-qwen \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --gpu-type H100 --replicas 1 --no-fast-boot \
    --deploy-only

  python run_infertutor_experiment.py \
    --label eval-mistral \
    --model mistralai/Mistral-7B-Instruct-v0.3 \
    --gpu-type H100 --replicas 1 --no-fast-boot \
    --deploy-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()
ROOT = Path(__file__).parent

QWEN_MODEL   = "Qwen/Qwen3-VL-4B-Instruct"
MISTRAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = (
    "You are InferTutor, a precise AI tutor for an inference engineering workshop. "
    "Answer with concrete systems reasoning. Keep answers compact, technical, and useful "
    "for a student optimizing vLLM serving."
)


def wait_for_health(url: str, model_name: str, timeout_s: int = 300):
    """Wait until the vLLM endpoint is healthy."""
    console.print(f"[bold]Checking health:[/bold] {model_name}")
    deadline = time.time() + timeout_s
    with httpx.Client(timeout=15) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{url.rstrip('/')}/health")
                if resp.status_code == 200:
                    console.print(f"[green]{model_name} is healthy[/green]")
                    return
            except Exception:
                pass
            time.sleep(5)
    raise TimeoutError(f"{model_name} did not become healthy within {timeout_s}s")


def query_model(url: str, model_name: str, question: str, max_tokens: int = 256) -> dict:
    """Send a single question to a model and return the full response text + metadata."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return {
                "response": None,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                "latency_ms": elapsed_ms,
            }

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return {
            "response": text,
            "error": None,
            "latency_ms": round(elapsed_ms, 1),
            "total_tokens": data.get("usage", {}).get("total_tokens", None),
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "response": None,
            "error": str(e),
            "latency_ms": round(elapsed_ms, 1),
        }


def run_eval(qwen_url: str, mistral_url: str, output_path: Path, max_tokens: int = 256):
    """Run all questions against both models and save results."""

    questions_path = ROOT / "eval_questions.json"
    if not questions_path.exists():
        raise FileNotFoundError(f"eval_questions.json not found at {questions_path}")

    data = json.loads(questions_path.read_text())
    questions = data["questions"]
    console.print(f"[bold]Loaded {len(questions)} questions[/bold]")

    # Check both models are healthy
    wait_for_health(qwen_url,    QWEN_MODEL)
    wait_for_health(mistral_url, MISTRAL_MODEL)

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running evaluation...", total=len(questions))

        for q in questions:
            qid      = q["id"]
            category = q["category"]
            question = q["question"]
            reference= q["reference"]

            progress.update(task, description=f"Q{qid:03d} [{category[:20]}]")

            # Query Qwen
            qwen_result = query_model(qwen_url, QWEN_MODEL, question, max_tokens)

            # Small pause to avoid hammering
            time.sleep(0.3)

            # Query Mistral
            mistral_result = query_model(mistral_url, MISTRAL_MODEL, question, max_tokens)

            results.append({
                "id":        qid,
                "category":  category,
                "question":  question,
                "reference": reference,
                "qwen": {
                    "model":      QWEN_MODEL,
                    "response":   qwen_result["response"],
                    "error":      qwen_result["error"],
                    "latency_ms": qwen_result["latency_ms"],
                    "score":      None,  # filled by human scorer
                    "notes":      "",    # human notes field
                },
                "mistral": {
                    "model":      MISTRAL_MODEL,
                    "response":   mistral_result["response"],
                    "error":      mistral_result["error"],
                    "latency_ms": mistral_result["latency_ms"],
                    "score":      None,  # filled by human scorer
                    "notes":      "",    # human notes field
                },
            })

            progress.advance(task)
            time.sleep(0.2)

    # Save results
    output = {
        "metadata": {
            "total_questions": len(results),
            "qwen_model":     QWEN_MODEL,
            "mistral_model":  MISTRAL_MODEL,
            "max_tokens":     max_tokens,
            "temperature":    0.0,
            "run_timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "scoring_rubric": {
                "3": "Fully correct — accurate, complete, no hallucinations",
                "2": "Partially correct — mostly right but missing key detail or minor inaccuracy",
                "1": "Mostly wrong — fundamental misunderstanding or significant hallucination",
                "0": "Completely wrong or refused to answer"
            },
            "scoring_instructions": (
                "For each question, read the reference answer, then score both model responses "
                "independently on the 0-3 scale. Fill in the 'score' field and optionally add "
                "notes about what was wrong or hallucinated."
            )
        },
        "results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[green bold]Saved {len(results)} results to {output_path}[/green bold]")

    # Print quick summary
    qwen_errors    = sum(1 for r in results if r["qwen"]["error"])
    mistral_errors = sum(1 for r in results if r["mistral"]["error"])
    qwen_avg_ms    = sum(r["qwen"]["latency_ms"] for r in results if not r["qwen"]["error"]) / max(1, len(results) - qwen_errors)
    mistral_avg_ms = sum(r["mistral"]["latency_ms"] for r in results if not r["mistral"]["error"]) / max(1, len(results) - mistral_errors)

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Qwen errors:    {qwen_errors} / {len(results)}")
    console.print(f"  Mistral errors: {mistral_errors} / {len(results)}")
    console.print(f"  Qwen avg latency:    {qwen_avg_ms:.0f}ms")
    console.print(f"  Mistral avg latency: {mistral_avg_ms:.0f}ms")
    console.print(f"\nNext step: open [bold]{output_path}[/bold] and fill in the 'score' fields.")


def main():
    parser = argparse.ArgumentParser(description="Run quality evaluation against Qwen and Mistral")
    parser.add_argument("--qwen-url",    required=True, help="Deployed Qwen vLLM endpoint URL")
    parser.add_argument("--mistral-url", required=True, help="Deployed Mistral vLLM endpoint URL")
    parser.add_argument("--output",      default="eval_responses.json", help="Output file path")
    parser.add_argument("--max-tokens",  type=int, default=256, help="Max tokens per response")
    args = parser.parse_args()

    output_path = ROOT / args.output
    run_eval(args.qwen_url, args.mistral_url, output_path, args.max_tokens)


if __name__ == "__main__":
    main()
