"""
InferTutor Quality Evaluation — Model Response Runner

Usage:
  python run_eval.py --model qwen --url https://your-modal-url.modal.run
  python run_eval.py --model mistral --url https://your-modal-url.modal.run

This script sends all 100 evaluation questions to the specified model endpoint
and saves the responses to eval/responses/{model}_responses.json.
"""

import argparse
import json
import time
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()
ROOT = Path(__file__).parent
EVAL_SET = ROOT / "eval_set.json"
RESPONSES_DIR = ROOT / "responses"
RESPONSES_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = (
    "You are InferTutor, a precise AI tutor for an inference engineering workshop. "
    "Answer with concrete systems reasoning. Keep answers compact, technical, and useful "
    "for a student optimizing vLLM serving."
)

MODEL_NAMES = {
    "qwen":    "Qwen/Qwen3-VL-4B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
}


def send_question(client: httpx.Client, url: str, model_name: str, question: str) -> dict:
    """Send one question and return timing + response text."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "max_tokens": 256,
        "temperature": 0.1,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        resp = client.post(
            f"{url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
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
        tokens_used = data.get("usage", {}).get("completion_tokens", None)
        return {
            "response": text,
            "error": None,
            "latency_ms": round(elapsed_ms, 1),
            "completion_tokens": tokens_used,
        }

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "response": None,
            "error": str(e),
            "latency_ms": round(elapsed_ms, 1),
        }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation questions against a model endpoint")
    parser.add_argument("--model", required=True, choices=["qwen", "mistral"],
                        help="Model shortname: qwen or mistral")
    parser.add_argument("--url",   required=True,
                        help="Modal endpoint URL, e.g. https://xxx--infertutor-yyy-serve.modal.run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last saved response (skip already-answered questions)")
    args = parser.parse_args()

    model_name = MODEL_NAMES[args.model]
    output_file = RESPONSES_DIR / f"{args.model}_responses.json"

    eval_data = json.loads(EVAL_SET.read_text())
    questions = eval_data["questions"]

    # Load existing responses if resuming
    existing = {}
    if args.resume and output_file.exists():
        saved = json.loads(output_file.read_text())
        existing = {str(r["id"]): r for r in saved.get("responses", [])}
        console.print(f"[yellow]Resuming: {len(existing)} questions already answered[/yellow]")

    console.print(f"\n[bold]Model:[/bold] {model_name}")
    console.print(f"[bold]Endpoint:[/bold] {args.url}")
    console.print(f"[bold]Questions:[/bold] {len(questions)}")
    console.print(f"[bold]Output:[/bold] {output_file}\n")

    # Verify endpoint is alive
    console.print("Checking endpoint health...")
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(f"{args.url.rstrip('/')}/health")
            if r.status_code == 200:
                console.print("[green]Endpoint is healthy. Starting evaluation...[/green]\n")
            else:
                console.print(f"[red]Health check failed: {r.status_code}[/red]")
                return
    except Exception as e:
        console.print(f"[red]Cannot reach endpoint: {e}[/red]")
        return

    responses = []
    errors = 0

    with httpx.Client(timeout=60) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running evaluation...", total=len(questions))

            for q in questions:
                qid = str(q["id"])

                # Skip if already answered and resuming
                if args.resume and qid in existing:
                    responses.append(existing[qid])
                    progress.advance(task)
                    continue

                progress.update(task, description=f"Q{q['id']:03d} [{q['category'][:20]}]")

                result = send_question(client, args.url, model_name, q["question"])

                entry = {
                    "id":               q["id"],
                    "category":         q["category"],
                    "difficulty":       q["difficulty"],
                    "question":         q["question"],
                    "reference_answer": q["reference_answer"],
                    "response":         result["response"],
                    "error":            result["error"],
                    "latency_ms":       result["latency_ms"],
                    "completion_tokens": result.get("completion_tokens"),
                }
                responses.append(entry)

                if result["error"]:
                    errors += 1
                    console.print(f"[red]Error Q{q['id']}: {result['error'][:80]}[/red]")

                # Save incrementally every 10 questions
                if len(responses) % 10 == 0:
                    _save(output_file, args.model, model_name, responses, eval_data["metadata"])

                progress.advance(task)
                time.sleep(0.2)  # Gentle pacing between requests

    # Final save
    _save(output_file, args.model, model_name, responses, eval_data["metadata"])

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Total questions: {len(responses)}")
    console.print(f"  Errors:          {errors}")
    console.print(f"  Success rate:    {((len(responses)-errors)/len(responses)*100):.1f}%")
    console.print(f"  Saved to:        {output_file}")


def _save(path, model_key, model_name, responses, metadata):
    latencies = [r["latency_ms"] for r in responses if r["response"]]
    output = {
        "metadata": {
            **metadata,
            "model_key":   model_key,
            "model_name":  model_name,
            "total_questions": len(responses),
            "errors":      sum(1 for r in responses if r["error"]),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        },
        "responses": responses,
    }
    path.write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
