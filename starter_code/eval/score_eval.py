"""
InferTutor Quality Evaluation — Human Scoring Tool

Usage:
  python score_eval.py

This tool loads both model responses, presents them side by side,
and lets you score each on a 1-3 scale. Progress is saved after
every question so you can stop and resume anytime.

Scoring rubric:
  3 = Correct and complete — factually accurate, covers key points, no hallucinations
  2 = Partially correct — mostly right but missing key points or minor inaccuracies
  1 = Incorrect or hallucination — factually wrong, invents flags/concepts, misses the point
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent
RESPONSES_DIR = ROOT / "responses"
SCORES_FILE = ROOT / "scores.json"

QWEN_FILE    = RESPONSES_DIR / "qwen_responses.json"
MISTRAL_FILE = RESPONSES_DIR / "mistral_responses.json"

RUBRIC = {
    "3": "Correct and complete — factually accurate, covers key points, no hallucinations",
    "2": "Partially correct — mostly right but missing key points or minor inaccuracies",
    "1": "Incorrect or hallucination — factually wrong, invents concepts/flags, misses the point",
}

SEP = "=" * 80
THIN = "-" * 80


def clear():
    os.system("clear" if os.name == "posix" else "cls")


def load_json(path):
    if not path.exists():
        print(f"ERROR: File not found: {path}")
        print("Run run_eval.py for both models first.")
        sys.exit(1)
    return json.loads(path.read_text())


def load_scores():
    if SCORES_FILE.exists():
        return json.loads(SCORES_FILE.read_text())
    return {}


def save_scores(scores):
    SCORES_FILE.write_text(json.dumps(scores, indent=2))


def get_score(prompt_label):
    while True:
        val = input(f"  Score for {prompt_label} (1/2/3, or s to skip, q to quit): ").strip().lower()
        if val in ("1", "2", "3"):
            return int(val)
        if val == "s":
            return None
        if val == "q":
            print("\nProgress saved. Run again to resume.\n")
            sys.exit(0)
        print("  Enter 1, 2, 3, s (skip), or q (quit).")


def print_rubric():
    print("\nSCORING RUBRIC:")
    for k, v in RUBRIC.items():
        print(f"  {k} = {v}")
    print()


def show_question(i, total, q_data, qwen_resp, mistral_resp, existing_score):
    clear()
    print(SEP)
    print(f"  QUESTION {i}/{total}  |  ID: {q_data['id']}  |  Category: {q_data['category']}  |  Difficulty: {q_data['difficulty'].upper()}")
    print(SEP)

    print("\nQUESTION:")
    print(f"  {q_data['question']}")

    print(f"\n{THIN}")
    print("REFERENCE ANSWER:")
    print(f"  {q_data['reference_answer']}")

    print(f"\n{THIN}")
    print(f"QWEN3-VL-4B RESPONSE  (latency: {qwen_resp.get('latency_ms', '?')}ms):")
    resp_q = qwen_resp.get("response") or f"[ERROR: {qwen_resp.get('error', 'no response')}]"
    for line in resp_q.split("\n"):
        print(f"  {line}")

    print(f"\n{THIN}")
    print(f"MISTRAL-7B RESPONSE  (latency: {mistral_resp.get('latency_ms', '?')}ms):")
    resp_m = mistral_resp.get("response") or f"[ERROR: {mistral_resp.get('error', 'no response')}]"
    for line in resp_m.split("\n"):
        print(f"  {line}")

    print(f"\n{THIN}")
    print_rubric()

    if existing_score:
        print(f"  [Already scored: Qwen={existing_score.get('qwen')}, Mistral={existing_score.get('mistral')}]")
        rescore = input("  Rescore this question? (y/n): ").strip().lower()
        if rescore != "y":
            return existing_score["qwen"], existing_score["mistral"]

    qwen_score    = get_score("Qwen3-VL-4B")
    mistral_score = get_score("Mistral-7B  ")
    return qwen_score, mistral_score


def main():
    print("\nLoading responses...")
    qwen_data    = load_json(QWEN_FILE)
    mistral_data = load_json(MISTRAL_FILE)

    qwen_map    = {str(r["id"]): r for r in qwen_data["responses"]}
    mistral_map = {str(r["id"]): r for r in mistral_data["responses"]}

    all_ids = sorted(qwen_map.keys(), key=lambda x: int(x))
    total = len(all_ids)

    scores = load_scores()
    scored = sum(1 for qid in all_ids if qid in scores and scores[qid].get("qwen") is not None)

    print(f"\n  Total questions: {total}")
    print(f"  Already scored:  {scored}")
    print(f"  Remaining:       {total - scored}")
    print(f"\n  Scores saved to: {SCORES_FILE}")
    print("\n  Controls: 1/2/3 to score | s to skip | q to quit and save")
    input("\n  Press Enter to start...\n")

    i = 0
    for qid in all_ids:
        i += 1
        qwen_resp    = qwen_map.get(qid, {})
        mistral_resp = mistral_map.get(qid, {})
        existing     = scores.get(qid)

        # Build reference from qwen (both have same question/reference)
        q_data = {
            "id":               int(qid),
            "category":         qwen_resp.get("category", ""),
            "difficulty":       qwen_resp.get("difficulty", ""),
            "question":         qwen_resp.get("question", ""),
            "reference_answer": qwen_resp.get("reference_answer", ""),
        }

        qscore, mscore = show_question(i, total, q_data, qwen_resp, mistral_resp, existing)

        scores[qid] = {
            "id":         int(qid),
            "category":   q_data["category"],
            "difficulty": q_data["difficulty"],
            "qwen":       qscore,
            "mistral":    mscore,
        }
        save_scores(scores)

    # Summary
    clear()
    print(SEP)
    print("  SCORING COMPLETE — SUMMARY")
    print(SEP)

    scored_pairs = [(s["qwen"], s["mistral"]) for s in scores.values()
                    if s.get("qwen") is not None and s.get("mistral") is not None]

    if scored_pairs:
        qwen_scores    = [p[0] for p in scored_pairs]
        mistral_scores = [p[1] for p in scored_pairs]

        qwen_avg    = sum(qwen_scores) / len(qwen_scores)
        mistral_avg = sum(mistral_scores) / len(mistral_scores)

        qwen_wins    = sum(1 for q, m in scored_pairs if q > m)
        mistral_wins = sum(1 for q, m in scored_pairs if m > q)
        ties         = sum(1 for q, m in scored_pairs if q == m)

        qwen_3s    = sum(1 for s in qwen_scores if s == 3)
        mistral_3s = sum(1 for s in mistral_scores if s == 3)
        qwen_1s    = sum(1 for s in qwen_scores if s == 1)
        mistral_1s = sum(1 for s in mistral_scores if s == 1)

        print(f"\n  Questions scored:  {len(scored_pairs)}")
        print(f"\n  OVERALL SCORES:")
        print(f"    Qwen3-VL-4B avg:  {qwen_avg:.2f} / 3.00")
        print(f"    Mistral-7B avg:   {mistral_avg:.2f} / 3.00")
        print(f"\n  HEAD-TO-HEAD:")
        print(f"    Qwen wins:    {qwen_wins}")
        print(f"    Mistral wins: {mistral_wins}")
        print(f"    Ties:         {ties}")
        print(f"\n  PERFECT SCORES (3/3):")
        print(f"    Qwen:    {qwen_3s} / {len(scored_pairs)} ({qwen_3s/len(scored_pairs)*100:.0f}%)")
        print(f"    Mistral: {mistral_3s} / {len(scored_pairs)} ({mistral_3s/len(scored_pairs)*100:.0f}%)")
        print(f"\n  HALLUCINATIONS / WRONG (1/3):")
        print(f"    Qwen:    {qwen_1s} / {len(scored_pairs)} ({qwen_1s/len(scored_pairs)*100:.0f}%)")
        print(f"    Mistral: {mistral_1s} / {len(scored_pairs)} ({mistral_1s/len(scored_pairs)*100:.0f}%)")

        # By category
        categories = sorted(set(s["category"] for s in scores.values()))
        print(f"\n  BY CATEGORY:")
        print(f"    {'Category':<38} {'Qwen avg':>8} {'Mistral avg':>11}")
        print(f"    {'-'*38} {'-'*8} {'-'*11}")
        for cat in categories:
            cat_scores = [(s["qwen"], s["mistral"]) for s in scores.values()
                          if s["category"] == cat and s.get("qwen") is not None]
            if cat_scores:
                qa = sum(p[0] for p in cat_scores) / len(cat_scores)
                ma = sum(p[1] for p in cat_scores) / len(cat_scores)
                winner = "Qwen" if qa > ma else ("Mistral" if ma > qa else "Tie")
                print(f"    {cat:<38} {qa:>8.2f} {ma:>11.2f}  ({winner})")

    print(f"\n  Scores saved to: {SCORES_FILE}")
    print(f"\n  Run analyze_eval.py to generate the full trade-off report.\n")


if __name__ == "__main__":
    main()
