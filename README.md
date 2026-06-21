# Qwen3-VL-4B vs Mistral-7B

## What This Is

A complete, self-contained package of all code, data, and results for the InferTutor Arena capstone project. The project compares two open-source LLMs — **Qwen3-VL-4B** and **Mistral-7B-Instruct-v0.3** — across two dimensions:

1. **Serving performance** — TTFT, ITL, throughput, and composite score under increasing concurrency on a single H100 GPU.
2. **Answer quality** — 100-question domain knowledge evaluation covering seven inference engineering categories, scored on a 0–3 rubric.

**Bottom line:** Qwen3-VL-4B wins on both dimensions. Deploy it at 100 concurrent users per H100.

---

## Key Results

### Serving Performance (at 100 concurrent users — sweet spot)

| Metric | Qwen3-VL-4B | Mistral-7B | Winner |
|---|---|---|---|
| TTFT p95 | 1,197 ms | 1,125 ms | Mistral (marginal) |
| ITL p95 | 5.6 ms | 7.2 ms | Qwen +22% |
| Throughput | 4,528 chunks/s | 3,960 chunks/s | Qwen +14% |
| Composite score | 67.8M | 49.0M | Qwen +38% |
| ITL at 150 users | 6.2 ms (stable) | 29.6 ms (spike) | Qwen ✓ |

### Answer Quality (100 questions, 0–3 scale)

| Metric | Qwen3-VL-4B | Mistral-7B | Winner |
|---|---|---|---|
| Overall average | 2.17 / 3.00 (72%) | 1.50 / 3.00 (50%) | Qwen +45% |
| Fully correct (score 3) | 40 / 100 | 7 / 100 | Qwen ✓ |
| Hallucinations (score 0) | 5 / 100 | 18 / 100 | Qwen ✓ |

---

## Folder Structure

```
Qwen_Mistral/
│
├── README.md                                   ← this file
├── Comparison_Report.pdf                       ← final report with quality eval (v2)
├── generate_comparison_report.py               ← generates v1 PDF from experiment data
│
└── starter_code/
    │
    ├── requirements.txt                        ← Python dependencies
    │
    │── Serving experiment scripts ──────────────────────────────────────
    ├── modal_infertutor_app.py                 ← vLLM app deployed on Modal GPU
    ├── modal_infertutor_app_generated.py       ← auto-generated Modal app variant
    ├── run_infertutor_experiment.py            ← runs a full concurrency scaling experiment
    ├── load_test_infertutor.py                 ← Locust-based load testing harness
    ├── score_infertutor.py                     ← computes composite score from results
    ├── prompts.json                            ← multiturn conversation prompts used in tests
    │
    │── Quality evaluation scripts ──────────────────────────────────────
    ├── run_eval.py                             ← sends 100 questions to both models via API
    ├── apply_scores.py                         ← applies human scores to eval_responses.json
    ├── update_report.py                        ← patches v1 PDF → v2 with Section 9 added
    │
    │── Evaluation data ─────────────────────────────────────────────────
    ├── eval_questions.json                     ← 100 inference engineering questions
    ├── eval_responses.json                     ← model responses + human scores (0–3)
    │
    ├── eval/
    │   ├── eval_set.json                       ← question set with reference answers
    │   ├── run_eval.py                         ← eval runner (eval/ version)
    │   └── score_eval.py                       ← scoring helper
    │
    └── results_infertutor/
        ├── model_comparison.json               ← aggregated comparison summary
        └── comparison/
            ├── qwen-mt-50_multiturn_50u_*.json    ← Qwen run at 50 users
            ├── qwen-mt-100_multiturn_100u_*.json  ← Qwen run at 100 users
            ├── qwen-mt-150_multiturn_150u_*.json  ← Qwen run at 150 users
            ├── qwen-mt-200_multiturn_200u_*.json  ← Qwen run at 200 users
            ├── mistral-mt-50_multiturn_50u_*.json ← Mistral run at 50 users
            ├── mistral-mt-100_multiturn_100u_*.json
            ├── mistral-mt-150_multiturn_150u_*.json
            └── mistral-mt-200_multiturn_200u_*.json
```

---

## How to Reproduce

### 1. Install dependencies

```bash
cd starter_code
pip install -r requirements.txt
pip install pypdf pdfplumber reportlab  # for PDF generation
```

### 2. Run a serving experiment (requires Modal + H100 GPU)

```bash
python run_infertutor_experiment.py \
  --model Qwen/Qwen3-VL-4B-Instruct \
  --label qwen-mt-100 \
  --gpu-type H100 \
  --replicas 1 \
  --no-fast-boot \
  --max-seqs 64 \
  --max-batch-tokens 8192 \
  --concurrent-inputs 128 \
  --mode multiturn \
  --users 100
```

Results are saved as JSON in `results_infertutor/`.

### 3. Regenerate the v1 serving performance PDF

```bash
cd ..  # root of Qwen_Mistral
python generate_comparison_report.py
# Output: InferTutor_Model_Comparison_Report.pdf (input for step 5)
```

### 4. Run the quality evaluation (requires API access to both models)

```bash
cd starter_code
python run_eval.py
# Sends all 100 questions to Qwen and Mistral, saves responses to eval_responses.json
```

Then apply human scores to `eval_responses.json` using the 0–3 rubric, then:

```bash
python apply_scores.py
```

### 5. Regenerate the v2 PDF (serving + quality combined)

```bash
cd starter_code
python update_report.py
# Reads: ../InferTutor_Model_Comparison_Report.pdf + eval_responses.json
# Output: ../Comparison_Report.pdf
```

---

## Experiment Configuration

| Setting | Value |
|---|---|
| GPU | 1 × H100 80GB per run |
| vLLM version | 0.21.0, compiled mode (torch.compile) |
| max_seqs | 64 |
| max_batch_tokens | 8,192 |
| Prefix cache | On |
| Workload mode | Multiturn (3-turn conversation history) |
| Concurrency levels | 50 / 100 / 150 / 200 users |
| Duration per run | 90 seconds |
| Scoring formula | `goodput × users / (ttft_p95_s × itl_p95_s × total_gpus)` |

---

## Production Recommendation

Deploy **Qwen3-VL-4B at 100 concurrent users per H100**.

| Metric | Measured (100u) | SLO | Alert |
|---|---|---|---|
| TTFT p95 | 1,197 ms | < 1,500 ms | > 2,000 ms |
| ITL p95 | 5.6 ms | < 10 ms | > 15 ms |
| Throughput | 4,528 chunks/s | > 3,500 chunks/s | < 2,500 chunks/s |
| Error rate | 0.0% | < 0.1% | > 0.5% |

Autoscale at 80 users. Never exceed 130 users on a single H100. For larger deployments, add H100 replicas rather than increasing concurrency per GPU.
