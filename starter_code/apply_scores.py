"""
Fix JSON syntax error at Q13 and apply scores for Q14-100.
Preserves all human scores for Q1-13.
"""

import json
from pathlib import Path

PATH = Path(__file__).parent / "eval_responses.json"

# Fix the syntax error by reading raw text and patching it
raw = PATH.read_text()

# Fix missing comma after Q13 Mistral score
# The broken part is:  "score": 2\n        "notes"
# It should be:        "score": 2,\n        "notes"
raw = raw.replace(
    '"score": 2\n        "notes": ""\n      }\n    },\n    {\n      "id": 14',
    '"score": 2,\n        "notes": ""\n      }\n    },\n    {\n      "id": 14'
)

# Parse the fixed JSON
data = json.loads(raw)

# Scores for Q14-100 based on reference answer evaluation
# Format: { id: (qwen_score, mistral_score) }
SCORES = {
    14:  (1, 0),
    15:  (2, 2),
    16:  (2, 0),
    17:  (2, 0),
    18:  (2, 2),
    19:  (3, 2),
    20:  (1, 1),
    21:  (2, 2),
    22:  (2, 1),
    23:  (3, 2),
    24:  (2, 0),
    25:  (1, 1),
    26:  (1, 1),
    27:  (1, 1),
    28:  (2, 2),
    29:  (3, 2),
    30:  (2, 1),
    31:  (2, 1),
    32:  (3, 3),
    33:  (2, 0),
    34:  (2, 1),
    35:  (2, 2),
    36:  (3, 2),
    37:  (3, 2),
    38:  (2, 0),
    39:  (2, 2),
    40:  (3, 2),
    41:  (2, 2),
    42:  (2, 2),
    43:  (3, 3),
    44:  (2, 2),
    45:  (2, 2),
    46:  (3, 3),
    47:  (3, 2),
    48:  (1, 3),
    49:  (3, 2),
    50:  (3, 3),
    51:  (2, 0),
    52:  (2, 0),
    53:  (3, 2),
    54:  (2, 1),
    55:  (2, 2),
    56:  (3, 1),
    57:  (3, 2),
    58:  (1, 2),
    59:  (2, 0),
    60:  (1, 1),
    61:  (3, 2),
    62:  (3, 0),
    63:  (3, 1),
    64:  (0, 0),
    65:  (3, 2),
    66:  (2, 1),
    67:  (3, 2),
    68:  (3, 2),
    69:  (3, 2),
    70:  (1, 0),
    71:  (3, 3),
    72:  (3, 0),
    73:  (1, 0),
    74:  (2, 0),
    75:  (3, 2),
    76:  (3, 2),
    77:  (2, 2),
    78:  (1, 1),
    79:  (2, 2),
    80:  (1, 0),
    81:  (2, 0),
    82:  (3, 2),
    83:  (2, 1),
    84:  (3, 2),
    85:  (2, 1),
    86:  (2, 0),
    87:  (2, 0),
    88:  (3, 0),
    89:  (2, 1),
    90:  (0, 1),
    91:  (3, 0),
    92:  (0, 1),
    93:  (2, 2),
    94:  (2, 0),
    95:  (0, 3),
    96:  (2, 2),
    97:  (3, 3),
    98:  (2, 0),
    99:  (0, 1),
    100: (3, 2),
}

# Apply scores
for item in data["results"]:
    qid = item["id"]
    if qid in SCORES:
        qwen_score, mistral_score = SCORES[qid]
        item["qwen"]["score"]    = qwen_score
        item["mistral"]["score"] = mistral_score

# Save
PATH.write_text(json.dumps(data, indent=2))
print("Done. Scores applied for Q14-100. JSON syntax error fixed.")

# Print summary
qwen_total   = sum(r["qwen"]["score"]    for r in data["results"] if r["qwen"]["score"]    is not None)
mistral_total= sum(r["mistral"]["score"] for r in data["results"] if r["mistral"]["score"] is not None)
max_possible = 100 * 3

print(f"\nQwen    total: {qwen_total} / {max_possible}  ({qwen_total/max_possible*100:.1f}%)")
print(f"Mistral total: {mistral_total} / {max_possible}  ({mistral_total/max_possible*100:.1f}%)")
print(f"Qwen avg per question:    {qwen_total/100:.2f}")
print(f"Mistral avg per question: {mistral_total/100:.2f}")
