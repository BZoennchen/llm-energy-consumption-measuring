import re
import json
import time
import requests
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import time
import perun

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:1b"
TEMPERATURE = 0.0

SYSTEM_PREFIX = (
    "You are a careful assistant. Answer multiple-choice questions by outputting ONLY the letter (A, B, C, or D).\n"
    "Do not include explanations.\n"
)

QUESTION_TEMPLATE = """\
Question: {question}

Choices:
A. {A}
B. {B}
C. {C}
D. {D}

Answer with a single letter: A, B, C, or D.
"""

def call_ollama(prompt, model=MODEL, temperature=TEMPERATURE, max_tokens=5):
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        },
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "").strip()

def extract_letter(text):
    # Find first standalone A/B/C/D (case-insensitive)
    m = re.search(r"\b([ABCD])\b", text.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: look for patterns like "Answer: C" or "(C)"
    m = re.search(r"[Aa]nswer[^A-D]*([ABCD])", text)
    if m:
        return m.group(1).upper()
    return None

def format_item(item):
    # Some loaders give choices as list of strings; others as dict; normalize:
    choices = item["choices"]
    if isinstance(choices, list) and len(choices) == 4:
        A, B, C, D = choices
    elif isinstance(choices, dict):
        A, B, C, D = choices["A"], choices["B"], choices["C"], choices["D"]
    else:
        # If the dataset gives only texts, adjust accordingly
        A, B, C, D = choices[0], choices[1], choices[2], choices[3]
    q = item["question"]
    return QUESTION_TEMPLATE.format(question=q, A=A, B=B, C=C, D=D)

def gold_letter(item):
    # Some configs store correct answer as letter, some as index; handle both.
    ans = item["answer"]
    if isinstance(ans, str) and ans.strip() in ["A", "B", "C", "D"]:
        return ans.strip().upper()
    # assume index 0..3
    if isinstance(ans, int):
        return ["A","B","C","D"][ans]
    # try to coerce
    try:
        idx = int(ans)
        return ["A","B","C","D"][idx]
    except Exception:
        # last resort: map string like "0"/"1"
        mapping = {"0":"A","1":"B","2":"C","3":"D"}
        return mapping.get(str(ans).strip(), None)

def eval_subject(subject_name, limit=None, sleep=0.0):
    # Load the 'test' split of a single MMLU subject
    ds = load_dataset("cais/mmlu", subject_name, split="test")
    n = len(ds) if limit is None else min(limit, len(ds))
    correct = 0
    rows = []

    for i in tqdm(range(n), desc=f"Evaluating {subject_name}"):
        item = ds[i]
        prompt = SYSTEM_PREFIX + format_item(item)
        # #(prompt)
        resp = call_ollama(prompt)
        # print(resp)
        pred = extract_letter(resp)
        gold = gold_letter(item)
        ok = (pred == gold)
        correct += int(ok)
        rows.append({
            "subject": subject_name,
            "idx": i,
            "pred": pred,
            "gold": gold,
            "correct": int(ok),
            "model_raw": resp
        })
        if sleep:
            time.sleep(sleep)

    acc = correct / n if n > 0 else 0.0
    return acc, pd.DataFrame(rows)

@perun.perun(data_out="my_results", format="json")
def main():
    # Choose a small set to start, then scale up
    subjects = [
        "high_school_computer_science"
      #  "abstract_algebra",
      #  "college_chemistry",
      #  "professional_law",
      #  "high_school_biology",
      #  "machine_learning",
    ]
    # Set limit=None to run full subject; start with 50 for a quick smoke test
    limit_per_subject = 5

    all_rows = []
    summary = []

    for s in subjects:
        acc, df = eval_subject(s, limit=limit_per_subject, sleep=0.0)
        summary.append({"subject": s, "accuracy": acc, "n": len(df)})
        all_rows.append(df)

    results_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)

    print("\nPer-subject summary:")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    overall_acc = results_df["correct"].mean()
    print(f"\nOverall accuracy (subset): {overall_acc:.3f}")

    # Optionally save
    results_df.to_csv("mmlu_ollama_results.csv", index=False)
    summary_df.to_csv("mmlu_ollama_summary.csv", index=False)
    print("\nSaved mmlu_ollama_results.csv and mmlu_ollama_summary.csv")

if __name__ == "__main__":
    main()
