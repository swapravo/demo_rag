"""
evaluate.py — RAG Chunk-Size Evaluation Harness
================================================
Tests chunk sizes [125, 150, 175, 250] against all 28 golden Q&A pairs.

Usage:
    python evaluate.py

Outputs:
    • Printed summary table to stdout
    • eval_results.json  — full per-question results for all chunk sizes
"""

import json
import os
import re
import textwrap
from datetime import datetime

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Configuration ────────────────────────────────────────────────────────────

CHUNK_SIZES      = [125, 150, 175, 250]
KNOWLEDGE_FILE   = "knowledge.txt"
GOLDEN_FILE      = "golden.txt"
RESULTS_FILE     = "eval_results.json"
EMBED_MODEL      = "text-embedding-3-small"
ANSWER_MODEL     = "gpt-4.1-mini"
JUDGE_MODEL      = "gpt-4.1-mini"
TOP_K            = 3          # chunks retrieved per question (matches step5/step6)


# ─── 1. Parse Golden Dataset ──────────────────────────────────────────────────

def parse_golden(path: str) -> list[dict]:
    """Parse golden.txt into a list of Q&A dicts.

    File structure per question:
        ────────────────────────────────────────────────  (separator)
        ID: Qxxx
        ────────────────────────────────────────────────  (separator)
        question: ...multi-line...
        expected_answer: ...
        source_section(s): ...
        answer_type: ...
        chunk_sensitivity: ...multi-line...
                                                          (blank line)
    """
    SEPARATOR = "-" * 80
    KNOWN_FIELDS = {"question", "expected_answer", "source_section(s)", "answer_type", "chunk_sensitivity"}

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    questions = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        # Detect an ID line that sits between two separator lines
        if line.startswith("ID:") and i > 0 and lines[i - 1].rstrip("\n") == SEPARATOR:
            qid = line[len("ID:"):].strip()

            # Skip the closing separator line (i+1)
            i += 2  # now pointing at first field line

            # Collect field lines until the next separator or EOF
            field_lines = []
            while i < len(lines) and lines[i].rstrip("\n") != SEPARATOR:
                field_lines.append(lines[i].rstrip("\n"))
                i += 1

            # Parse key: value pairs (values may span multiple lines)
            fields: dict[str, str] = {}
            current_key: str | None = None
            current_val: list[str] = []

            for fline in field_lines:
                # Check if line starts a new known field
                matched_field = None
                for kf in KNOWN_FIELDS:
                    if fline.startswith(kf + ":"):
                        matched_field = kf
                        break

                if matched_field:
                    # Save previous field
                    if current_key:
                        fields[current_key] = " ".join(current_val).strip()
                    current_key = matched_field
                    current_val = [fline[len(matched_field) + 1:].strip()]
                elif current_key and fline.strip():
                    current_val.append(fline.strip())

            # Save last field
            if current_key:
                fields[current_key] = " ".join(current_val).strip()

            sens_raw = fields.get("chunk_sensitivity", "")
            # Take only the label before the em-dash explanation
            sens = sens_raw.split("—")[0].split("–")[0].strip()

            questions.append({
                "id":                qid,
                "question":          fields.get("question", ""),
                "expected_answer":   fields.get("expected_answer", ""),
                "source_sections":   fields.get("source_section(s)", ""),
                "answer_type":       fields.get("answer_type", ""),
                "chunk_sensitivity": sens,
            })
        else:
            i += 1

    return questions


# ─── 2. Build In-Memory FAISS Index ───────────────────────────────────────────

def build_index(chunks: list[str]) -> tuple:
    """Embed chunks and return (faiss_index, embeddings_array)."""
    response = client.embeddings.create(model=EMBED_MODEL, input=chunks)
    embeddings = np.array(
        [item.embedding for item in response.data], dtype="float32"
    )
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


# ─── 3. Retrieve & Answer ─────────────────────────────────────────────────────

def retrieve_and_answer(
    question: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
) -> tuple[str, list[str]]:
    """Retrieve top-k chunks and generate an answer (mirrors step6_generate_answer.py)."""
    q_resp = client.embeddings.create(model=EMBED_MODEL, input=question)
    q_vec  = np.array([q_resp.data[0].embedding], dtype="float32")

    distances, indices = index.search(q_vec, TOP_K)
    retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]

    context = "\n\n".join(retrieved)
    prompt  = f"""Answer ONLY using the information below.

Context:

{context}

Question:

{question}"""

    answer_resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return answer_resp.choices[0].message.content.strip(), retrieved


# ─── 4. LLM Judge ─────────────────────────────────────────────────────────────

def judge(question: str, expected: str, generated: str) -> str:
    """Score generated answer as PASS / PARTIAL / FAIL using an LLM judge."""
    prompt = textwrap.dedent(f"""
        You are an objective evaluator for a RAG (Retrieval-Augmented Generation) system.

        Question: {question}

        Reference Answer (ground truth): {expected}

        Generated Answer: {generated}

        Evaluate the generated answer strictly against the reference answer.
        Respond with EXACTLY one of: PASS, PARTIAL, or FAIL — on its own line, with no other text.

        Scoring criteria:
        - PASS   : The generated answer is factually correct and covers all key points in the reference.
        - PARTIAL: The generated answer is partially correct — it contains some key facts but misses or
                   misstates others.
        - FAIL   : The generated answer is incorrect, irrelevant, or fabricates facts not in the reference.

        Special cases:
        - If the reference says the answer is "not stated" / "trick question" and the generated answer
          correctly says it is not in the policy, that is PASS.
        - If the reference says "not stated" but the generated answer fabricates a number, that is FAIL.

        Your verdict (PASS / PARTIAL / FAIL):
    """).strip()

    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    verdict = resp.choices[0].message.content.strip().upper()
    # Normalise in case the model adds punctuation
    for label in ("PASS", "PARTIAL", "FAIL"):
        if label in verdict:
            return label
    return "FAIL"  # fallback


# ─── 5. Score Aggregation ─────────────────────────────────────────────────────

SCORE_MAP = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}


def aggregate(results: list[dict]) -> dict:
    """Compute accuracy metrics from a list of per-question result dicts."""
    total   = len(results)
    counts  = {"PASS": 0, "PARTIAL": 0, "FAIL": 0}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    weighted = sum(SCORE_MAP[r["verdict"]] for r in results) / total

    # By answer_type
    by_type: dict[str, dict] = {}
    for r in results:
        at = r["answer_type"].split("/")[0].strip()  # normalise compound types
        by_type.setdefault(at, []).append(r["verdict"])

    type_summary = {
        at: {
            "n":        len(verdicts),
            "PASS":     verdicts.count("PASS"),
            "PARTIAL":  verdicts.count("PARTIAL"),
            "FAIL":     verdicts.count("FAIL"),
            "weighted": sum(SCORE_MAP[v] for v in verdicts) / len(verdicts),
        }
        for at, verdicts in by_type.items()
    }

    # By chunk_sensitivity (first word, guard against empty)
    by_sens: dict[str, dict] = {}
    for r in results:
        raw = r["chunk_sensitivity"].strip()
        sens = (raw.split()[0].rstrip("-").upper()) if raw else "UNKNOWN"
        by_sens.setdefault(sens, []).append(r["verdict"])

    sens_summary = {
        s: {
            "n":        len(verdicts),
            "PASS":     verdicts.count("PASS"),
            "PARTIAL":  verdicts.count("PARTIAL"),
            "FAIL":     verdicts.count("FAIL"),
            "weighted": sum(SCORE_MAP[v] for v in verdicts) / len(verdicts),
        }
        for s, verdicts in by_sens.items()
    }

    return {
        "total":          total,
        "counts":         counts,
        "pass_pct":       round(counts["PASS"]    / total * 100, 1),
        "partial_pct":    round(counts["PARTIAL"] / total * 100, 1),
        "fail_pct":       round(counts["FAIL"]    / total * 100, 1),
        "weighted_score": round(weighted * 100, 1),
        "by_answer_type": type_summary,
        "by_sensitivity": sens_summary,
    }


# ─── 6. Pretty-Print Summary ──────────────────────────────────────────────────

def print_summary(all_summaries: dict[int, dict], all_results: dict[int, list]):
    """Print a formatted comparison table to stdout."""
    W = 80
    print("\n" + "=" * W)
    print("  RAG CHUNK-SIZE EVALUATION RESULTS".center(W))
    print("=" * W)

    # ── Overall accuracy table ────────────────────────────────────────────────
    print(f"\n{'Chunk':>8}  {'PASS':>6}  {'PARTIAL':>8}  {'FAIL':>6}  {'Weighted%':>10}")
    print("-" * 45)
    for cs, s in all_summaries.items():
        print(
            f"{cs:>8}  {s['pass_pct']:>5.1f}%  "
            f"{s['partial_pct']:>7.1f}%  "
            f"{s['fail_pct']:>5.1f}%  "
            f"{s['weighted_score']:>9.1f}%"
        )

    # ── Per-question breakdown for each chunk size ────────────────────────────
    for cs, results in all_results.items():
        print(f"\n{'─'*W}")
        print(f"  Chunk size {cs} — Per-question breakdown")
        print(f"{'─'*W}")
        print(f"  {'ID':<6}  {'Verdict':<8}  {'Type':<18}  {'Sensitivity':<12}  Question (truncated)")
        print(f"  {'-'*6}  {'-'*8}  {'-'*18}  {'-'*12}  {'-'*30}")
        for r in results:
            verdict_icon = {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗"}.get(r["verdict"], "?")
            q_trunc = r["question"][:40].replace("\n", " ")
            print(
                f"  {r['id']:<6}  "
                f"{verdict_icon} {r['verdict']:<6}  "
                f"{r['answer_type'][:18]:<18}  "
                f"{r['chunk_sensitivity'][:12]:<12}  "
                f"{q_trunc}..."
            )

    # ── By answer type ────────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  Weighted Score by Answer Type")
    print(f"{'─'*W}")
    all_types = sorted(
        {t for s in all_summaries.values() for t in s["by_answer_type"]}
    )
    header = f"  {'Answer Type':<22}" + "".join(f"{cs:>9}" for cs in all_summaries)
    print(header)
    print(f"  {'-'*22}" + "".join(f"  {'------':>7}" for _ in all_summaries))
    for at in all_types:
        row = f"  {at:<22}"
        for s in all_summaries.values():
            val = s["by_answer_type"].get(at, {}).get("weighted", None)
            row += f"  {val*100:>6.1f}%" if val is not None else f"  {'N/A':>7}"
        print(row)

    # ── By chunk sensitivity ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print("  Weighted Score by Chunk Sensitivity")
    print(f"{'─'*W}")
    all_sens = sorted(
        {s for sm in all_summaries.values() for s in sm["by_sensitivity"]}
    )
    header = f"  {'Sensitivity':<14}" + "".join(f"{cs:>9}" for cs in all_summaries)
    print(header)
    print(f"  {'-'*14}" + "".join(f"  {'------':>7}" for _ in all_summaries))
    for sens in all_sens:
        row = f"  {sens:<14}"
        for sm in all_summaries.values():
            val = sm["by_sensitivity"].get(sens, {}).get("weighted", None)
            row += f"  {val*100:>6.1f}%" if val is not None else f"  {'N/A':>7}"
        print(row)

    print("\n" + "=" * W)


# ─── 7. Main ──────────────────────────────────────────────────────────────────

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Parsing golden dataset...")
    golden = parse_golden(GOLDEN_FILE)
    print(f"  -> {len(golden)} questions loaded")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading knowledge base...")
    with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    all_results:   dict[int, list] = {}
    all_summaries: dict[int, dict] = {}

    for chunk_size in CHUNK_SIZES:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] -- Chunk size: {chunk_size} --")

        # 7a. Chunk
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"  Chunks created: {len(chunks)}")

        # 7b. Embed & index
        print(f"  Embedding {len(chunks)} chunks...")
        index, _ = build_index(chunks)
        print(f"  Index built ({index.ntotal} vectors)")

        # 7c. Evaluate each question
        results = []
        for qi, qa in enumerate(golden, 1):
            print(f"  [{qi:>2}/{len(golden)}] {qa['id']}  {qa['question'][:55].replace(chr(10),' ')}...")

            generated, retrieved = retrieve_and_answer(qa["question"], index, chunks)
            verdict = judge(qa["question"], qa["expected_answer"], generated)

            icon = {"PASS": "v", "PARTIAL": "~", "FAIL": "x"}.get(verdict, "?")
            print(f"         -> [{icon}] {verdict}")

            results.append({
                "chunk_size":        chunk_size,
                "id":                qa["id"],
                "question":          qa["question"],
                "expected_answer":   qa["expected_answer"],
                "generated_answer":  generated,
                "retrieved_chunks":  retrieved,
                "verdict":           verdict,
                "answer_type":       qa["answer_type"],
                "chunk_sensitivity": qa["chunk_sensitivity"],
                "source_sections":   qa["source_sections"],
            })

        all_results[chunk_size]   = results
        all_summaries[chunk_size] = aggregate(results)
        s = all_summaries[chunk_size]
        print(
            f"  Summary: PASS={s['counts']['PASS']}  "
            f"PARTIAL={s['counts']['PARTIAL']}  "
            f"FAIL={s['counts']['FAIL']}  "
            f"Weighted={s['weighted_score']}%"
        )

    # 7d. Save JSON
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "chunk_sizes":   CHUNK_SIZES,
        "num_questions": len(golden),
        "summaries":     {str(cs): all_summaries[cs] for cs in CHUNK_SIZES},
        "results":       {str(cs): all_results[cs]   for cs in CHUNK_SIZES},
    }
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Results saved -> {RESULTS_FILE}")

    # 7e. Print summary
    print_summary(all_summaries, all_results)


if __name__ == "__main__":
    main()
