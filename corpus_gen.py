"""
Generates SPIEC-300-v2: 300 synthetic interactions (100/domain), each with a
PRE-SELECTED ground-truth gap category (so the label is correct by
construction, not by post-hoc LLM judgment of its own output -- see
manuscript methodology note). Checkpointed to JSONL so a long run can be
stopped/resumed.

Run: python3 corpus_gen.py --domain teaching --n 100
     python3 corpus_gen.py --domain healthcare --n 100
     python3 corpus_gen.py --domain customer_service --n 100
   or: python3 corpus_gen.py --all   (writes corpus.jsonl)
"""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from taxonomy import categories_for_domain, DOMAINS
from prompts import CORPUS_GENERATION_SYSTEM, CORPUS_GENERATION_USER_TEMPLATE
from llm_client import MOCK_MODE

OUT_PATH = Path("corpus.jsonl")
ROLES = {
    "teaching": ("teacher", "students"),
    "healthcare": ("physician", "patient"),
    "customer_service": ("customer service agent", "customer"),
}
SEVERITIES = ["low", "moderate", "high"]

# 14-dimensional structured multimodal feature vector (paper Sec. 5.1.2).
# These are SYNTHETIC Percept-layer stub features -- the Percept Layer is not
# implemented, so these are deterministic stand-ins derived from the record id
# and target severity, NOT real sensor readings. Disclosed as such in the
# manuscript. They are NOT a decodable function of the gap category, so they do
# not leak the ground-truth label; higher-severity interactions carry mildly
# more "stress/disengagement"-loaded feature values.
FEATURE_DIMS = [
    "speech_rate_norm", "speech_volume_var", "prosody_range", "pause_ratio",
    "gaze_on_task_ratio", "facial_affect_positive", "facial_affect_negative",
    "gesture_rate", "motion_restlessness", "interpersonal_distance",
    "turn_taking_balance", "interruption_rate", "ambient_noise_norm",
    "biometric_arousal",
]
_SEVERITY_LOAD = {"low": 0.06, "moderate": 0.13, "high": 0.20}
# Dims where a higher value indicates a WORSE / more-degraded interaction, so
# severity should push them up; the rest are pushed down by severity.
_NEGATIVE_DIMS = {
    "speech_volume_var", "pause_ratio", "facial_affect_negative",
    "motion_restlessness", "interruption_rate", "ambient_noise_norm",
    "biometric_arousal",
}


def compute_feature_vector(interaction_id: str, severity: str) -> dict:
    """Deterministic synthetic 14-dim Percept-stub vector in [0,1] per dim.
    Stable across runs/machines (sha256, not hash())."""
    load = _SEVERITY_LOAD.get(severity, 0.35)
    vec = {}
    for dim in FEATURE_DIMS:
        h = hashlib.sha256(f"{interaction_id}::{dim}".encode()).hexdigest()
        base = int(h[:8], 16) / 0xFFFFFFFF  # base noise in [0,1]
        shift = load if dim in _NEGATIVE_DIMS else -load
        vec[dim] = round(min(1.0, max(0.0, 0.5 + shift + 0.4 * (base - 0.5))), 3)
    return vec


def _load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["interaction_id"])
    return ids


def generate_transcript_raw(domain: str, idx: int, seed: int) -> dict:
    """Generates one transcript. Transcripts are free text (not JSON), so this
    calls the plain chat completion endpoint directly rather than the
    JSON-mode `chat_json` helper (that one is used by the condition runners,
    which do need structured output). Falls back to a clearly-marked mock
    transcript when MOCK_MODE is on (no API key set)."""
    rng = random.Random(seed)
    cats = categories_for_domain(domain)
    cat = cats[idx % len(cats)]
    severity = SEVERITIES[rng.randrange(len(SEVERITIES))]
    role, counterpart = ROLES[domain]
    target_words = max(200, min(900, int(rng.gauss(500, 120))))
    domain_example = cat.domain_examples.get(domain, cat.description)
    user_prompt = CORPUS_GENERATION_USER_TEMPLATE.format(
        domain=domain, gap_id=cat.id, gap_name=cat.name, gap_description=cat.description,
        domain_example=domain_example, severity=severity, role=role, counterpart=counterpart,
        target_words=target_words,
    )

    if MOCK_MODE:
        transcript = (f"[MOCK TRANSCRIPT domain={domain} gap={cat.id}:{cat.name} "
                       f"severity={severity}] Lorem ipsum interaction text placeholder "
                       f"used only to validate the pipeline offline.")
    else:
        from llm_client import _client, MODEL, _token_param_name  # local import: only exists in real mode
        kwargs = {
            "model": MODEL, "temperature": 0.8, _token_param_name(MODEL): 1200,
            "messages": [{"role": "system", "content": CORPUS_GENERATION_SYSTEM},
                         {"role": "user", "content": user_prompt}],
        }
        resp = _client.chat.completions.create(**kwargs)
        transcript = resp.choices[0].message.content

    interaction_id = f"{domain}::{idx:04d}"
    return {
        "interaction_id": interaction_id,
        "domain": domain,
        "role": role,
        "counterpart": counterpart,
        "gap_category_id": cat.id,
        "gap_category_name": cat.name,
        "meta_class": cat.meta_class,
        "severity": severity,
        "target_words": target_words,
        "transcript": transcript,
        "feature_vector": compute_feature_vector(interaction_id, severity),
        "mock": MOCK_MODE,
    }


def run(domain: str, n: int, out_path: Path = OUT_PATH, seed_base: int = 1337) -> None:
    existing = _load_existing_ids(out_path)
    with out_path.open("a") as f:
        for i in range(n):
            iid = f"{domain}::{i:04d}"
            if iid in existing:
                continue
            record = generate_transcript_raw(domain, i, seed=seed_base + i)
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  wrote {iid}  ({'MOCK' if record['mock'] else 'REAL'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.all:
        for d in DOMAINS:
            print(f"Generating {d} ...")
            run(d, args.n, out_path)
    else:
        if not args.domain:
            ap.error("--domain required unless --all")
        run(args.domain, args.n, out_path)

    print(f"Done. Total records in {out_path}: {sum(1 for _ in out_path.open())}")


if __name__ == "__main__":
    main()
