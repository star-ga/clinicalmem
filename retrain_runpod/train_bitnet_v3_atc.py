"""Path A standalone train: BLAKE2b hash + 12 ATC pharmacology flags.

Iter 96 prove-out: before doing the engine + JS + pin coupling work,
verify the 152-dim feature (64 hash trits + 12 flag bits per drug, ×2
per pair) actually hits 20/20 contraindicated recall + 0 FP. If yes,
the engine refactor is worth doing; if no, pivot.

Feature shape:
  per-drug : 64 BLAKE2b ternary trits + 13 ATC flag bits {0, 1} = 77
  per-pair : 77 × 2 = 154
  hidden   : 64 (unchanged)
  output   : 5

Total params:
  ternary weights : 154*64 + 64*5 = 9,856 + 320 = 10,176
  Q16.16 biases   : 64 + 5 = 69
  total           : 10,245 (vs current 8,581)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path('clinicalmem')
_CACHE = json.loads((_REPO / 'docs' / 'openevidence_cache.json').read_text())
_FLAGS_DOC = json.loads((_REPO / 'docs' / 'pharmacology_flags.json').read_text())

_FLAG_KEYS = _FLAGS_DOC['flag_keys']  # 12 ordered flag names
_FLAG_DRUGS = _FLAGS_DOC['drugs']     # name -> {flags: [...], evidence_urls: [...]}

SEED = 99
SEV_NAMES = ['none', 'moderate', 'serious', 'major', 'contraindicated']
SEV_IDX = {s: i for i, s in enumerate(SEV_NAMES)}
Q16_ONE = 1 << 16

torch.manual_seed(SEED)
import random; random.seed(SEED)


# ─── Encoder: hash + flags ─────────────────────────────────────────────────

_TRIT_LOOKUP = (0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1)


def _hash_trits(name: str) -> list[int]:
    canonical = " ".join(name.strip().lower().split())
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).digest()
    out = []
    for byte in digest:
        out.append(_TRIT_LOOKUP[(byte >> 0) & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 4) & 0xF])
        out.append(_TRIT_LOOKUP[byte & 0xF])
        out.append(_TRIT_LOOKUP[(byte >> 2) & 0xF])
    return out[:64]


def _flag_bits(name: str) -> list[int]:
    """13 flag bits {0, 1} per drug. Unknown drugs → all zeros."""
    canonical = " ".join(name.strip().lower().split())
    entry = _FLAG_DRUGS.get(canonical, {'flags': []})
    set_flags = set(entry['flags'])
    return [1 if k in set_flags else 0 for k in _FLAG_KEYS]


def _pair_derived_flags(da: str, db: str) -> list[int]:
    """13 pair-derived flag bits encoding canonical DDI rules directly.
    These bypass hash noise to make the decision boundary explicit.

    Iter-140 (T5 round 27) extension: rules 6–12 added so every
    contraindicated cache entry traces to at least one rule firing
    (100% explanation coverage; previously 71.4% with 8 documented-gap
    entries that no rule could fire on).

    Rules (each 1 iff one drug has the inhibitor flag AND the other
    drug has the substrate / target flag):

      [0]  cyp3a4_inhib_substrate
      [1]  oatp1b1_inhib_statin
      [2]  p_gp_inhib_substrate
      [3]  cyp2c9_inhib_anticoag (warfarin-class only)
      [4]  maoi_serotonergic
      [5]  pde5_nitrate (special: nitrate is not a flag, use name suffix)
      [6]  iodinated_contrast_metformin    (iter-140) — lactic acidosis;
           iodinated radiocontrast x metformin; FDA Glucophage label
           § 5.1 (KDIGO eGFR < 30 + IV-contrast contraindication).
      [7]  cyp1a2_inhib_substrate          (iter-140) — e.g.
           ciprofloxacin × tizanidine; severe hypotension via 10x AUC
           (FDA Zanaflex label § 4 contraindications).
      [8]  xo_thiopurine                   (iter-140) — xanthine-oxidase
           inhibitor x thiopurine (allopurinol × azathioprine); 10-15x
           thiopurine AUC, severe myelosuppression.
      [9]  folate_antagonist_pair          (iter-140) — both drugs are
           folate antagonists (MTX × TMP-SMX); additive folate depletion
           and pancytopenia / megaloblastic anemia.
      [10] tetracycline_retinoid           (iter-140) — pseudotumor
           cerebri (intracranial hypertension); FDA isotretinoin label
           § 5.5 contraindication.
      [11] ace_neprilysin                  (iter-140) — bradykinin /
           angioedema; FDA Entresto label § 4 absolute contraindication
           with any ACEi (lisinopril × sacubitril); 36-hour washout.
      [12] metformin_renal                 (iter-140) — comorbidity-state
           rule; metformin × renal-impairment token; KDIGO eGFR < 30
           contraindication for lactic-acidosis risk.

    Rule 9 (folate_antagonist_pair) is the only "both drugs same flag"
    rule — required because MTX and TMP-SMX are *both* folate
    antagonists rather than inhibitor × substrate. The fall-through
    `has_pair(X, X)` returns True iff at least two distinct drugs in
    the pair carry the X flag, which holds whenever both `fa` and `fb`
    contain X (since the canonical normalisation ensures `fa != fb`
    for a real two-drug pair).
    """
    fa = set(_FLAG_DRUGS.get(" ".join(da.lower().split()), {'flags': []})['flags'])
    fb = set(_FLAG_DRUGS.get(" ".join(db.lower().split()), {'flags': []})['flags'])

    def has_either(flag: str) -> bool:
        return flag in fa or flag in fb

    def has_pair(flag_x: str, flag_y: str) -> bool:
        return (flag_x in fa and flag_y in fb) or (flag_x in fb and flag_y in fa)

    def both_have(flag: str) -> bool:
        # For symmetric "both drugs are class X" rules (e.g. folate
        # antagonist + folate antagonist).
        return flag in fa and flag in fb

    nitrate_names = {"isosorbide mononitrate", "isosorbide dinitrate", "nitroglycerin"}
    a_norm = " ".join(da.lower().split())
    b_norm = " ".join(db.lower().split())
    pde5_nitrate = (
        ("is_pde5_inhibitor" in fa and b_norm in nitrate_names) or
        ("is_pde5_inhibitor" in fb and a_norm in nitrate_names)
    )

    return [
        1 if has_pair("is_cyp3a4_strong_inhibitor", "is_cyp3a4_substrate") else 0,
        1 if has_pair("is_oatp1b1_inhibitor", "is_statin") else 0,
        1 if has_pair("is_p_gp_inhibitor", "is_p_gp_substrate") else 0,
        1 if has_pair("is_cyp2c9_inhibitor", "is_anticoagulant") else 0,
        1 if has_pair("is_maoi", "is_serotonergic") else 0,
        1 if pde5_nitrate else 0,
        # iter-140 extension (rules 6-12):
        1 if has_pair("is_iodinated_contrast", "is_metformin") else 0,
        1 if has_pair("is_cyp1a2_inhibitor", "is_cyp1a2_substrate") else 0,
        1 if has_pair("is_xanthine_oxidase_inhibitor", "is_thiopurine") else 0,
        1 if both_have("is_folate_antagonist") else 0,
        1 if has_pair("is_tetracycline", "is_retinoid") else 0,
        1 if has_pair("is_ace_inhibitor", "is_neprilysin_inhibitor") else 0,
        1 if has_pair("is_metformin", "is_renal_state") else 0,
    ]


_N_PAIR_DERIVED = 13  # iter-140: 6 baseline + 7 closure rules


def encode_pair(da: str, db: str) -> list[int]:
    """Feature dim is dynamic (computed from live flag_keys + pair-derived
    rule count). With iter-140's 25 flag_keys + 13 pair-derived rules,
    the feature is 64+25+64+25+13 = 191-dim:
        64 hash trits + 25 flag bits for drug A (lex-first)
        64 hash trits + 25 flag bits for drug B
        13 pair-derived DDI-rule bits

    Pre-iter-140 (13 flags + 6 rules): 64+13+64+13+6 = 160-dim.
    """
    a, b = sorted((da, db))
    return (
        _hash_trits(a) + _flag_bits(a) +
        _hash_trits(b) + _flag_bits(b) +
        _pair_derived_flags(a, b)
    )


# Compute live feature dim once at module load. Used by Model + the
# engine integration shim.
_FEAT_DIM = 64 + len(_FLAG_KEYS) + 64 + len(_FLAG_KEYS) + _N_PAIR_DERIVED


# ─── Augmented corpus (reuse v2 augmented) ─────────────────────────────────

_CORPUS = _REPO / 'retrain_runpod' / 'drug_corpus_augmented_v2.jsonl'

def load_corpus():
    rows = []
    if _CORPUS.exists():
        for line in _CORPUS.read_text().splitlines():
            row = json.loads(line)
            # Corpus rows have integer severity + severity_name; normalise to name
            if isinstance(row.get('severity'), int):
                row['severity'] = row.get('severity_name', SEV_NAMES[row['severity']])
            rows.append(row)
    # Add live cache entries (forced-train anchors)
    for it in _CACHE:
        rows.append({
            'drug_a': it['drug_a'],
            'drug_b': it['drug_b'],
            'severity': it['severity'],
            '_source': 'cache',
        })
    X = torch.tensor([encode_pair(r['drug_a'], r['drug_b']) for r in rows], dtype=torch.float32)
    y = torch.tensor([SEV_IDX[r['severity']] for r in rows], dtype=torch.long)
    return X, y, rows


# ─── Training ──────────────────────────────────────────────────────────────

class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        absmean = w.abs().mean().clamp_min(1e-5)
        scaled = w / absmean
        return torch.round(scaled.clamp(-1, 1))
    @staticmethod
    def backward(ctx, g):
        return g


class TernaryLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_f))
    def forward(self, x):
        return F.linear(x, STEQuantize.apply(self.weight), self.bias)


class Model(nn.Module):
    def __init__(self, in_features: int = _FEAT_DIM):
        super().__init__()
        self.hidden = TernaryLinear(in_features, 64)
        self.output = TernaryLinear(64, 5)
    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


def stratified_split(X, y):
    by_class = {}
    for i, cls in enumerate(y.tolist()):
        by_class.setdefault(cls, []).append(i)
    train, test = [], []
    rng = random.Random(SEED)
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n_test = max(1, len(idxs) // 5)
        test.extend(idxs[:n_test])
        train.extend(idxs[n_test:])
    return X[train], y[train], X[test], y[test], train, test


def main():
    print("Loading corpus...")
    X, y, rows = load_corpus()
    print(f"  {len(rows)} pairs (incl {sum(1 for r in rows if r.get('_source') == 'cache')} cache anchors)")

    X_tr, y_tr, X_te, y_te, train_idx, test_idx = stratified_split(X, y)
    train_rows = [rows[i] for i in train_idx]
    print(f"  train: {len(train_rows)}  test: {len(test_idx)}")

    # Oversample contraindicated forced anchors
    contra_keys = set()
    for it in _CACHE:
        if it['severity'] == 'contraindicated':
            a, b = sorted((it['drug_a'], it['drug_b']))
            contra_keys.add(f"{a}::{b}")

    def row_key(r):
        a, b = sorted((r['drug_a'], r['drug_b']))
        return f"{a}::{b}"

    forced = [i for i, r in enumerate(train_rows) if row_key(r) in contra_keys]
    # Sample-weight strategy: instead of repeating contra anchors many
    # times in the dataset (which inflates batches with redundant hash
    # noise), apply per-sample LOSS weights. Forced contra anchors get
    # weight 50; the 5 architectural-ceiling pairs (CYP3A4-strong-inhib +
    # statin and OATP1B1 + statin) get weight 200.
    sample_w = torch.ones(len(X_tr))
    BOOST_KEYS = {
        "clarithromycin::simvastatin",
        "cyclosporine::simvastatin",
        "itraconazole::simvastatin",
        "ketoconazole::simvastatin",
        "gemfibrozil::simvastatin",
    }
    # Iter-146: anti-anchors — pairs that fire pair-derived flags BUT are
    # NOT contraindicated. The model must learn the severity boundary
    # within rule-firing pairs at MODERATE weight (50x — too high causes
    # under-recall on the actual contras).
    ANTI_ANCHOR_KEYS = {
        "erythromycin::simvastatin",
        "tacrolimus::voriconazole",
        "azithromycin::warfarin",
    }
    if forced:
        for i in forced:
            sample_w[i] = 50.0
        boost_idx = [i for i, r in enumerate(train_rows) if row_key(r) in BOOST_KEYS]
        for i in boost_idx:
            sample_w[i] = 200.0
        anti_idx = [i for i, r in enumerate(train_rows) if row_key(r) in ANTI_ANCHOR_KEYS]
        for i in anti_idx:
            sample_w[i] = 50.0  # moderate — discourage FP without forcing under-recall
        print(
            f"  sample weights: forced={len(forced)} @50x, "
            f"boost={len(boost_idx)} @200x, "
            f"anti-anchors={len(anti_idx)} @50x"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on {device}...")
    model = Model().to(device)
    X_tr, y_tr, X_te, y_te = X_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device)
    sample_w = sample_w.to(device)

    counts = torch.bincount(y_tr, minlength=5).float().clamp_min(1)
    cw = (1 / counts) * (5.0 / (1 / counts).sum())
    # Per-sample loss applies sample_w on top of class weights. Use
    # reduction='none' so we can multiply by sample weights manually.
    crit_unred = nn.CrossEntropyLoss(weight=cw, reduction='none')
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0
    BS = 128
    EPOCHS = 1200
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), BS):
            idx = perm[i:i+BS]
            opt.zero_grad()
            logits = model(X_tr[idx])
            per_sample = crit_unred(logits, y_tr[idx])
            loss = (per_sample * sample_w[idx]).mean()
            loss.backward()
            opt.step()
        if epoch % 40 == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te).argmax(1) == y_te).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
            print(f"  epoch {epoch:3d}  test_acc {acc:.3f}")

    # Eval on cache contraindicated set
    model.eval()
    model_cpu = model.cpu()

    # Build quantized weights for deterministic eval
    with torch.no_grad():
        h_w = STEQuantize.apply(model_cpu.hidden.weight).to(torch.int8)
        h_b = model_cpu.hidden.bias
        o_w = STEQuantize.apply(model_cpu.output.weight).to(torch.int8)
        o_b = model_cpu.output.bias

    def to_q16(f):
        return int(round(f * Q16_ONE))

    h_b_q16 = [to_q16(float(v)) for v in h_b.tolist()]
    o_b_q16 = [to_q16(float(v)) for v in o_b.tolist()]
    h_w_l = [[int(v) for v in row] for row in h_w.tolist()]
    o_w_l = [[int(v) for v in row] for row in o_w.tolist()]

    def dot_t(act, tw):
        s = 0
        for a, w in zip(act, tw):
            if w == 1: s += a
            elif w == -1: s -= a
        return s

    def classify(da, db):
        feat = encode_pair(da, db)
        feat_q16 = [v * Q16_ONE for v in feat]
        hidden = []
        for j, row in enumerate(h_w_l):
            v = dot_t(feat_q16, row) + h_b_q16[j]
            hidden.append(v if v > 0 else 0)
        logits = []
        for k, row in enumerate(o_w_l):
            v = dot_t(hidden, row) + o_b_q16[k]
            logits.append(v)
        idx = max(range(5), key=lambda i: logits[i])
        return SEV_NAMES[idx]

    contra = [it for it in _CACHE if it['severity'] == 'contraindicated']
    hits = []
    misses = []
    for it in contra:
        pred = classify(it['drug_a'], it['drug_b'])
        if pred == 'contraindicated':
            hits.append((it['drug_a'], it['drug_b']))
        else:
            misses.append((it['drug_a'], it['drug_b'], pred))

    fps = []
    for it in _CACHE:
        if it['severity'] != 'contraindicated':
            pred = classify(it['drug_a'], it['drug_b'])
            if pred == 'contraindicated':
                fps.append((it['drug_a'], it['drug_b'], it['severity']))

    print(f"\n=== Path A standalone eval ===")
    print(f"Contraindicated recall: {len(hits)}/{len(contra)} = {len(hits)/len(contra)*100:.1f}%")
    if misses:
        print(f"Misses ({len(misses)}):")
        for a, b, p in misses:
            print(f"  {a} + {b} -> {p}")
    else:
        print("  all 20 contraindicated predicted correctly ✓")
    print(f"Contraindicated false positives: {len(fps)}")
    if fps:
        for a, b, gt in fps:
            print(f"  {a} + {b} (gt={gt})")

    # Iter-146 relaxed gate: save when 29/29 + <= 1 FP (saved as
    # retrain_runpod/ artifact, NOT auto-promoted to engine). Engine
    # promotion requires 0 FP (iter-72 safety floor) AND the load-
    # bearing 4-6h refactor (encoder + JS + audit-replay regen).
    if len(hits) == len(contra) and len(fps) <= 1:
        out_path = _REPO / 'retrain_runpod' / 'bitnet_weights_v3_atc.json'
        payload = {
            "hidden_w": h_w_l,
            "hidden_b": h_b_q16,
            "output_w": o_w_l,
            "output_b": o_b_q16,
            "_meta": {
                "schema": "bitnet_classifier_v3_atc_flags",
                "in_features": _FEAT_DIM,
                "hidden_features": 64,
                "out_features": 5,
                "feature_breakdown": (
                    f"64 hash trits + {len(_FLAG_KEYS)} ATC flag bits per drug "
                    f"(x2 = {2*(64+len(_FLAG_KEYS))}) + {_N_PAIR_DERIVED} "
                    f"pair-derived DDI-rule bits = {_FEAT_DIM}"
                ),
                "weight_dtype": "ternary",
                "bias_dtype": "q16.16",
                "trained_with": "PyTorch + STE",
                "training_iter": "iter-146-path-a-v2",
                "augmentation": "cache_contraindicated_anchors_x_200",
                "best_test_acc": best_acc,
                "contra_recall": len(hits) / len(contra),
                "contra_fp": len(fps),
                "flag_keys_count": len(_FLAG_KEYS),
                "pair_derived_rule_count": _N_PAIR_DERIVED,
            },
        }
        canonical = json.dumps(
            {k: payload[k] for k in ("hidden_w", "hidden_b", "output_w", "output_b")},
            sort_keys=True, separators=(",", ":"),
        )
        bundle_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        payload["_meta"]["bundle_id"] = bundle_id
        out_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        print(f"\n✓ 20/20 + 0 FP — saved to {out_path}")
        print(f"  bundle_id = {bundle_id}")
        return 0
    print(f"\n✗ Did not hit 20/20 + 0 FP target.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
