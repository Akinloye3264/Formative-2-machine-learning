"""Voice verification model (Task 3). Run from the repository root:

    python src/preprocess_audio.py          # features (+ ECAPA embeddings) and splits
    python src/train_voice_model.py --validation-only   # compare methods, no test data
    python src/train_voice_model.py         # compare, choose, then test ONCE and save

What it does (see src/voice_verification.py for the methods and the rule):
  1. enrols the four members from the TRAIN split (take 1 + its augmentations);
  2. tunes every candidate method on the VALIDATION split only
     (members' take 2 + the two validation outsiders, all with augmentations);
  3. picks the method with the lowest validation FRR under FAR <= 5% and
     mismatch <= 5% (ties: lower FAR + mismatch, then lower EER);
  4. evaluates ONLY the chosen method, once, on the TEST split (members' take 3
     + the two test outsiders, who were never seen before);
  5. saves models/voice_model.joblib (the tested model, unchanged),
     outputs/metrics/voice_metrics.json, outputs/metrics/voice_verification_trials.csv
     and outputs/figures/voice_far_frr_threshold.png.

Outsiders never train anything: validation outsiders only set the threshold,
test outsiders only measure it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import voice_verification as vv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "voice_model.joblib"
METRICS_PATH = ROOT / "outputs" / "metrics" / "voice_metrics.json"
TRIALS_PATH = ROOT / "outputs" / "metrics" / "voice_verification_trials.csv"
FIGURE_PATH = ROOT / "outputs" / "figures" / "voice_far_frr_threshold.png"

THRESHOLD_RULE = (f"Lowest validation FRR subject to outsider FAR <= {vv.MAX_FAR:.0%} and "
                  f"mismatch rate <= {vv.MAX_MISMATCH:.0%}; middle of the tied threshold range; "
                  "equal-error-rate threshold if no setting meets both limits. "
                  "Tuned on validation clips (originals + augmentations) only.")
SELECTION_RULE = ("Method chosen on validation only: meets both limits first, then lowest FRR, "
                  "then lowest FAR + mismatch, then lowest EER.")


# ----------------------------------------------------------------------------- data
def load_data(features_csv, unauth_csv):
    members = pd.read_csv(features_csv)
    outsiders = pd.read_csv(unauth_csv)
    has_emb = bool(vv.embedding_columns(members)) and bool(vv.embedding_columns(outsiders))
    if not has_emb:
        print("WARNING: no ecapa_* columns in the feature CSVs (SpeechBrain was not installed when "
              "src/preprocess_audio.py ran), so method C (ECAPA) is skipped and the result will "
              "differ from the committed model.")
    split = lambda df, s: df[df["split"] == s].reset_index(drop=True)  # noqa: E731
    data = {s: (split(members, s), split(outsiders, s)) for s in ("train", "validation", "test")}
    if not data["train"][1].empty:
        raise SystemExit("Outsiders must never be in the train split.")
    return data, members, has_emb


def candidates(members: pd.DataFrame, has_emb: bool) -> list[vv.Candidate]:
    hand = vv.handcrafted_columns(members)
    out = [vv.Candidate("A  logreg + confidence + distance (previous method, retuned)",
                        "logreg_conf_dist", hand),
           vv.Candidate("B1 cosine to centroid, all hand-crafted features", "cosine_centroid", hand),
           vv.Candidate("B2 cosine to centroid, MFCC features only", "cosine_centroid",
                        vv.handcrafted_columns(members, mfcc_only=True))]
    if has_emb:
        out.append(vv.Candidate("C  cosine to centroid, ECAPA speaker embedding",
                                "ecapa_cosine", vv.embedding_columns(members)))
    return out


def originals(df):
    return df[df["augmentation"] == "original"].reset_index(drop=True)


def summarise(res: dict) -> dict:
    keep = ["far", "far_errors", "far_trials", "frr", "frr_errors", "frr_trials",
            "mismatch_rate", "mismatch_rate_errors", "mismatch_rate_trials",
            "accuracy", "f1_weighted", "f1_macro", "log_loss", "clips"]
    out = {k: res[k] for k in keep}
    for m in ("far", "frr", "mismatch_rate"):
        out[f"{m}_95ci"] = vv.clopper_pearson(res[f"{m}_errors"], res[f"{m}_trials"])
    return out


# ----------------------------------------------------------------------------- selection
def compare_on_validation(data, cands):
    (train_m, _), (val_m, val_o) = data["train"], data["validation"]
    rows, fitted = [], {}
    for c in cands:
        bundle = vv.fit(c, train_m)
        both, trials = vv.make_trials(val_m, val_o, bundle["classes"])
        S = vv.raw_scores(bundle, both)
        params, grid = vv.tune(bundle, S, trials)
        bundle["params"] = params
        all_clips, _ = vv.evaluate(bundle, val_m, val_o)
        orig, _ = vv.evaluate(bundle, originals(val_m), originals(val_o))
        e = vv.eer(bundle, S, trials)
        fitted[c.name] = (bundle, grid)
        rows.append({"candidate": c.name, "method": c.method, "n_features": len(c.features),
                     **{k: v for k, v in params.items()}, "val_eer": e,
                     "val_far": all_clips["far"], "val_frr": all_clips["frr"],
                     "val_mismatch": all_clips["mismatch_rate"],
                     "val_far_originals": orig["far"], "val_frr_originals": orig["frr"],
                     "val_mismatch_originals": orig["mismatch_rate"],
                     "val_id_accuracy": all_clips["accuracy"],
                     "val_log_loss": all_clips["log_loss"]})
    table = pd.DataFrame(rows)
    order = table.assign(_infeasible=~table.meets_limits,
                         _imp=table.val_far + table.val_mismatch).sort_values(
        ["_infeasible", "val_frr", "_imp", "val_eer"], kind="stable")
    chosen = order.iloc[0]["candidate"]
    return table, fitted, chosen


# ----------------------------------------------------------------------------- figure
def sweep(bundle, members, outsiders, thresholds):
    both, trials = vv.make_trials(members, outsiders, bundle["classes"])
    S = vv.raw_scores(bundle, both)
    rows, claims = trials.row.to_numpy(), trials.claim_idx.to_numpy()
    out = []
    for t in thresholds:
        p = {**bundle["params"], "threshold": float(t)}
        out.append({"threshold": t, **vv.rates(trials, vv.decide(bundle, S, rows, claims, p))})
    return pd.DataFrame(out)


def plot(bundle, data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = [("far", "FAR: outsider accepted", "#2a78d6"),
              ("frr", "FRR: member rejected", "#eb6834"),
              ("mismatch_rate", "Mismatch: other member accepted", "#1baf7a")]
    t0 = bundle["params"]["threshold"]
    if bundle["method"] == "logreg_conf_dist":
        ts, xlabel = np.round(np.arange(0.25, 1.0, 0.01), 2), "Confidence threshold"
    else:
        ts, xlabel = np.round(np.arange(-0.2, 1.0, 0.005), 3), "Cosine-similarity threshold"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, facecolor=surface)
    for ax, split, title in [(axes[0], "validation", "Validation (threshold chosen here)"),
                             (axes[1], "test", "Test (threshold fixed, evaluated once)")]:
        curve = sweep(bundle, *data[split], ts)
        ax.set_facecolor(surface)
        for key, label, color in series:
            ax.plot(curve.threshold, curve[key] * 100, color=color, lw=2, label=label)
        ax.axhline(vv.MAX_FAR * 100, color=muted, lw=1, ls=":")
        ax.text(ts[0], vv.MAX_FAR * 100 + 1.5, "5% limit", color=muted, fontsize=9)
        ax.axvline(t0, color=ink, lw=1.2, ls="--")
        ax.text(t0, 101, f" chosen {t0:.3g}", color=ink, fontsize=9, va="bottom")
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=16)
        ax.set_xlabel(xlabel, color=muted)
        ax.grid(color=grid_c, lw=0.8)
        ax.tick_params(colors=muted)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_ylim(-2, 100)
    axes[0].set_ylabel("Error rate (%)", color=muted)
    axes[1].legend(loc="center right", frameon=False, labelcolor=ink, fontsize=9)
    fig.suptitle(f"Voice verification error rates vs threshold ({bundle['candidate'][:2].strip()}: "
                 f"{bundle['method']}; originals + augmentations)", color=ink, fontsize=12, x=0.01,
                 ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=surface)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap_ = argparse.ArgumentParser(description="Train, tune and test the voice verification model")
    ap_.add_argument("--features", default=PROCESSED / "audio_features.csv", type=Path)
    ap_.add_argument("--unauthorized", default=PROCESSED / "audio_features_unauthorized.csv", type=Path)
    ap_.add_argument("--validation-only", action="store_true",
                     help="compare methods on validation and stop (never touches test data)")
    args = ap_.parse_args()

    data, members, has_emb = load_data(args.features, args.unauthorized)
    for s, (m, o) in data.items():
        print(f"{s:>10}: {m.source_file.nunique():2d} member recordings ({len(m)} clips), "
              f"{o.speaker_id.nunique() if len(o) else 0} outsiders ({len(o)} clips)")

    table, fitted, chosen = compare_on_validation(data, candidates(members, has_emb))
    pd.set_option("display.width", 200)
    show = ["candidate", "threshold", "distance_factor", "require_top1", "meets_limits",
            "val_far", "val_frr", "val_mismatch", "val_eer",
            "val_far_originals", "val_frr_originals", "val_mismatch_originals"]
    print("\nVALIDATION comparison (all clips; *_originals = real recordings only):")
    print(table[[c for c in show if c in table]].round(3).to_string(index=False))
    print(f"\nChosen on validation: {chosen}")
    if args.validation_only:
        return

    bundle, _ = fitted[chosen]
    # ---- the single look at the test split ----
    test_all, trials_all = vv.evaluate(bundle, *data["test"])
    test_orig, _ = vv.evaluate(bundle, *map(originals, data["test"]))
    val_all, val_trials = vv.evaluate(bundle, *data["validation"])
    val_orig, _ = vv.evaluate(bundle, *map(originals, data["validation"]))

    n = lambda s: {"member_recordings": int(data[s][0].source_file.nunique()),  # noqa: E731
                   "member_clips": int(len(data[s][0])),
                   "outsider_speakers": sorted(data[s][1].speaker_id.unique().tolist()),
                   "outsider_clips": int(len(data[s][1]))}
    metrics = {
        # top-level keys kept for notebooks/04_model_evaluation.ipynb (TEST, real recordings)
        "model_type": f"{bundle['method']} ({chosen[:2].strip()})",
        "accuracy": test_orig["accuracy"], "f1_weighted": test_orig["f1_weighted"],
        "log_loss": test_orig["log_loss"],
        "train_rows": int(len(data["train"][0])), "test_rows": int(test_orig["clips"]),
        "verification": {
            "method": bundle["method"], "candidate": chosen,
            "params": bundle["params"], "threshold_rule": THRESHOLD_RULE,
            "selection_rule": SELECTION_RULE,
            "identification_metrics_note": ("accuracy / F1 / log loss: logistic regression "
                                            "identifying which member is speaking, on the "
                                            "chosen method's features"),
        },
        "splits": {"rule": ("members: take 1 train, take 2 validation, take 3 test; outsiders "
                            "split by speaker (seed 42); augmentations follow their original"),
                   **{s: n(s) for s in ("train", "validation", "test")}},
        "validation": {"originals": summarise(val_orig), "with_augmentations": summarise(val_all)},
        "test": {"originals": summarise(test_orig), "with_augmentations": summarise(test_all)},
        "method_comparison_validation": json.loads(table.to_json(orient="records")),
        "previous_model": {
            "description": ("logistic regression, confidence >= 0.6, distance factor 2.0, no "
                            "validation set; notebook 05 leave-one-member-out test"),
            "unknown_voice_far": 0.594, "trials": 32},
    }
    bundle.update(metrics={k: metrics[k] for k in ("accuracy", "f1_weighted", "log_loss")},
                  threshold=bundle["params"]["threshold"])

    MODEL_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    pd.concat([val_trials.assign(split="validation"), trials_all.assign(split="test")]) \
        .drop(columns=["row", "claim_idx"]).round(4).to_csv(TRIALS_PATH, index=False)
    plot(bundle, data, FIGURE_PATH)

    print("\nTEST (evaluated once)        FAR      FRR      mismatch   acc    F1     log loss")
    for label, r in [("validation, originals", val_orig), ("test, originals", test_orig),
                     ("test, + augmentations", test_all)]:
        print(f"{label:24s} {r['far']:6.1%} ({r['far_errors']}/{r['far_trials']})  "
              f"{r['frr']:6.1%} ({r['frr_errors']}/{r['frr_trials']})  "
              f"{r['mismatch_rate']:6.1%} ({r['mismatch_rate_errors']}/{r['mismatch_rate_trials']})  "
              f"{r['accuracy']:.3f}  {r['f1_weighted']:.3f}  {r['log_loss']:.3f}")
    for p in (MODEL_PATH, METRICS_PATH, TRIALS_PATH, FIGURE_PATH):
        print("Saved", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
