"""Voice verification against a CLAIMED identity (Task 3 / voice model).

In the app the face model first says who the person claims to be (e.g.
member02). The voice check then answers one question: "is this voice
member02?" -- not "which of the four members does this sound most like?".

Three scoring methods are compared (the choice is made on validation data only):

  logreg_conf_dist   The previous method. Logistic regression over the four
                     members; accept when the claimed member is the top guess,
                     its probability >= t, and the clip is within
                     f x (largest training distance) of that member's centroid.
  cosine_centroid    Cosine similarity between the clip and the claimed
                     member's enrolment centroid, on standardised hand-crafted
                     features (all of extract_features, or MFCCs only).
  ecapa_cosine       Same cosine rule on a pretrained ECAPA-TDNN speaker
                     embedding (speechbrain/spkrec-ecapa-voxceleb).

For the cosine methods an optional extra rule ("top-1") also requires the
claimed member to be the best-matching member; it is kept only if it helps
on validation.

Trials (each clip is tried against claimed identities):
  genuine   a member's clip claiming their own identity   -> FRR = share rejected
  mismatch  a member's clip claiming another member       -> share accepted
  outsider  an outsider's clip claiming any member        -> FAR = share accepted

Threshold rule (validation only): the lowest FRR such that outsider FAR <= 5%
and mismatch rate <= 5%. When a range of thresholds gives that same lowest
FRR, the middle of the range is used (most margin on both sides). If no
threshold meets both limits, the equal-error-rate threshold is used instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MAX_FAR = 0.05
MAX_MISMATCH = 0.05
METHODS = ("logreg_conf_dist", "cosine_centroid", "ecapa_cosine")
META_COLS = {"sample_id", "member_id", "speaker_id", "phrase", "source_file",
             "augmentation", "split", "role"}


# ----------------------------------------------------------------------------- features
def handcrafted_columns(df: pd.DataFrame, mfcc_only: bool = False) -> list[str]:
    cols = [c for c in df.columns if c not in META_COLS and not c.startswith("ecapa_")
            and pd.api.types.is_numeric_dtype(df[c])]
    return [c for c in cols if c.startswith("mfcc_")] if mfcc_only else cols


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("ecapa_")]


def _l2(Z: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.where(n == 0, 1.0, n)


# ----------------------------------------------------------------------------- fitting
@dataclass
class Candidate:
    """One method + feature set; `params` holds the tuned decision settings."""
    name: str
    method: str
    features: list[str]
    params: dict = field(default_factory=dict)


def fit(candidate: Candidate, train: pd.DataFrame) -> dict:
    """Enrol the four members from TRAIN rows only (originals + augmentations).

    Returns a plain dict (the saved model bundle, minus thresholds).
    """
    X = train[candidate.features].to_numpy(float)
    y = train["member_id"].astype(str).to_numpy()
    # Identification model (used for accuracy / F1 / log loss and by method A).
    clf = Pipeline([("prep", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000))]).fit(X, y)
    classes = list(clf.classes_)
    if candidate.method == "ecapa_cosine":
        Z = _l2(X)                         # embeddings are compared by angle only
    else:
        Z = clf.named_steps["prep"].transform(X)
    centroids = {m: Z[y == m].mean(axis=0) for m in classes}
    max_dist = {m: float(np.linalg.norm(Z[y == m] - centroids[m], axis=1).max()) for m in classes}
    return {"kind": "voice", "method": candidate.method, "candidate": candidate.name,
            "features": list(candidate.features), "label": "member_id", "classes": classes,
            "model": clf, "centroids": centroids, "max_train_distance": max_dist,
            "params": dict(candidate.params)}


def _represent(bundle: dict, X: np.ndarray) -> np.ndarray:
    if bundle["method"] == "ecapa_cosine":
        return _l2(X)
    return bundle["model"].named_steps["prep"].transform(X)


def raw_scores(bundle: dict, df: pd.DataFrame) -> dict:
    """Per clip x member: LR probability, cosine similarity and distance to centroid."""
    X = df[bundle["features"]].to_numpy(float)
    Z = _represent(bundle, X)
    C = np.stack([bundle["centroids"][m] for m in bundle["classes"]])
    return {"proba": bundle["model"].predict_proba(X),
            "cos": _l2(Z) @ _l2(C).T,
            "dist": np.linalg.norm(Z[:, None, :] - C[None, :, :], axis=2)}


def decide(bundle: dict, S: dict, rows: np.ndarray, claims: np.ndarray, params: dict) -> np.ndarray:
    """Accept / reject for clip `rows[i]` claiming member index `claims[i]`."""
    if bundle["method"] == "logreg_conf_dist":
        P = S["proba"][rows]
        top1 = P.argmax(axis=1) == claims
        conf_ok = P[np.arange(len(rows)), claims] >= params["threshold"]
        limits = np.array([bundle["max_train_distance"][m] for m in bundle["classes"]])
        dist_ok = S["dist"][rows, claims] <= params["distance_factor"] * limits[claims]
        return top1 & conf_ok & dist_ok
    cos = S["cos"][rows]
    ok = cos[np.arange(len(rows)), claims] >= params["threshold"]
    if params.get("require_top1"):
        ok &= cos.argmax(axis=1) == claims
    return ok


def verification_score(bundle: dict, S: dict, rows, claims) -> np.ndarray:
    """The number compared with the threshold (probability or cosine similarity)."""
    key = "proba" if bundle["method"] == "logreg_conf_dist" else "cos"
    return S[key][rows, claims]


# ----------------------------------------------------------------------------- trials
def make_trials(members: pd.DataFrame, outsiders: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Every clip x every claimed member, labelled genuine / mismatch / outsider."""
    both = pd.concat([members, outsiders], ignore_index=True)
    is_outsider = np.r_[np.zeros(len(members), bool), np.ones(len(outsiders), bool)]
    rows = []
    for i, r in enumerate(both.itertuples(index=False)):
        for k, claim in enumerate(classes):
            if is_outsider[i]:
                kind = "outsider"
            else:
                kind = "genuine" if r.member_id == claim else "mismatch"
            rows.append({"row": i, "claim_idx": k, "claimed_member": claim, "trial": kind,
                         "sample_id": r.sample_id, "speaker_id": r.speaker_id,
                         "augmentation": r.augmentation, "phrase": r.phrase})
    return both, pd.DataFrame(rows)


def rates(trials: pd.DataFrame, accepted: np.ndarray) -> dict:
    t = trials.assign(accepted=accepted)
    out = {}
    for kind, metric, fn in [("outsider", "far", lambda a: a.mean()),
                             ("genuine", "frr", lambda a: 1 - a.mean()),
                             ("mismatch", "mismatch_rate", lambda a: a.mean())]:
        sub = t[t.trial == kind]
        out[metric] = float(fn(sub.accepted)) if len(sub) else float("nan")
        errors = int(sub.accepted.sum() if kind != "genuine" else (~sub.accepted).sum())
        out[f"{metric}_errors"] = errors
        out[f"{metric}_trials"] = int(len(sub))
    return out


# ----------------------------------------------------------------------------- tuning
def _grid(bundle: dict, S: dict, trials: pd.DataFrame) -> list[dict]:
    rows, claims = trials.row.to_numpy(), trials.claim_idx.to_numpy()
    out = []
    if bundle["method"] == "logreg_conf_dist":
        for f in np.round(np.arange(0.25, 3.001, 0.05), 2):
            for t in np.round(np.arange(0.25, 1.0, 0.01), 2):
                p = {"threshold": float(t), "distance_factor": float(f)}
                out.append({**p, **rates(trials, decide(bundle, S, rows, claims, p))})
        return out
    for top1 in (False, True):
        for t in np.round(np.arange(-1.0, 1.0005, 0.005), 3):
            p = {"threshold": float(t), "require_top1": top1}
            out.append({**p, **rates(trials, decide(bundle, S, rows, claims, p))})
    return out


def _pick(group: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Apply the threshold rule to one sweep (fixed other settings)."""
    ok = group[(group.far <= MAX_FAR) & (group.mismatch_rate <= MAX_MISMATCH)]
    if ok.empty:   # no threshold meets the limits: fall back to the equal error rate
        gap = (group.frr - np.maximum(group.far, group.mismatch_rate)).abs()
        best = group[gap == gap.min()]
        return best.iloc[len(best) // 2], False
    best = ok[ok.frr == ok.frr.min()]
    best = best[(best.far + best.mismatch_rate) == (best.far + best.mismatch_rate).min()]
    return best.iloc[len(best) // 2], True        # middle of the tied threshold range


def tune(bundle: dict, S: dict, trials: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Choose decision settings on VALIDATION trials with the rule in the module docstring."""
    grid = pd.DataFrame(_grid(bundle, S, trials))
    other = "distance_factor" if bundle["method"] == "logreg_conf_dist" else "require_top1"
    picks = []
    for value, g in grid.groupby(other, sort=True):
        row, feasible = _pick(g.sort_values("threshold"))
        picks.append((not feasible, row.frr, row.far + row.mismatch_rate, row, feasible))
    picks.sort(key=lambda p: p[:3])
    best = [p for p in picks if p[:3] == picks[0][:3]]
    chosen, feasible = best[len(best) // 2][3:]  # ties between settings: take the middle one
    params = {"threshold": float(chosen.threshold), other: chosen[other].item()
              if hasattr(chosen[other], "item") else chosen[other]}
    if other == "require_top1":
        params["require_top1"] = bool(params["require_top1"])
    params["meets_limits"] = bool(feasible)
    return params, grid


def eer(bundle: dict, S: dict, trials: pd.DataFrame) -> float:
    """Equal error rate between genuine and outsider trials (score threshold sweep)."""
    sc = verification_score(bundle, S, trials.row.to_numpy(), trials.claim_idx.to_numpy())
    g, o = sc[trials.trial.to_numpy() == "genuine"], sc[trials.trial.to_numpy() == "outsider"]
    ts = np.unique(np.concatenate([g, o, [np.inf]]))
    frr = np.array([(g < t).mean() for t in ts])
    far = np.array([(o >= t).mean() for t in ts])
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2)


# ----------------------------------------------------------------------------- evaluation
def identification_metrics(bundle: dict, members: pd.DataFrame) -> dict:
    """Which member is speaking (closed set): accuracy, F1 and log loss of the LR model."""
    X = members[bundle["features"]].to_numpy(float)
    y = members["member_id"].astype(str).to_numpy()
    clf = bundle["model"]
    pred = clf.predict(X)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "log_loss": float(log_loss(y, clf.predict_proba(X), labels=list(clf.classes_))),
            "clips": int(len(y))}


def evaluate(bundle: dict, members: pd.DataFrame, outsiders: pd.DataFrame,
             params: dict | None = None) -> tuple[dict, pd.DataFrame]:
    """FAR / FRR / mismatch (+ identification metrics) on one split, plus per-trial table."""
    params = params or bundle["params"]
    both, trials = make_trials(members, outsiders, bundle["classes"])
    S = raw_scores(bundle, both)
    rows, claims = trials.row.to_numpy(), trials.claim_idx.to_numpy()
    acc = decide(bundle, S, rows, claims, params)
    trials = trials.assign(score=verification_score(bundle, S, rows, claims), accepted=acc)
    res = {**rates(trials, acc), **identification_metrics(bundle, members)}
    return res, trials


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float]:
    """Exact 95% interval for an error rate: shows how little a few trials can prove."""
    from scipy.stats import beta
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


# ----------------------------------------------------------------------------- deployed use
def verify(bundle: dict, features: dict, expected_member: str | None = None):
    """One clip -> (best_matching_member, its score, authorized).

    `features` is the dict from preprocess_audio.extract_features (plus ecapa_*
    columns for the ECAPA method). With expected_member, `authorized` is the
    claimed-identity decision; without it, the clip is checked against its own
    best match. The score is the cosine similarity (cosine methods) or the
    classifier probability (logreg_conf_dist) of the best-matching member.
    """
    missing = [c for c in bundle["features"] if c not in features]
    if missing:
        hint = ""
        if any(c.startswith("ecapa_") for c in missing):
            hint = (" The voice model needs the ECAPA speaker embedding; install SpeechBrain so "
                    "extract_features adds it: pip install torch torchaudio --index-url "
                    "https://download.pytorch.org/whl/cpu && pip install speechbrain")
        raise ValueError(f"Voice features missing: {missing[:3]}"
                         f"{' ...' if len(missing) > 3 else ''}.{hint}")
    df = pd.DataFrame([{c: features[c] for c in bundle["features"]}])
    S = raw_scores(bundle, df)
    key = "proba" if bundle["method"] == "logreg_conf_dist" else "cos"
    best = int(S[key][0].argmax())
    claim = best if expected_member is None else bundle["classes"].index(expected_member) \
        if expected_member in bundle["classes"] else None
    if claim is None:                          # claimed identity is not an enrolled member
        return bundle["classes"][best], float(S[key][0, best]), False
    ok = bool(decide(bundle, S, np.array([0]), np.array([claim]), bundle["params"])[0])
    return bundle["classes"][best], float(S[key][0, best]), ok
