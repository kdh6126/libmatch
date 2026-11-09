"""libmatch — check whether two sequence-label datasets differ only as intended.

When a method scores better on dataset A than on dataset B, that gap is only
interpretable if the two datasets differ along the axis the comparison was
about. Putting a synthetic benchmark next to a measured one makes this
particularly hard. A typical synthetic set is millions of mutually unrelated
sequences; a typical measured set is tens of thousands of variants around a
single parent. They differ not only in where the labels came from but, at the
same time, in size, in structure, in whether they can be aligned, and in
measurement noise.

This tool counts those axes and reports which of them are mismatched. It does
not say which dataset is correct -- it shows that a mismatch exists and how
large it is, and leaves it to the user to decide whether to control for it or
to report it.

Five axes are measured.

  size        number of sequences
  structure   variants around one parent, or a set of unrelated sequences
  dimension   number of positions that vary (where the set can be aligned)
  noise       noise-to-signal ratio, where replicate measurements exist;
              reported as not measured where they do not, which is not the
              same as reporting it as zero
  budget      the range of training sizes available

The third is the one most often missed. Matching size and structure leaves the
number of mutated positions untouched, and that axis changes how well a
surrogate model fits, which in turn changes search performance.

    libmatch report A.csv B.csv
    libmatch report A.csv B.csv --seq-col cdr3 --label-col energy
    libmatch report A.csv B.csv --curve        also fit learning curves

A collection can also be measured once and then queried, which answers the
question the other way round: given a dataset you intend to deploy on, which
public datasets could be compared against it without confounding?

    libmatch profile assays/*.csv --catalogue catalogue.csv
    libmatch match target.csv --catalogue catalogue.csv --top 10

Input is CSV, TSV or Parquet with a sequence column and a label column. The
tool is not tied to any particular benchmark.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd

__version__ = "0.1.0"
NL = chr(10)


# --------------------------------------------------------------------------
# Reading input

def read_table(path, seq_col=None, label_col=None, wt=None):
    """Read CSV/TSV/Parquet and settle on the sequence and label columns."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        d = pd.read_parquet(path)
    else:
        sep = "\t" if ext in (".tsv", ".tab") else ","
        d = pd.read_csv(path, sep=sep)
    if seq_col is None:
        # "mutated_sequence" 가 "mutant" 앞에 와야 한다. ProteinGym 의 assay
        # 파일은 둘을 같이 담고 있는데, 표기를 집으면 야생형 서열을 따로 받아야
        # 하고 위치 기준을 맞출 일이 생긴다. 전장 서열이 이미 있으면 그것을
        # 쓰는 편이 언제나 옳다.
        seq_col = _guess(d, ("sequence", "mutated_sequence", "seq", "cdr3",
                             "cdrh3", "aa", "mutant", "variant"), object)
    if label_col is None:
        label_col = _guess(d, ("label", "y", "energy", "affinity", "score",
                               "fitness", "dms_score"), float)
    for c, what in ((seq_col, "sequence"), (label_col, "label")):
        if c not in d.columns:
            raise SystemExit(
                f"{path}: no {what} column {c!r}. "
                f"Columns present: {', '.join(map(str, d.columns[:12]))}")
    out = d[[seq_col, label_col]].copy()
    out.columns = ["sequence", "label"]
    out["sequence"] = out.sequence.astype(str).str.strip().str.upper()

    if looks_like_mutations(out.sequence.tolist()):
        if not wt:
            raise SystemExit(
                f"{path}: column {seq_col!r} holds mutation notation rather "
                f"than full-length sequences (e.g. {out.sequence.iloc[0]}). "
                f"Pass the wild-type sequence with --wt, or a file holding "
                f"it with --wt-file.")
        built, msgs = [], 0
        for v in out.sequence:
            seq, why = apply_mutations(v, wt)
            if seq is None:
                built.append(None)
            else:
                built.append(seq)
                msgs += bool(why)
        out["sequence"] = built
        n_bad = int(out.sequence.isna().sum())
        if n_bad:
            print(f"  [{os.path.basename(path)}] could not expand {n_bad:,} "
                  f"mutations", file=sys.stderr)
            out = out[out.sequence.notna()]
        if msgs:
            print(f"  [{os.path.basename(path)}] in {msgs:,} mutations the "
                  f"wild-type residue disagreed with the notation -- check "
                  f"that the position numbering matches", file=sys.stderr)
    out["label"] = pd.to_numeric(out.label, errors="coerce")
    n0 = len(out)
    # The parentheses matter: & binds tighter than >, so without them the
    # label filter silently disappears and NaN-labelled rows survive.
    out = out[(out.sequence.str.len() > 0) & out.label.notna()]
    if len(out) < n0:
        print(f"  [{os.path.basename(path)}] dropped {n0 - len(out):,} rows "
              f"with no usable label", file=sys.stderr)
    if out.empty:
        raise SystemExit(f"{path}: no usable rows.")
    return out.reset_index(drop=True)


MUT_RE = None


def looks_like_mutations(values, sample=200):
    """Is the sequence column really mutation notation -- 'T2L', 'A1P:D2N'?"""
    import re
    global MUT_RE
    if MUT_RE is None:
        MUT_RE = re.compile(r"^[A-Z*]\d+[A-Z*]$")
    vals = list(values[:sample])
    if not vals:
        return False
    hits = sum(all(MUT_RE.match(t) for t in str(v).split(":") if t)
               for v in vals)
    return hits / len(vals) > 0.9


def apply_mutations(notation, wt):
    """Expand 'A1P:D2N' into a full sequence. Positions are 1-based.

    Deep mutational scanning data is usually distributed as mutation notation
    rather than full sequences. A tool that cannot take that format directly
    forces every user to write a conversion script, and those scripts commonly
    get the position offset wrong by one. So the wild-type residue named in
    the notation is checked against the wild-type sequence given, and any
    disagreement is reported.
    """
    out = list(wt)
    bad = 0
    for t in str(notation).split(":"):
        t = t.strip()
        if not t:
            continue
        frm, pos, to = t[0], int(t[1:-1]), t[-1]
        if not 1 <= pos <= len(wt):
            return None, "position outside the sequence"
        if wt[pos - 1] != frm:
            bad += 1
        out[pos - 1] = to
    return "".join(out), ("wild-type residue disagrees with notation"
                          if bad else "")


def _guess(d, names, kind):
    low = {str(c).lower(): c for c in d.columns}
    for n in names:
        if n in low:
            return low[n]
    # No name matched, so fall back on dtype: sequences are strings, labels
    # are numeric.
    for c in d.columns:
        if kind is float and pd.api.types.is_numeric_dtype(d[c]):
            return c
        if kind is object and not pd.api.types.is_numeric_dtype(d[c]):
            return c
    return names[0]


# --------------------------------------------------------------------------
# The five axes

@dataclass
class Arm:
    """The five properties of one dataset."""
    name: str
    n: int
    lengths: tuple
    alignable: bool
    structure: str            # "mutational neighbourhood" | "unrelated set"
    parent: str | None
    n_varying: int | None
    median_mutations: float | None
    replicates: int
    noise_sd: float | None
    noise_to_signal: float | None
    label_sd: float
    usable_train_max: int
    notes: list = field(default_factory=list)


def consensus(seqs):
    """Per-position modal residue. In a variant neighbourhood this is the
    parent."""
    L = len(seqs[0])
    arr = np.frombuffer("".join(seqs).encode(), dtype="S1").reshape(-1, L)
    out = []
    for j in range(L):
        v, c = np.unique(arr[:, j], return_counts=True)
        out.append(v[int(np.argmax(c))].decode())
    return "".join(out)


def identity_to(seqs, ref, sample=400, rng=None):
    """Identity to a reference sequence; lengths need not match.

    Affinity-maturation libraries usually hold length fixed, but once
    insertions and deletions are present the members differ by a few residues.
    Requiring equal length before a set counts as a neighbourhood would
    classify real measured libraries as "unrelated sequences" -- the first
    thing to break when this tool is pointed at somebody else's data. So the
    measure used here does not depend on length.
    """
    from difflib import SequenceMatcher
    rng = rng or np.random.default_rng(0)
    idx = (rng.choice(len(seqs), sample, replace=False)
           if len(seqs) > sample else range(len(seqs)))
    # autojunk=False is essential. By default SequenceMatcher treats any
    # element occurring in more than 1% of a sequence of length 200 or more
    # as too common to be useful and discards it. Proteins have only twenty
    # residues, so in a 243-residue sequence every residue clears that
    # threshold. The identity of a single-mutant pair then comes out as 0.47
    # instead of 0.996, and the shuffled baseline as 0.00. Short CDRH3s (17
    # residues) never reach the threshold, so the bug stays invisible there
    # and appears only on long measured sets -- exactly where this tool is
    # meant to be used.
    return np.array([SequenceMatcher(None, ref, seqs[i], autojunk=False).ratio()
                     for i in idx])


def concentration(seqs, ref, rng=None):
    """Where the set sits between random and identical. 0 random, 1 identical.

    Using raw identity as a threshold makes the verdict depend on length --
    two mutations in 17 residues gives 0.88, five in 125 gives 0.96. The same
    kind of library produces different numbers, and any fixed threshold
    necessarily misclassifies one of them.

    So sequences shuffled to the same composition provide the baseline. A
    neighbourhood sits far above that baseline; an unrelated set sits close to
    it. Normalising by the gap cancels out length and composition.
    """
    rng = rng or np.random.default_rng(0)
    obs = float(np.median(identity_to(seqs, ref, rng=rng)))
    pick = (list(rng.choice(seqs, 200, replace=False)) if len(seqs) > 200
            else list(seqs))
    sh = ["".join(rng.permutation(list(x))) for x in pick]
    null = float(np.median(identity_to(sh, ref, rng=rng)))
    if null >= 1.0:
        return obs, null, float("nan")
    return obs, null, (obs - null) / (1.0 - null)


def describe(d, name, concentration_cut=0.6, holdout_frac=0.25,
             deterministic=None):
    """Measure the five properties of one arm.

    ``deterministic`` says what one label per sequence means for this table.
    A computed oracle returns one value because recomputing would return the
    same value, so its replicate spread is zero; a measured assay distributed
    one score per variant looks identical here but its spread was averaged
    away before the file was written. Nothing in the table distinguishes
    them, so the caller declares it. Left undeclared, the noise axis is
    reported as not measured rather than as zero.
    """
    seqs = d.sequence.tolist()
    y = d.label.to_numpy(float)
    lens = pd.Series([len(s) for s in seqs])
    uniq = sorted(lens.unique())
    modal = int(lens.mode()[0])
    at_modal = [s for s in seqs if len(s) == modal]
    frac_modal = len(at_modal) / len(seqs)
    alignable = len(uniq) == 1
    notes = []

    # Build the consensus at the modal length, so that it stays defined even
    # when lengths are mixed.
    ref = consensus(at_modal)
    med_ident, null_ident, conc = concentration(seqs, ref)

    parent = n_var = med_mut = None
    if conc >= concentration_cut:
        structure = "mutational neighbourhood"
        parent = ref if ref in set(seqs) else None
        arr = np.frombuffer("".join(at_modal).encode(),
                            dtype="S1").reshape(-1, modal)
        pa = np.frombuffer(ref.encode(), dtype="S1")
        n_var = int((arr != pa).any(0).sum())
        med_mut = float(np.median((arr != pa).sum(1)))
        if parent is None:
            notes.append("the consensus sequence is not itself in the data "
                         "-- either the parent was left out, or the design is "
                         "not symmetric around it")
        notes.append(f"concentration towards the consensus {conc:.2f} "
                     f"(identity {med_ident:.2f}, {null_ident:.2f} for the "
                     f"same composition shuffled)")
        if frac_modal < 0.99:
            notes.append(f"lengths range {uniq[0]}-{uniq[-1]}, so varying "
                         f"positions were counted only on the "
                         f"{frac_modal:.0%} at the modal length {modal}")
    else:
        structure = "unrelated set"
        notes.append(f"concentration towards the consensus is {conc:.2f}, "
                     f"below {concentration_cut:.2f} (identity "
                     f"{med_ident:.2f}, {null_ident:.2f} for the same "
                     f"composition shuffled)")
        if not alignable:
            notes.append(f"lengths range {uniq[0]}-{uniq[-1]}, so an aligned "
                         f"one-hot encoding is not defined")

    g = d.groupby("sequence").label
    rep = g.count()
    n_rep = int((rep > 1).sum())
    noise_sd = nts = None
    if n_rep >= 5:
        within = g.std(ddof=1).dropna()
        noise_sd = float(np.sqrt((within ** 2).mean()))
        sd = y.std(ddof=1)
        nts = float(noise_sd / sd) if sd > 0 else None
        # A handful of replicated sequences in a large set still yields a
        # number, and that number then stands for the whole arm. Say how
        # thin the basis is rather than letting it pass as the arm's noise.
        frac_rep = n_rep / len(rep)
        if frac_rep < 0.05:
            notes.append(f"noise estimated from the {n_rep:,} of {len(rep):,} "
                         f"sequences ({frac_rep:.1%}) that were measured more "
                         f"than once")
    elif rep.max() > 1:
        notes.append(f"only {n_rep} sequences occur more than once, too few "
                     f"to estimate noise")
    elif deterministic:
        # A lattice simulation and the like. Recomputing returns the same
        # number, so the spread is not unknown, it is zero.
        noise_sd, nts = 0.0, 0.0
        notes.append("labels declared deterministic, so one value per "
                     "sequence means a replicate spread of zero")
    else:
        # A measured assay whose replicates were averaged away before release
        # looks exactly like the deterministic case from here, and its noise
        # is not zero -- it is unrecoverable. Undeclared, the honest report is
        # that the axis was not measured. ``noise_sd`` and ``nts`` stay None.
        notes.append("each sequence appears once and the labels are not "
                     "declared deterministic, so replicate spread cannot be "
                     "estimated; declare the labels deterministic "
                     "(deterministic=True, or --deterministic) for a "
                     "computed oracle, or measure noise on the raw "
                     "replicates")

    return Arm(name=name, n=len(d), lengths=(uniq[0], uniq[-1]),
               alignable=alignable, structure=structure, parent=parent,
               n_varying=n_var, median_mutations=med_mut, replicates=n_rep,
               noise_sd=noise_sd, noise_to_signal=nts,
               label_sd=float(y.std(ddof=1)),
               usable_train_max=int(len(d) * (1 - holdout_frac)), notes=notes)


# --------------------------------------------------------------------------
# Comparison

MISMATCH = "mismatched"
MATCH = "matched"
# 세 번째 판정. "두 쪽이 다르다" 도 "두 쪽이 같다" 도 아닌, "이 축은 이
# 자료로는 물을 수 없다" 는 뜻이다. 상태를 둘로만 두면 물을 수 없는 축이
# 반드시 둘 중 하나로 접히고, 접히는 쪽은 대개 "같다" 다 -- 아무것도 재지
# 않았다는 사실이 일치했다는 주장으로 바뀐다.
UNAVAILABLE = "unavailable"


def compare(a: Arm, b: Arm):
    """Judge each axis mismatched or not. Never says which side is correct."""
    rows = []

    def add(axis, va, vb, mismatch, why="", status=None):
        rows.append(dict(axis=axis, a=va, b=vb,
                         status=status or (MISMATCH if mismatch else MATCH),
                         note=why))

    ratio = max(a.n, b.n) / max(1, min(a.n, b.n))
    add("size", f"{a.n:,}", f"{b.n:,}", ratio >= 2,
        f"{ratio:.0f}x apart" if ratio >= 2 else "")

    add("structure", a.structure, b.structure, a.structure != b.structure,
        "only one side is a variant neighbourhood around a parent"
        if a.structure != b.structure else "")

    if a.n_varying is None or b.n_varying is None:
        add("dimension", _fmt(a.n_varying, "{} positions"),
            _fmt(b.n_varying, "{} positions"), True,
            "one side cannot be aligned, so the axis is not comparable")
    else:
        r = max(a.n_varying, b.n_varying) / max(1, min(a.n_varying,
                                                       b.n_varying))
        add("dimension", f"{a.n_varying} positions",
            f"{b.n_varying} positions", r >= 1.5,
            f"{r:.1f}x apart" if r >= 1.5 else "")

    if a.noise_to_signal is None or b.noise_to_signal is None:
        # Not a mismatch. A mismatch is a difference that was measured, and
        # nothing was measured here; counting it as one would let a question
        # that was never asked swell the tally of axes that differ.
        both = a.noise_to_signal is None and b.noise_to_signal is None
        who = ("neither side has replicate measurements" if both else
               f"{(a if a.noise_to_signal is None else b).name} has no "
               f"replicate measurements")
        add("noise", _fmt(a.noise_to_signal, "{:.3f}"),
            _fmt(b.noise_to_signal, "{:.3f}"), False,
            f"{who}, so the axis cannot be compared", status=UNAVAILABLE)
    else:
        d = abs(a.noise_to_signal - b.noise_to_signal)
        add("noise", f"{a.noise_to_signal:.3f}", f"{b.noise_to_signal:.3f}",
            d >= 0.1, f"difference {d:.3f}" if d >= 0.1 else "")

    r = max(a.usable_train_max, b.usable_train_max) / max(
        1, min(a.usable_train_max, b.usable_train_max))
    add("budget", f"up to {a.usable_train_max:,}",
        f"up to {b.usable_train_max:,}", r >= 2,
        "little overlap in the training sizes both sides can reach"
        if r >= 2 else "")
    return pd.DataFrame(rows)


def _fmt(v, f="{}"):
    return "--" if v is None else f.format(v)


# --------------------------------------------------------------------------
# Optional: learning curves

def curve(d, sizes, seeds, holdout):
    """Held-out Spearman of ridge regression against training size.

    The two arms are only comparable if measured under one protocol, so the
    encoder is chosen automatically -- per-position one-hot where the set
    aligns, terminal residues plus dipeptide composition where it does not.
    The report states when that choice itself differs between the arms.
    """
    try:
        from scipy.stats import spearmanr
        from sklearn.linear_model import Ridge
    except ImportError as e:
        raise SystemExit(
            f"--curve needs scipy and scikit-learn, which are optional "
            f"({e}). Install them with: pip install \"libmatch[curve]\"")

    seqs = d.sequence.tolist()
    y = d.label.to_numpy(np.float32)
    X, enc = encode(seqs)
    out = []
    for s in range(seeds):
        perm = np.random.default_rng(1000 + s).permutation(len(y))
        te, pool = perm[:holdout], perm[holdout:]
        for m in sizes:
            if m > len(pool):
                continue
            tr = pool[:m]
            r = Ridge(alpha=1.0).fit(X[tr], y[tr])
            rho = float(spearmanr(y[te], r.predict(X[te])).statistic)
            out.append(dict(n_train=m, seed=s, spearman=round(rho, 4)))
    return pd.DataFrame(out), enc


AA = "ACDEFGHIKLMNPQRSTVWY"
AAI = {a: i for i, a in enumerate(AA)}


def encode(seqs):
    """Per-position one-hot where aligned, else terminal residues plus
    dipeptide composition."""
    lens = {len(s) for s in seqs}
    if len(lens) == 1:
        L = lens.pop()
        arr = np.array([list(s) for s in seqs])
        var = [j for j in range(L) if len(set(arr[:, j])) > 1]
        X = np.zeros((len(seqs), len(var) * 21), np.float32)
        for c, j in enumerate(var):
            for i, ch in enumerate(arr[:, j]):
                X[i, c * 21 + AAI.get(ch, 20)] = 1.0
        return X, f"aligned one-hot ({len(var)} varying positions)"
    NT = 5
    X = np.zeros((len(seqs), 2 * NT * 20 + 400 + 1), np.float32)
    for i, s in enumerate(seqs):
        for j in range(min(NT, len(s))):
            k = AAI.get(s[j])
            if k is not None:
                X[i, j * 20 + k] = 1.0
            k = AAI.get(s[len(s) - 1 - j])
            if k is not None:
                X[i, NT * 20 + j * 20 + k] = 1.0
        for j in range(len(s) - 1):
            p, q = AAI.get(s[j]), AAI.get(s[j + 1])
            if p is not None and q is not None:
                X[i, 2 * NT * 20 + p * 20 + q] += 1.0
        X[i, -1] = len(s)
    return X, "terminal residues + dipeptide counts + length"


# --------------------------------------------------------------------------
# Catalogues: profile many datasets once, then ask which of them match a target

CAT_COLS = ["dataset", "n", "len_min", "len_max", "alignable", "structure",
            "n_varying", "median_mutations", "replicates", "noise_to_signal",
            "label_sd", "usable_train_max"]


def arm_to_row(arm: Arm):
    return {"dataset": arm.name, "n": arm.n, "len_min": arm.lengths[0],
            "len_max": arm.lengths[1], "alignable": arm.alignable,
            "structure": arm.structure, "n_varying": arm.n_varying,
            "median_mutations": arm.median_mutations,
            "replicates": arm.replicates,
            "noise_to_signal": arm.noise_to_signal,
            "label_sd": arm.label_sd,
            "usable_train_max": arm.usable_train_max}


def _opt(r, k, cast):
    v = r.get(k)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def arm_from_row(r):
    """카탈로그 한 줄을 Arm 으로. 다시 읽지 않고 비교만 한다."""
    return Arm(name=str(r["dataset"]), n=int(r["n"]),
               lengths=(int(r["len_min"]), int(r["len_max"])),
               alignable=bool(r["alignable"]), structure=str(r["structure"]),
               parent=None, n_varying=_opt(r, "n_varying", int),
               median_mutations=_opt(r, "median_mutations", float),
               replicates=int(_opt(r, "replicates", int) or 0),
               noise_sd=None,
               noise_to_signal=_opt(r, "noise_to_signal", float),
               label_sd=_opt(r, "label_sd", float) or 0.0,
               usable_train_max=int(r["usable_train_max"]), notes=[])


def distance(a: Arm, b: Arm):
    """축을 몇 개 어긋났는지로 먼저 가르고, 그 안에서 얼마나 떨어졌는지.

    어긋난 축의 수만으로 줄을 세우면 동점이 수십 개 나온다. 같은 수의 축이
    어긋났다면 크기 차가 작은 쪽이 더 쓸 만한 대조이므로, 비교 가능한 축의
    로그비 합을 이어서 쓴다. 구조가 다르면 어떤 로그비도 그 차이를 대신하지
    못하므로 따로 더한다.
    """
    d = 0.0
    for x, y in ((a.n, b.n), (a.usable_train_max, b.usable_train_max),
                 (a.n_varying, b.n_varying)):
        if x and y:
            d += abs(np.log10(max(x, 1) / max(y, 1)))
    if a.structure != b.structure:
        d += 1.0
    if a.noise_to_signal is not None and b.noise_to_signal is not None:
        d += abs(a.noise_to_signal - b.noise_to_signal)
    return float(d)


def _stem(name):
    """확장자를 뗀 이름. 표적이 자기 자신과 견주는 것을 막는 데만 쓴다.

    카탈로그는 'GLPA_HUMAN_Elazar_2016' 처럼 확장자 없이 적힌 것도 있고
    'alphaseq_measured.csv' 처럼 붙여 적힌 것도 있다. 이름을 글자 그대로
    견주면 두 관례가 섞인 순간 표적이 후보 목록 맨 위에 거리 0.00 으로
    자기 자신을 올린다 -- 가장 잘 맞는 대조가 자기 자신이라는 답은 아무것도
    말해 주지 않는다.
    """
    return os.path.splitext(str(name))[0]


def match(target: Arm, cat, top=10):
    """카탈로그에서 표적에 가장 정합인 데이터셋을 고른다."""
    rows = []
    for _, r in cat.iterrows():
        if _stem(r["dataset"]) == _stem(target.name):
            continue
        other = arm_from_row(r)
        cmp_df = compare(target, other)
        rows.append({"dataset": other.name,
                     "mismatched": int((cmp_df.status == MISMATCH).sum()),
                     "unavailable": int(
                         (cmp_df.status == UNAVAILABLE).sum()),
                     "distance": round(distance(target, other), 3),
                     **{x.axis: x.status for _, x in cmp_df.iterrows()}})
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    return d.sort_values(["mismatched", "distance"]).head(top).reset_index(
        drop=True)


def render_match(target: Arm, d, cat_n, top):
    L = [f"libmatch {__version__}", "",
         f"Target  {target.name}",
         f"        {target.n:,} sequences, {target.structure}, "
         + (f"{target.n_varying} varying positions"
            if target.n_varying is not None else "not alignable"),
         "", f"Closest of {cat_n} catalogued datasets", ""]
    if d.empty:
        return NL.join(L + ["  the catalogue is empty."]) + NL
    w = max(len(x) for x in d.dataset.tolist() + ["dataset"])
    L.append(f"  {'dataset':<{w}}  off  dist  " +
             "  ".join(f"{x[:4]:<4}" for x in AXIS_ORDER))
    for _, r in d.iterrows():
        marks = "  ".join(
            f"{({MATCH: '.', MISMATCH: 'X'}.get(r[x], '?')):<4}"
            for x in AXIS_ORDER)
        L.append(f"  {r.dataset:<{w}}  {r.mismatched}/5  "
                 f"{r.distance:>4.2f}  {marks}")
    best = d.iloc[0]
    L += ["", f"'.' means the axis is shared, 'X' that it is not, '?' that "
              f"it could",
          "not be measured on these two. The ranking is by",
          "number of mismatched axes first and by how far apart the "
          "comparable",
          "axes are second, so a dataset at the top differs from the target "
          "in",
          "the fewest ways that would confound a comparison."]
    if best.mismatched == 0 and not best.get("unavailable", 0):
        L += ["", f"{best.dataset} matches the target on all five axes."]
    elif best.mismatched == 0:
        L += ["", f"{best.dataset} differs on no measured axis, but "
                  f"{best.unavailable} of the five",
              "could not be compared on these two datasets."]
    else:
        L += ["", f"No catalogued dataset matches on all five; the closest, "
                  f"{best.dataset},",
              f"still differs on {best.mismatched}. Report those axes "
              f"alongside any comparison."]
    return NL.join(L) + NL


AXIS_ORDER = ["size", "structure", "dimension", "noise", "budget"]


# --------------------------------------------------------------------------
# Report

def render(a: Arm, b: Arm, cmp_df, curves=None):
    L = [f"libmatch {__version__}", "",
         f"A  {a.name}", f"B  {b.name}", "",
         "Five axes", ""]
    w = max(len(str(x)) for x in cmp_df.a.tolist() + cmp_df.b.tolist() + ["A"])
    L.append(f"  {'axis':<10s} {'A':<{w}s}  {'B':<{w}s}  verdict")
    for _, r in cmp_df.iterrows():
        line = f"  {r.axis:<10s} {str(r.a):<{w}s}  {str(r.b):<{w}s}  {r.status}"
        if r.note:
            line += f"  ({r.note})"
        L.append(line)

    bad = cmp_df[cmp_df.status == MISMATCH]
    na = cmp_df[cmp_df.status == UNAVAILABLE]
    tally = f"{len(bad)} of {len(cmp_df)} axes mismatched"
    if len(na):
        tally += f", {len(na)} not comparable ({', '.join(na.axis)})"
    L += ["", tally]
    if len(bad):
        L += ["",
              "The two datasets differ on these axes at the same time. Any",
              "difference measured by comparing them can be attributed to any",
              "one of those axes, so the comparison is not interpretable "
              "unless",
              "the report states which axis was intended and which were",
              "controlled."]
    elif len(na):
        L += ["",
              "No axis was measured to differ, but the axes named above "
              "could",
              "not be compared at all. An axis that was never measured is "
              "not",
              "an axis that agrees, so this is not yet a clean comparison."]
    else:
        L += ["",
              "All five axes match. A difference observed between these two",
              "datasets can reasonably be attributed to the intended axis."]

    for arm in (a, b):
        if arm.notes:
            L += ["", f"Notes on {arm.name}"]
            L += [f"  - {x}" for x in arm.notes]

    if curves:
        L += ["", "Learning curves (ridge regression, held-out Spearman)", ""]
        for nm, (df, enc) in curves.items():
            g = df.groupby("n_train").spearman.agg(["mean", "std"])
            L.append(f"  {nm}  [{enc}]")
            for n, row in g.iterrows():
                L.append(f"    {int(n):>7,}  {row['mean']:+.3f} "
                         f"+/- {row['std']:.3f}")
        # Compare the encoder kind, not its full label: the aligned one-hot
        # label carries the position count, so two arms that both used it
        # would otherwise look like they had used different encoders.
        encs = {enc.split(" (")[0] for _, enc in curves.values()}
        if len(encs) > 1:
            L += ["",
                  "  The two arms used different encoders. Aligned one-hot",
                  "  requires a common length, so this is a property of the",
                  "  data rather than a choice: comparing representations "
                  "here",
                  "  varies the encoder and the library together."]
    return NL.join(L) + NL


# --------------------------------------------------------------------------

def _read_wt(path):
    """Read a wild-type sequence from a FASTA or plain-text file."""
    if not path:
        return None
    raw = io.open(path, encoding="utf-8").read()
    return "".join(x.strip() for x in raw.splitlines()
                   if x.strip() and not x.startswith(">")).upper()


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="libmatch",
        description="check whether two sequence-label datasets differ only "
                    "as intended")
    p.add_argument("command", choices=["report", "profile", "match"],
                   help="report: compare two tables. profile: measure one or "
                        "more tables into a catalogue. match: rank a "
                        "catalogue against one target table")
    p.add_argument("a")
    p.add_argument("b", nargs="?", default=None,
                   help="the second table for 'report'; further tables for "
                        "'profile'; unused for 'match'")
    p.add_argument("more", nargs="*",
                   help="further tables to profile")
    p.add_argument("--catalogue", default=None,
                   help="catalogue CSV: written by 'profile', read by 'match'")
    p.add_argument("--top", type=int, default=10,
                   help="how many matches to show")
    p.add_argument("--seq-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--wt", default=None,
                   help="wild-type sequence; needed when the sequence column "
                        "holds mutation notation")
    p.add_argument("--wt-b", default=None,
                   help="wild-type for B, when it differs from A")
    p.add_argument("--wt-file", default=None,
                   help="file holding the wild-type sequence (FASTA or plain)")
    p.add_argument("--wt-file-b", default=None,
                   help="file holding B's wild-type sequence")
    # describe() 는 라벨이 계산값인지 호출자가 선언하게 되어 있고, 선언이
    # 없으면 잡음 축을 0 이 아니라 "재지 못함" 으로 적는다. 그 선언이
    # 파이썬 API 에만 있어서, 명령줄로 쓰는 사람은 격자 오라클을 아무리
    # 돌려도 잡음을 적을 수 없었다 -- 문서가 권하는 길이 막혀 있었다.
    p.add_argument("--deterministic", action="store_true",
                   help="A 의 라벨은 계산값이다: 다시 계산해도 같은 값이 "
                        "나오므로 복제 퍼짐이 0 이다. 실측 어세이에는 쓰지 "
                        "말 것 -- 한 변이당 한 점수로 배포되었다는 것은 "
                        "퍼짐이 0 이라는 뜻이 아니다")
    p.add_argument("--deterministic-b", action="store_true",
                   help="B 의 라벨이 계산값일 때. 'report' 에서만 쓴다")
    p.add_argument("--curve", action="store_true",
                   help="also fit learning curves (slow)")
    p.add_argument("--sizes", nargs="*", type=int,
                   default=[200, 500, 1000, 2000])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--json", default=None, help="also write results as JSON")
    a = p.parse_args(argv)

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    wt = a.wt or _read_wt(a.wt_file)
    wt_b = a.wt_b or _read_wt(a.wt_file_b) or wt

    if a.command == "profile":
        paths = [a.a] + ([a.b] if a.b else []) + list(a.more)
        rows = []
        for p_ in paths:
            d = read_table(p_, a.seq_col, a.label_col, wt)
            rows.append(arm_to_row(describe(d, os.path.basename(p_),
                                deterministic=a.deterministic)))
            print(f"  {rows[-1]['dataset']}: {rows[-1]['n']:,} sequences, "
                  f"{rows[-1]['structure']}, "
                  f"{rows[-1]['n_varying']} varying positions")
        cat = pd.DataFrame(rows)[CAT_COLS]
        out = a.catalogue or "catalogue.csv"
        cat.to_csv(out, index=False, encoding="utf-8")
        print(f"wrote {out}  ({len(cat)} datasets)")
        return 0

    if a.command == "match":
        if not a.catalogue:
            raise SystemExit("'match' needs --catalogue, a CSV written by "
                             "'libmatch profile'.")
        cat = pd.read_csv(a.catalogue)
        missing = [c for c in ("dataset", "n", "structure",
                               "usable_train_max") if c not in cat.columns]
        if missing:
            raise SystemExit(f"{a.catalogue}: not a catalogue, missing "
                             f"{', '.join(missing)}")
        d = read_table(a.a, a.seq_col, a.label_col, wt)
        target = describe(d, os.path.basename(a.a),
                          deterministic=a.deterministic)
        hits = match(target, cat, a.top)
        print(render_match(target, hits, len(cat), a.top))
        if a.json:
            with open(a.json, "w", encoding="utf-8") as fh:
                json.dump({"version": __version__, "target": asdict(target),
                           "matches": hits.to_dict("records")}, fh,
                          ensure_ascii=False, indent=2)
            print(f"wrote {a.json}")
        return 0 if (len(hits) and hits.iloc[0].mismatched == 0) else 1

    if not a.b:
        raise SystemExit("'report' needs two tables.")
    da = read_table(a.a, a.seq_col, a.label_col, wt)
    db = read_table(a.b, a.seq_col, a.label_col, wt_b)
    A = describe(da, os.path.basename(a.a),
                 deterministic=a.deterministic)
    B = describe(db, os.path.basename(a.b),
                 deterministic=a.deterministic_b)
    cmp_df = compare(A, B)

    curves = None
    if a.curve:
        hold = min(1500, min(len(da), len(db)) // 4)
        sizes = [s for s in a.sizes if s <= min(len(da), len(db)) - hold]
        if not sizes:
            print("  not enough data to fit learning curves", file=sys.stderr)
        else:
            curves = {A.name: curve(da, sizes, a.seeds, hold),
                      B.name: curve(db, sizes, a.seeds, hold)}

    text = render(A, B, cmp_df, curves)
    print(text)
    if a.json:
        blob = {"version": __version__, "a": asdict(A), "b": asdict(B),
                "axes": cmp_df.to_dict("records")}
        if curves:
            blob["curves"] = {k: v[0].to_dict("records")
                              for k, v in curves.items()}
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=2)
        print(f"wrote {a.json}")
    # 0 은 "다섯 축을 다 재었고 다 맞다" 는 뜻이어야 한다. 재지 못한 축을
    # 통과로 세면 물어보지 않은 질문이 합격 판정이 된다.
    return 0 if (cmp_df.status == MATCH).all() else 1


if __name__ == "__main__":
    sys.exit(main())
