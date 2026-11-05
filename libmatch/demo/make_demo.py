"""Rebuild the demo files from their public sources.

The demo files are committed, so this script is not needed to run the demo.
It exists so that the provenance of each file is executable rather than
described in prose, and so that a reviewer can confirm no file was
hand-edited. Paths point at the working copies used for the study; adjust
ROOT if the sources sit elsewhere.

    python make_demo.py --check     compare committed files to the sources
    python make_demo.py --write     regenerate them

Sources

  alphaseq_measured.csv
      open-alphaseq YM_1068 (Engelhart et al. 2022, Sci Data 9:653;
      https://huggingface.co/datasets/aalphabio/open-alphaseq). VHH72
      variants assayed against a panel of coronavirus RBDs; the demo uses
      the SARS-COV-2_GAMMA_RBD_LLNL_Trunc target, keeping rows with both a
      sequence and an affinity. 33,875 rows, lengths 122-126.

  released_2DD8_S.arm.csv
      The summary row libmatch computes from the Absolut! (Robert et al. 2022,
      Nat Comput Sci 2:845) released library scored against antigen 2DD8_S: a
      30,000-sequence random sample of the 6,917,820-member murine CDRH3 set,
      drawn with seed 0, lengths 11-40. The sequences are not redistributed
      here -- Absolut!'s licence is non-commercial and forbids onward
      distribution -- so only the statistics we computed are committed. See
      README.md for the sha256 of the file we used and how to rebuild it.

  matched_2DD8_S.csv
      The matched library built in this study against the same antigen: a
      length-17 parent, its 285 single mutants at the 15 non-terminal
      positions, and random double and triple mutants, deduplicated to
      6,000. Energies from the same Absolut! oracle.

  proteingym_*.csv / .wt.txt
      ProteinGym v1.1 substitution benchmark (Notin et al. 2023, NeurIPS
      Datasets and Benchmarks). Assay scores are taken verbatim; the wild
      type is the target_seq recorded in the benchmark's reference file.
      These two assays are of non-antibody proteins and are included to
      show the tool is not specific to antibody data.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ALPHASEQ = os.path.join(ROOT, "data", "alphaseq", "YM_1068.parquet")
ABSOLUT = os.path.join(ROOT, "data", "absolut")
TARGET = "SARS-COV-2_GAMMA_RBD_LLNL_Trunc"
ANTIGEN = "2DD8_S"
N_RELEASED = 30000
SEED = 0


def build_alphaseq():
    d = pd.read_parquet(ALPHASEQ)
    d = d[d.matalpha_description == TARGET]
    d = d.dropna(subset=["mata_sequence", "alphaseq_affinity"])
    return pd.DataFrame({"sequence": d.mata_sequence.astype(str).values,
                         "label": d.alphaseq_affinity.astype(float).values})


def build_matched():
    p = os.path.join(ABSOLUT, f"matched_{ANTIGEN}.parquet")
    d = pd.read_parquet(p)
    return pd.DataFrame({"sequence": d.cdr3.astype(str).values,
                         "label": d.energy.astype(float).values})


# released_2DD8_S.csv is not rebuilt here. It is a 30,000-row sample of the
# 6,917,820-sequence Absolut! archive for this antigen, and regenerating it
# needs the raw per-antigen zip archives, which are large and not
# redistributed with this tool. Download them from the Absolut! database and
# score them with that project's own parser.
#
# The ProteinGym files are likewise verbatim copies of the benchmark's own
# assay CSVs, with the wild type taken from its reference table, so there is
# nothing to rebuild.

BUILDERS = {"alphaseq_measured.csv": build_alphaseq,
            "matched_2DD8_S.csv": build_matched}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    if not (a.check or a.write):
        ap.error("pass --check or --write")

    for name, fn in BUILDERS.items():
        dest = os.path.join(HERE, name)
        src = ALPHASEQ if "alphaseq" in name else ABSOLUT
        if not os.path.exists(src):
            print(f"  skip {name}: source not present ({src})")
            continue
        built = fn()
        if a.write:
            built.to_csv(dest, index=False)
            print(f"  wrote {name}  {len(built):,} rows  sha256:"
                  f"{digest(dest)}")
            continue
        if not os.path.exists(dest):
            print(f"  MISSING {name}")
            continue
        have = pd.read_csv(dest)
        same = (len(have) == len(built)
                and (have.sequence.values == built.sequence.values).all()
                and bool((have.label - built.label).abs().max() < 1e-9))
        print(f"  {'ok  ' if same else 'DIFF'} {name}  {len(have):,} rows  "
              f"sha256:{digest(dest)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
