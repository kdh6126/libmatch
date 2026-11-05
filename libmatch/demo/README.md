# Demo datasets

Five entries, from four sources. Two are measured, two are simulated, and two
are of proteins unrelated to antibodies. One is present as a summary row
rather than as data, for the licence reason given below. Together they let `run_demo.py` check
that the axis verdicts are what they should be, and they let a reader see the
tool behave on data it was not developed against.

`make_demo.py --check` rebuilds the two files that can be rebuilt from local
sources and confirms they match what is committed here.

| File | Rows | Lengths | Source |
| --- | --- | --- | --- |
| `alphaseq_measured.csv` | 33,875 | 122–126 | open-alphaseq YM_1068 |
| `matched_2DD8_S.arm.csv` | 1 summary row | — | this study, Absolut! oracle, summarised |
| `released_2DD8_S.arm.csv` | 1 summary row | — | Absolut! released library, summarised |
| `proteingym_RASH_HUMAN_Bandaru_2017.csv` | 3,134 | 189 | ProteinGym |
| `proteingym_TPK1_HUMAN_Weile_2017.csv` | 3,181 | 243 | ProteinGym |

## alphaseq_measured.csv

Yeast two-hybrid affinity measurements from the open-alphaseq release
(Engelhart et al. 2022, *Sci Data* 9:653;
<https://huggingface.co/datasets/aalphabio/open-alphaseq>). Campaign YM_1068
assays VHH72 variants against a panel of coronavirus receptor-binding
domains. The demo takes the `SARS-COV-2_GAMMA_RBD_LLNL_Trunc` target and keeps
rows that have both a sequence and an affinity, giving 33,875 measurements of
33,667 distinct sequences. `alphaseq_affinity` is the label.

Only 208 sequences are measured more than once, so the noise estimate rests on
0.6% of the set; the report says so rather than presenting the figure as the
whole arm's noise.

## matched_2DD8_S.arm.csv and released_2DD8_S.arm.csv

Two libraries scored against the same antigen by the same oracle, so that
everything except the library design is held fixed.

`released_2DD8_S.arm.csv` stands in for a 30,000-sequence sample (seed 0) of
the Absolut! murine CDRH3 set of 6,917,820 sequences, scored against antigen
2DD8_S (Robert et al. 2022, *Nat Comput Sci* 2:845). Its members are unrelated
to one another and their lengths run 11–40, so it cannot be aligned.

The sequences themselves are not redistributed here. Absolut! is released
under a non-commercial licence that does not permit onward distribution, and
nothing in that sample is our own work beyond drawing it, so we have no right
to pass it on. What is committed instead is the single summary row that
`libmatch` computes from it. The five axes are functions of that summary, not
of the sequences, so every verdict in `run_demo.py` is identical to the one
the full file gives.

To rebuild the full file: obtain the per-antigen archive for 2DD8_S from the
Absolut! database (<https://greifflab.org/absolut/>) under its own licence,
draw 30,000 rows with seed 0, and keep the CDR3 and energy columns. The file
we used has
sha256 `3af5f9c2ee6a27aa7b28362d1006e52e3e01512a59751215194721bc4f16cbe8`.

`matched_2DD8_S.arm.csv` summarises the library built in this study to resemble
a real affinity-maturation campaign against the same antigen: a length-17
parent, all 285 single mutants at its 15 non-terminal positions, and random
double and triple mutants, deduplicated to 6,000 sequences.

Its sequences are ours, but their labels are energies computed by Absolut!, and
nothing in that project's licence grants us the right to redistribute them. Only
the summary row is committed. The full file has
sha256 `07550a362d553669f1a09cba99c08e1f4becb0ceefffda41c83f055ac7cd4178`
and is rebuilt by `python src/exp68_more_matched.py` once Absolut! is obtained
from its own source; see `data/absolut/README.md`.

Comparing these two is the case the tool exists for. Four of the five axes are
mismatched, so any difference measured between them has four candidate
explanations.

## ProteinGym assays

Two substitution assays from the ProteinGym v1.1 benchmark (Notin et al. 2023,
*NeurIPS Datasets and Benchmarks*), of human HRAS and human thiamin
pyrophosphokinase. Neither is an antibody.

They are distributed as mutation notation (`T2L`, `E2Q`) rather than
sequences, so they also exercise the expansion path: the `.wt.txt` file beside
each holds the wild-type sequence, taken from the benchmark's own reference
table, and `libmatch` checks that the wild-type residue named in every
mutation agrees with it.

Both are correctly read as mutational neighbourhoods around a wild type and
are close to the matched lattice library on size, structure, noise and budget.
They differ on dimension — 165 and 241 varying positions against 15 — which is
the axis a comparison of this kind most often leaves uncontrolled.

## Licences

`proteingym_*.csv` and `alphaseq_measured.csv` are excerpts of two public
datasets, redistributed here only so the demo runs without a download.
ProteinGym is MIT-licensed; open-alphaseq is released by its authors for
academic and commercial use. Both remain under the terms set by their own
source, and the original papers — not this tool — are what to cite when
using the data.

No Absolut! data is redistributed, and nothing derived from it carries energies.
`matched_2DD8_S.arm.csv` and `released_2DD8_S.arm.csv` hold only summary
statistics we computed, which are covered by this repository's MIT licence.
The sequences and energies behind them are rebuilt from your own licensed
Absolut! download; `data/absolut/README.md` gives the command and the hashes.
