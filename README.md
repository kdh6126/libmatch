# libmatch

Code, result tables and reproduction map for

> **Library size and sequence relatedness confound comparisons between
> synthetic benchmarks and experimental antibody data**
> Dohun Kim, Minho Bae — Electronics and Telecommunications Research Institute

and for `libmatch`, the tool that measures five operational axes of a
sequence–label dataset — size, structure, dimension, measurement noise and
training budget — and reports which of them differ between two datasets, or
cannot be measured from the data as released.

## What the article claims, and what is here to check it

A synthetic benchmark and a measured dataset usually differ in several ways at
once — label origin, size, structure, alignability and measurement noise — so a
performance gap between them cannot be attributed to any one of those. These are
properties of the libraries, and are not the same list as the five operational
axes `libmatch` measures from a file. The article builds matched libraries that
hold four of the five properties fixed and re-measures four reported
discrepancies under matching.

Two of the four disappear, a third proves unmeasurable, and the fourth narrows
by three orders of magnitude without closing. That last result is the one to
scrutinise, and every number behind it is in `results/`.

Each number in the article and its supplement comes from a table in
`results/`, written by one script in `src/`. **[REPRODUCTION.md](REPRODUCTION.md)**
maps every figure, table and check to the script that produced it, lists the
dependency order, and records the environment the bundle was verified in. It
is generated from the source, so it cannot drift from the code.

## Layout

| | |
|---|---|
| `libmatch/` | the tool: five-property measurement, demo, regression test |
| `src/` | the 69 scripts the article depends on, of 258 in the working repository |
| `results/` | 57 result tables — every reported number reads from one of these |
| `data/absolut/` | no data: the specification and sha256 of each of the 15 matched libraries, and the command that rebuilds them |
| `manuscript/` | LaTeX sources and the 11 figures they include |
| `CHECKSUMS.txt` | sha256 of all 162 files |

## Checking it without the oracle

These four need neither the Absolut! binary nor the 6.9-million-row antigen
tables, and finish in seconds:

```
pip install -r requirements.txt
python src/audit_new_claims.py    # every stated number against its table
python src/check_si_refs.py       # supplementary cross-references
python src/check_sources.py       # source integrity
python libmatch/run_demo.py       # tool regression test
```

`requirements.txt` records the Python version and OS the pins were verified
on. Nothing reported in the article needs a GPU.

## Using the tool on your own data

```
pip install -e libmatch
libmatch profile your_data.csv
libmatch match benchmark.csv measured.csv
```

`profile` reports where one dataset sits on the five axes; `match` reports
which axes differ between two, and by how much. It does not say which dataset
is correct — only that a mismatch exists and how large it is. See
[libmatch/README.md](libmatch/README.md).

## What is not here

The primary datasets are third-party and are downloaded rather than
redistributed: the Absolut! binary and per-antigen tables
(<https://greifflab.org/absolut/>), open-alphaseq
(<https://huggingface.co/datasets/aalphabio/open-alphaseq>), AlphaBind
(<https://github.com/A-Alpha-Bio/alphabind>), and the c-Met panning reads
(BioProject PRJNA607865). The Absolut! binary runs under Linux only.

Small excerpts of ProteinGym and open-alphaseq *are* included, under
`libmatch/demo/`, so that the tool's regression test runs without a download;
both sources permit that. Each stays under the terms set by its own source.
Nothing from Absolut! is included there: the one case that needs its released
library enters as a summary row instead, which leaves every verdict in the
regression test unchanged. [libmatch/demo/README.md](libmatch/demo/README.md)
records the provenance and terms of every file, and which paper to cite for
it.

The matched libraries this study built are not here either, for the same
reason: their sequences are ours, but their labels are energies computed by
running Absolut!. `data/absolut/README.md` carries each one's specification,
row count, byte size and sha256, and the command that rebuilds it. That
includes the quarantined first version of `matched_1NCA_N` and its rebuilt
replacement; which of the two the article uses, and why, is stated in the
supplementary information.

## Licence

**[LICENSE](LICENSE) is MIT, and it covers the code only.** Third-party raw
data follows the access and licence conditions set by its own source, whatever
this repository's licence says. Where that distinction bites, it is stated
below rather than left to the reader to notice.

| Path | Terms |
|---|---|
| `libmatch/`, `src/` | MIT. Our code. |
| `manuscript/` | Our text and figures. |
| `results/` | Our analysis outputs: summary statistics, test results and percentiles we computed. Free to reuse with citation. No row of any table pairs a sequence from a third-party library with that library's labels. |
| `data/absolut/` | No data. A specification, a per-file sha256 and a rebuild command, so that a reader with their own licensed Absolut! download can reconstruct the libraries we used and check they got the same files. |
| `libmatch/demo/` | Mixed, and not all MIT. Per-file provenance and terms in [libmatch/demo/README.md](libmatch/demo/README.md). |

Nothing computed with Absolut! is redistributed here: not the binary, not the
per-antigen tables, not the released library, and not the matched libraries we
built with it. Absolut! is distributed under a non-commercial licence that
forbids onward distribution of the software, and that clause grants no one the
right to redistribute labels computed with it. Absence of a prohibition is not
a grant, so this package claims no right that was not given.

That exclusion costs nothing in reproducibility. The energies are
deterministic and the sequence sets are drawn from fixed seeds, so a rebuild
from your own download is identical to the files we used, and the digests in
`data/absolut/README.md` establish whether it is. Where a comparison needs the
released library, what is committed is the summary row `libmatch` computes
from it — which is what the five axes are functions of, so every verdict in
the regression test is unchanged.

Obtain Absolut! from <https://greifflab.org/absolut/> and read its licence
before use. If you need terms broader than non-commercial research use for
anything derived from it, that is a question for the Absolut! authors and
Inven2, not for us.

## Citing

See [CITATION.cff](CITATION.cff).
