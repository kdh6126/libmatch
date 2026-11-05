# libmatch

Check whether two sequence–label datasets differ only in the way a comparison
intends.

When a method scores better on dataset A than on dataset B, that gap is
interpretable only if A and B differ along the axis the comparison was about.
Putting a synthetic benchmark beside a measured one makes this hard, because
the two usually differ on several axes at once: a synthetic set is often
millions of mutually unrelated sequences, a measured set is often tens of
thousands of variants around a single parent. They differ not only in where
the labels came from but simultaneously in size, in structure, in whether they
can be aligned, and in measurement noise.

`libmatch` counts five such axes and reports which are mismatched. It does not
say which dataset is correct — it shows that a mismatch exists and how large
it is, and leaves the decision of whether to control for it or report it to
the user.

| Axis | What it measures |
| --- | --- |
| size | number of sequences |
| structure | variants around one parent, or a set of unrelated sequences |
| dimension | number of positions that vary, where the set can be aligned |
| noise | noise-to-signal ratio, where replicate measurements exist |
| budget | the range of training sizes available |

The third axis is the one most often missed. Matching size and structure
leaves the number of mutated positions untouched, and that axis changes how
well a surrogate model fits, which in turn changes search performance.

## Install

```
pip install .
```

Requires Python 3.9+, numpy and pandas. `scipy` and `scikit-learn` are needed
only for the optional `--curve` mode and are declared as an extra:

```
pip install ".[curve]"
```

## Use

Compare two datasets:

```
libmatch report A.csv B.csv
libmatch report A.csv B.csv --seq-col cdr3 --label-col energy
libmatch report A.csv B.csv --curve --json out.json
```

Or measure a collection once and ask which of its members could be compared
against a dataset you intend to deploy on:

```
libmatch profile assays/*.csv --catalogue catalogue.csv
libmatch match my_target.csv --catalogue catalogue.csv --top 10
```

`match` ranks the catalogue by how many of the five axes are mismatched, and
among ties by how far apart the comparable axes are, so the datasets at the top
are the ones that differ from your target in the fewest ways that would
confound a comparison. Its exit status is 0 only if some catalogued dataset
matches on all five axes.

```
$ libmatch match measured.csv --catalogue proteingym.csv --top 3

Target  measured.csv
        33,875 sequences, mutational neighbourhood, 89 varying positions

Closest of 217 catalogued datasets

  dataset                       off  dist  size  stru  dime  nois  budg
  PABP_YEAST_Melamed_2013       1/5   0.49  .     .     .     X     .
  Q8WTC7_9CNID_Somermeyer_2022  2/5   0.76  .     .     X     X     .
  Q6WV12_9MAXI_Somermeyer_2022  2/5   0.79  .     .     X     X     .
```

Input is CSV, TSV or Parquet with a sequence column and a label column; the
columns are guessed from their names, or from their dtypes if the names are
unfamiliar. Deep mutational scanning data distributed as mutation notation
(`T2L`, `A1P:D2N`) is expanded automatically when a wild-type sequence is
supplied:

```
libmatch report assay.csv library.csv --wt-file wildtype.txt
libmatch report assay.csv other.csv  --wt-file wt_a.txt --wt-file-b wt_b.txt
```

The wild-type residue named in each mutation is checked against the sequence
given, and any disagreement is reported — an off-by-one in position numbering
is the usual cause.

The exit status is 1 if any axis is mismatched and 0 otherwise, so the tool
can gate a pipeline.

## Example

A yeast-display affinity dataset against a lattice-simulation library built to
match it:

```
$ libmatch report demo/alphaseq_measured.csv demo/matched_2DD8_S.csv

Five axes

  axis       A                         B                         verdict
  size       33,875                    6,000                     mismatched  (6x apart)
  structure  mutational neighbourhood  mutational neighbourhood  matched
  dimension  89 positions              15 positions              mismatched  (5.9x apart)
  noise      0.326                     0.000                     mismatched  (difference 0.326)
  budget     up to 25,406              up to 4,500               mismatched  (little overlap ...)

4 of 5 axes mismatched
```

The library was built to match the measured set, and on structure it does.
The dimension axis is still 5.9-fold apart, and that is the axis that governs
how a surrogate model fits.

## How the axes are decided

**Structure** is not decided by a fixed identity threshold, because identity
depends on length: two mutations in 17 residues gives 0.88, five in 125 gives
0.96. The same kind of library would land on either side of any fixed cut. So
sequences shuffled to the same composition provide a baseline, and the
reported *concentration* is where the observed identity sits between that
baseline and 1. A set is called a mutational neighbourhood at concentration
0.60 or above. Identity is computed with an alignment-free ratio so that
libraries with indels are not thrown out for having unequal lengths.

**Noise** is estimated as the root-mean-square within-sequence standard
deviation over sequences measured more than once, divided by the standard
deviation of the labels. Where every sequence appears exactly once the noise
is set to 0 rather than left unknown: deterministic oracles such as lattice
simulations look like this, and treating "no replicates" as "not comparable"
would blank out the axis in exactly the synthetic-versus-measured comparison
it exists for. If replicates existed but were averaged away before export,
run the tool on the raw data instead — the report says so.

**Dimension** is counted at the modal length, so it stays defined when lengths
are mixed; the report states what fraction of the data that covers.

## Demo and tests

`run_demo.py` runs four dataset pairs and asserts the verdict on each axis, so
a change to the axis logic fails there rather than silently in a paper.

```
python run_demo.py
python run_demo.py --verbose   # full report for each case
```

The four cases run from the data the tool was written on out to data unrelated
to antibodies:

1. two libraries against the same antigen, one built to a recipe and one taken
   from a published benchmark — 4 of 5 axes mismatched;
2. a measured yeast-display dataset against the matched lattice library —
   structure agrees, dimension is 5.9× apart;
3. and 4. two ProteinGym deep mutational scans of non-antibody proteins,
   distributed as mutation notation — both correctly read as neighbourhoods
   around a wild type, with dimension 11× and 16× apart.

See [demo/README.md](demo/README.md) for the provenance of each file.

## Licence

MIT. See [LICENSE](LICENSE).
