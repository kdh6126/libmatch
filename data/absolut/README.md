# The matched libraries are not included here

Each file below holds 6,000 CDRH3 sequences that we designed, labelled with
binding energies computed by running Absolut! (Robert et al. 2022,
*Nat Comput Sci* 2:845).

Absolut! is distributed under a non-commercial licence that forbids onward
distribution of the software. That clause does not, by itself, grant anyone
permission to redistribute labels computed with it, and the project's site
carries an All Rights Reserved notice. We therefore do not pass these files
on. Absence of a prohibition is not a grant.

## Rebuilding them

Obtain Absolut! and the per-antigen tables from <https://greifflab.org/absolut/>
under its own licence, then:

```
python src/exp68_more_matched.py
```

The script writes the files below. Compare your sha256 against this table to
confirm you rebuilt what we used. The energies are deterministic, so they
should agree exactly; the sequence sets are drawn with fixed seeds.

| File | Rows | Bytes | sha256 |
|---|---|---|---|
| `unreliable/matched_1NCA_N.parquet` | 6,000 | 78,325 | `48b6ffd99053c1dab892774fc4b0fa003aeff573259cfe1bbdd7bbc3fb2c571f` |
| `matched_1ADQ_A.parquet` | 6,000 | 80,746 | `0e1cd0151617bf8c4f047a023eee906470113aa4bb9e2a419b34004dab35364c` |
| `matched_1ADQ_A_weak.parquet` | 300 | 4,833 | `338788bbb0f68db2f9bf1a2c48e0fef5f4180b9f434785e4843dbb2a00b08388` |
| `matched_1NCA_N.parquet` | 6,000 | 81,587 | `1c178da190db1aa70a4ac1638eddd7c58986e08976b3c4fa37c9ee400825fb81` |
| `matched_1NCA_N_weak.parquet` | 300 | 4,954 | `9f5f9c07e43f55c2656394e3dcdb2d268bb60240962950f46c8990d4a0858bdc` |
| `matched_1OSP_O.parquet` | 6,000 | 77,562 | `fa14136bf77fa63a4aaaa56decfd45cdbee068e9e0c54fb52e61662f62922f52` |
| `matched_1OSP_O_weak.parquet` | 300 | 4,929 | `b7db4263e6bddd008646eae510fb2fef7ad6be896760a492a8c6d9eae70e9a43` |
| `matched_1ZTX_E.parquet` | 6,000 | 78,589 | `2dd9bee11027720ccc538d4cdee5557a5e11cdae9e44a170934af4f5ca1bf8b0` |
| `matched_1ZTX_E_weak.parquet` | 300 | 4,984 | `a1a2603670501000335cd44293f4fce351ac3803f8e33760987ca47574a5d1e8` |
| `matched_2DD8_S.parquet` | 6,000 | 80,282 | `49cf34d25ed9a149d0f158357446c444ed1a55e741f2df007f15b559f6da93c3` |
| `matched_2DD8_S_rank1parent.parquet` | 6,000 | 81,981 | `f9b726ea9c50bdc8805389eb0938866621e649f791e568909ebc57f8c120ae3e` |
| `matched_2DD8_S_weak.parquet` | 300 | 4,914 | `0081e2043a6bb8bf01e4d290b1ad3837a328578b7c0c57bdf919e3d3af02a62c` |
| `matched_2Q8A_A.parquet` | 6,000 | 79,677 | `dd70226415fff5557cee3548d062ad58485b6cee0ba7cc7221fb5139d11bdd53` |
| `matched_3VRL_C.parquet` | 6,000 | 80,331 | `8f905a28b7d5bb68f5e82d9f11d1498c1575cfe8443fa98f21ccb8bbe824ad43` |
| `matched_3VRL_C_weak.parquet` | 300 | 4,855 | `951adb107c1b3590ec4bb43af187c0e80bfe0ccdf37398e95fbd31e3c6c8b880` |

`unreliable/` holds the first 1NCA_N library, which a scoring run wrote while
failing. It was quarantined and rebuilt. Supplementary Table S10 compares the
two; rebuilding both needs the same script and the quarantined scoring path
described there.

## What is here instead

Everything computed *from* these libraries that does not carry Absolut!
energies: the result tables in `results/`, from which every reported number
is taken, and the summary rows the tool computes. Those are our own
measurements of our own libraries and are covered by this repository's
MIT licence.
