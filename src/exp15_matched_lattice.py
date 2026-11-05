"""Experiment 15 -- remove the confound between 'lattice' and 'library structure'.

Our lattice arm is 6.9M unrelated CDRH3s of varying length; our real arm is a
mutational neighbourhood around one parent. Aligned one-hot is close to ideal
for the second and meaningless for the first, so "one-hot wins on the lattice,
PLM wins on real data" may be a statement about library structure rather than
about the benchmark being synthetic.

This builds a lattice library with the SAME structure as the real one: all
single mutants of a parent CDRH3 plus random double and triple mutants, which
is exactly how the AlphaSeq designs were made. Scored with the Absolut! binary,
so the labels are genuine lattice energies.

If the PLM wins here too, the inversion was about library structure.
If one-hot still wins, it was about the lattice.
"""
import argparse, os, subprocess, sys
import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
SRC = "tools/Absolut-main/src"
WSL_SRC = "/mnt/d/ubuntu/BIO/Infectious/tools/Absolut-main/src"


def mutants(parent, n_total, rng, core=(1, None)):
    """All k=1 mutants, then random k=2 and k=3 -- the AlphaSeq design recipe."""
    lo, hi = core[0], (core[1] or len(parent) - 1)
    pos = list(range(lo, hi))
    out = {parent}
    for i in pos:                                        # every single mutant
        for a in AA:
            if a != parent[i]:
                out.add(parent[:i] + a + parent[i + 1:])
    # out 에는 모항체가 먼저 들어가 있으므로 하나 뺀다. 이 수를 그대로
    # 찍었더니 15자리 x 19 = 285 여야 할 값이 286 으로 보고돼
    # 원고에도 그렇게 실렸다.
    n1 = len(out) - 1
    while len(out) < n_total:
        k = 2 if rng.random() < 0.6 else 3
        s = list(parent)
        for i in rng.choice(pos, size=min(k, len(pos)), replace=False):
            s[i] = AA[int(rng.integers(20))]
        out.add("".join(s))
    return sorted(out), n1


def score_with_absolut(seqs, antigen, tag):
    """Run the Absolut! binary in repertoire mode and read back best-slice energies."""
    src = f"{SRC}/{tag}_input.txt"
    with open(src, "w") as f:
        for i, s in enumerate(seqs):
            f.write(f"{i}\t{s}\n")
    cmd = (f"cd {WSL_SRC} && rm -f {antigen}FinalBindings_Process_*.txt && "
           f"./AbsolutNoLib repertoire {antigen} {tag}_input.txt 16")
    print(f"  Absolut! 채점 시작 ({len(seqs):,}개) ...", flush=True)
    # capture_output swallowed the binary's stderr, so a scoring run that
    # partially failed looked identical to one that succeeded. Keep the streams
    # and surface them on a non-zero exit rather than discovering the gap later
    # in the energies.
    p = subprocess.run(["wsl", "-d", "Ubuntu-22.04", "--", "bash", "-lc", cmd],
                       capture_output=True, text=True, timeout=None)
    if p.returncode != 0:
        raise RuntimeError(
            f"Absolut! 비정상 종료 (코드 {p.returncode})\n"
            f"stderr: {p.stderr[-2000:]}")
    if p.stderr.strip():
        print(f"  [Absolut! stderr] {p.stderr.strip()[-500:]}", flush=True)
    out = [p for p in os.listdir(SRC) if p.startswith(f"{antigen}FinalBindings")]
    if not out:
        raise RuntimeError("Absolut! 출력 파일이 없습니다")
    best = {}
    for p in out:
        with open(f"{SRC}/{p}", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                q = line.rstrip("\n").split("\t")
                if len(q) < 5 or q[2] != "true" or not q[1].isalpha():
                    continue
                try:
                    best[q[1]] = float(q[4])
                except ValueError:
                    pass
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--antigen", default="2DD8_S")
    ap.add_argument("--parent", default="CARDRLLWLRLWYFDVW")
    ap.add_argument("--n", type=int, default=20000)
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    seqs, n1 = mutants(a.parent, a.n, rng)
    print(f"모항체 {a.parent} (길이 {len(a.parent)})")
    print(f"설계 {len(seqs):,}개 — 단일변이 {n1:,}, 나머지 이중·삼중")

    energies = score_with_absolut(seqs, a.antigen, "mut")
    df = pd.DataFrame({"cdr3": seqs})
    df["energy"] = df["cdr3"].map(energies)
    got = int(df["energy"].notna().sum())
    print(f"채점 완료: {got:,}/{len(df):,} "
          f"(에너지 {df.energy.min():.2f} ~ {df.energy.max():.2f})")
    # This line used to read `df = df[df.energy.notna()]`, which quietly
    # shrank the library whenever the scorer failed. 1NCA_N was built that way
    # and its parent energy came out 13 kcal from the released oracle; the
    # discrepancy was only caught much later, by hand. A partial library is not
    # a smaller library, it is a biased one, so refuse to write it.
    if got != len(df):
        raise SystemExit(
            f"채점 실패 {len(df) - got:,}개. 부분 결과는 저장하지 않습니다 — "
            "실패 원인을 확인한 뒤 다시 실행하십시오.")
    if a.parent not in energies:
        raise SystemExit("모항체가 채점되지 않았습니다. 저장하지 않습니다.")
    df["parent_margin"] = df["energy"] - float(energies[a.parent])
    df.to_parquet(f"data/absolut/matched_{a.antigen}.parquet", index=False)
    print(f"wrote data/absolut/matched_{a.antigen}.parquet  {df.shape}")
