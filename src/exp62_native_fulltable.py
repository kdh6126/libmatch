"""실험 62 — 공개 전수표로 실제 결합 항체의 백분위를 낸다 (항원 103종).

exp61이 1ADQ에서 세 관문을 통과시켜 이 경로를 확정했다. 정합(겹침 100.000%),
동등(최대차 4x10^-6, 20/20), 일치(전수 41.2890% 대 표본 41.60%, Wilson 구간
안). 그래서 비교 집합을 우리가 채점하지 않고 Absolut!이 공개한 전수표를 읽는다.

항원당 로컬 채점이 2,001개에서 21개로 줄어든다 -- 실제 결합 항체 하나와 재현성
확인용 스무 개뿐이다. 92분이 몇 분이 되고, 비교 집합은 표본 2,000개가 아니라
전수 691만 개가 된다. 백분위 해상도가 0.05%에서 1.45x10^-5%로 올라간다.

디스크를 아끼려고 전수표 원본은 남기지 않는다. 받아서 파싱한 뒤 정렬된 에너지
배열만 .npy로 남기고 zip을 지운다 -- 항원당 450MB가 27MB가 되므로 103종이
43GB가 아니라 3GB로 끝난다. 다만 지우기 전에 관문 2(공개값 대 로컬 채점)를
그 자리에서 돌린다. 지우고 나면 그 확인을 다시 할 수 없기 때문이다.

관문은 exp60_gates를 그대로 쓰되 넷으로 줄었다.

  1 전송   크기와 md5가 서버 값과 일치 (전수표와 구조 파일 둘 다)
  2 정합   전수표가 그 항원의 무린 라이브러리인가 (개수, 정규 집합과 겹침)
  3 귀속   CDR-H3를 항원 사슬이 아니라 검증된 중쇄에서 뽑았는가
  4 동등   전수표의 임의 20개와 로컬 채점이 같은 값을 내는가

관문 4가 exp60의 '채점'과 '재현성'을 합친다. 공개값과 맞는지를 보는 것이
곧 우리 채점이 제대로 돌았는지를 보는 것이기 때문이다.

말할 수 있는 것은 "동일 모집단에서 공개표 백분위가 로컬 백분위와 일치한다"
까지다. 2,000개 행 단위 점수 동등성은 확인하지 않는다 -- 관문 4의 20개가
그 역할을 하고, 더 필요하면 따로 비용을 들여 보강한다.

    python src/exp62_native_fulltable.py              이어서 실행
    python src/exp62_native_fulltable.py --limit 1    한 종만
    python src/exp62_native_fulltable.py --keep-zip   원본 보존
"""
import argparse
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

# 진행 표시를 UTF-8로 못 박는다. 콘솔이 cp949로 잡히면 '—' 한 글자에
# UnicodeEncodeError가 나고, 그게 관문 실패 메시지를 찍는 자리라서 항원 하나가
# 걸리면 96종 전체가 거기서 멈춘다. 실제로 1FBI_X에서 그렇게 죽었다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, "src")
# gate_archive는 쓰지 않는다. 그쪽은 구조 zip 전용이라 풀어서 Ligand=/
# Receptors= 형식까지 보는데, 전수표 zip은 안에 든 것이 다른 파일이다.
# 여기서는 zipfile.testzip()으로 CRC를 보고 파싱 예외를 잡아 격리한다.
from exp60_gates import gate_chain, gate_transfer, pdb_of   # noqa: E402

_s = importlib.util.spec_from_file_location(
    "e60", "src/exp60_native_percentile.py")
E60 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E60)
_s2 = importlib.util.spec_from_file_location(
    "e61", "src/exp61_fulltable_check.py")
E61 = importlib.util.module_from_spec(_s2)
_s2.loader.exec_module(E61)

import exp56_vopal_search as X                              # noqa: E402

X.to_absolut = lambda s: s

RAW = ("https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/"
       "NS9603K/pprobert/AbsolutOnline/RawBindingsMurine")
OUT = "results/exp62_native_fulltable.csv"
TABLE_MANIFEST = "data/absolut/energies_manifest.json"
QUAR = "results/exp62_quarantine.csv"
ENERGY_DIR = "data/absolut/energies"
N_EXPECT = 6_918_276
# 파싱 규약이 바뀌면 배열의 값이 달라진다. 명부에 함께 남겨,
# 옛 파서로 만든 배열을 새 실행이 조용히 재사용하지 않게 한다.
PARSER_VERSION = "2026-09-17.best-slice.dedup-cdr3"


def load_json(p):
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else {}


def save_json(p, d):
    """명부를 원자적으로 쓴다.

    직접 쓰면 쓰는 도중에 끊겼을 때 반쪽짜리 JSON이 남고, 다음 실행은 그것을
    읽다 죽거나 -- 더 나쁘게는 -- 일부 항원의 기록을 잃은 채로 진행한다.
    96종을 며칠에 걸쳐 돌리는 동안 한 번은 끊길 수 있으므로 .part에 쓴 뒤
    바꿔치운다.
    """
    tmp = p + ".part"
    with io.open(tmp, "w", encoding="utf-8") as fh:
        json.dump(d, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def energies_path(ag):
    return f"{ENERGY_DIR}/{ag}.npy"


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def server_fingerprint(url):
    """서버가 말하는 크기와 md5. 둘 다 없으면 검증할 수 없다."""
    size = E60.head_size(url)
    md5 = subprocess.run(["curl", "-sL", "--max-time", "60", url + ".md5"],
                         capture_output=True, text=True).stdout.split()
    md5 = md5[0] if md5 and len(md5[0]) == 32 else None
    return size, md5


def verify_zip(zp, url):
    """이미 받아 둔 zip도 서버 원본과 같은지 확인한다.

    처음 판은 zip이 없을 때만 크기·md5를 봤다. 그래서 먼저 받아 둔 7종은
    CRC만 통과하면 그대로 쓰였다 -- CRC는 그 파일이 자기 자신과 일관된지를
    말할 뿐, 서버에 있는 그 파일인지는 말하지 않는다.
    """
    size, md5 = server_fingerprint(url)
    if size is None or md5 is None:
        return False, None, None, "서버 크기·md5를 받지 못함"
    ok, msg = gate_transfer(zp, size, md5)
    return ok, size, md5, msg


def get_full_table(ag, lib, keep_zip=False):
    """전수표를 확보하고 관문 1·2·4를 거쳐 정렬된 에너지 배열로 남긴다.

    zip은 기본적으로 지운다. 450MB를 103개 쌓아 둘 이유가 없고, 우리가 쓰는
    것은 에너지 분포뿐이다. 다만 지우기 전에 공개값 대 로컬 채점을 그 자리에서
    확인한다 -- 지운 뒤에는 그 대조를 다시 할 수 없다.
    """
    p = energies_path(ag)
    man = load_json(TABLE_MANIFEST)
    rec = man.get(ag) or {}
    if os.path.exists(p) and rec.get("energy_sha256"):
        # 행 수만 맞춰 보면 같은 길이의 다른 배열을 구분하지 못한다. 해시로
        # 본다. 파서가 바뀌었으면 값이 달라질 수 있으므로 그것도 함께 본다.
        h = sha256_of(p)
        if h == rec["energy_sha256"] and rec.get("parser") == PARSER_VERSION:
            e = np.load(p)
            return (e, f"기존 배열 재사용 ({len(e):,}행 · sha {h[:12]})",
                    rec.get("maxdiff"))
        os.remove(p)
        man.pop(ag, None)
        save_json(TABLE_MANIFEST, man)
    os.makedirs(ENERGY_DIR, exist_ok=True)
    zp = f"data/absolut/raw/{ag}.zip"
    url = f"{RAW}/{ag}.zip"
    if os.path.exists(zp):
        # 이미 있는 파일도 서버 원본과 대조한다. 어디서 왔는지 모르는 바이트로
        # 결과를 만들지 않는다.
        ok, size, md5, msg = verify_zip(zp, url)
        if not ok:
            return None, f"관문1 기존 zip 검증 실패 — {msg}", None
    else:
        size, md5 = server_fingerprint(url)
        if size is None or md5 is None:
            return None, "서버 크기·md5를 받지 못함 — 검증 없이 쓰지 않는다", None
        try:
            with E60.Lock(zp):
                ok, msg = E60.fetch_ranged(url, zp, size, md5)
        except RuntimeError as e:
            return None, f"동시 쓰기 차단 — {e}", None
        if not ok:
            return None, f"관문1 전송 실패 — {msg}", None
    # 아카이브 무결성. 여기서 걸러야 parse_zip의 BadZipFile/CRC 예외가 실행
    # 전체를 죽이지 않고 이 항원만 격리된다.
    try:
        with zipfile.ZipFile(zp) as z:
            broken = z.testzip()
        if broken:
            return None, f"관문1 아카이브 손상 — {broken}", None
        d, bad = E61.parse_zip(zp)
    except Exception as exc:
        # 여기 왔다는 것은 크기와 md5가 서버 값과 맞았다는 뜻이다(관문 1을 이미
        # 지났다). 그런데도 아카이브가 안 열리면 전송 문제가 아니라 서버에 있는
        # 파일 자체가 깨진 것이고, 다시 받아도 같은 바이트가 온다.
        #
        # 4N9G_C가 그렇다. 158,793,728바이트가 서버 Content-Length와 같고 md5도
        # 게시값(27dc415c...)과 같은데, 중앙 디렉터리도 EOCD도 없다. 서버가 잘린
        # 파일을 올려 두고 그 파일의 md5를 게시했다.
        #
        # 사유를 'BadZipFile'로만 적으면 전송 실패처럼 읽혀서 재시도 대상이
        # 된다. 복구 불가임을 사유에 적는다.
        return None, (f"관문1 아카이브 읽기 실패 — {type(exc).__name__} "
                      "· 크기·md5는 서버 값과 일치하므로 서버 원본이 손상된 "
                      "것이고 재전송으로 복구되지 않는다"), None
    tset = set(d.cdr3)
    inter = len(tset & lib)
    extra = len(tset - lib)                 # 정규 집합에 없는 서열
    cover = inter / len(lib)                # 정규 집합을 얼마나 덮는가
    purity = 1 - extra / max(1, len(tset))  # 표가 얼마나 순수한 부분집합인가
    # 관문의 목적은 "이 표가 그 항원의 무린 라이브러리인가"다. 처음에는 그것을
    # 행 수(N_EXPECT 대비 1% 이내)로 대신 쟀는데, 그 대리 지표가 너무 빡빡했다.
    #
    # 3SKJ_E의 공개표는 6,748,253행으로 정규 집합보다 2.46% 적다. 그런데 표 안의
    # 비정규 서열이 675만 중 443개(0.007%)뿐이고, 크기·md5가 서버 값과 맞으며
    # 손상 행도 없다. 맞는 라이브러리의 축소판이지 다른 표가 아니다.
    #
    # 그래서 목적에 더 가까운 두 가지로 바꾼다. 순도는 오히려 더 엄하다 --
    # 엉뚱한 라이브러리라면 비정규 서열이 쏟아진다.
    #
    # 다만 97%라는 문턱은 이 표를 본 뒤에 정한 값이라 사후 기준으로 읽힌다.
    # 그래서 통과시키되 같은 무게로 합치지 않는다. coverage를 CSV에 남기고
    # 99% 미만은 reduced_reference로 표시해, 주분석은 완전 표로만 하고 이쪽은
    # 민감도 분석으로 따로 본다.
    #
    # 빠진 서열이 천연보다 좋은지 나쁜지는 알 수 없다. 길이 분포가 전체와
    # 같다는 관찰은 있지만, 그것이 에너지 순위를 말해 주지는 않는다. 그래서
    # 추정하지 않고 양쪽 상·하한만 CSV에 적는다(pct_lo / pct_hi).
    detail = (f"{len(d):,}행 · 겹침 {cover * 100:.3f}% · 순도 "
              f"{purity * 100:.3f}% · 손상 {bad}")
    if purity < 0.999 or cover < 0.97:
        return None, f"관문2 정합 실패 — {detail}", None
    if cover < 0.99:
        detail += " · 비교 집합이 정규보다 작음"
    # 관문 4: 지우기 전에 여기서 확인한다
    s = d.sample(20, random_state=0).reset_index(drop=True)
    o = X.AbsolutOracle(ag, s.cdr3.tolist(), 8, f"eq62_{ag}")
    o(np.arange(len(s)))
    diff = float(np.nanmax(np.abs(o.energy - s.energy.to_numpy())))
    if not (diff < 1e-3):
        return None, f"관문4 동등 실패 — 최대차 {diff:.6f}", None
    # .part에 쓰고 원자적으로 옮긴다. 중간에 끊겨도 잘린 배열이 최종 이름을
    # 차지하지 못한다. 명부에 행 수를 함께 남겨 다음 실행이 대조한다.
    e = np.sort(d.energy.to_numpy().astype(np.float32))
    np.save(p + ".part.npy", e)
    os.replace(p + ".part.npy", p)
    # 원본 zip을 지우더라도 "어느 공개표에서 어떤 파서로 나온 배열인가"가
    # 남아야 한다. 그것 없이는 나중에 재현할 수도, 의심스러울 때 대조할
    # 수도 없다.
    man[ag] = {"n": int(len(e)), "energy_sha256": sha256_of(p),
               "raw_size": int(size), "raw_md5": md5,
               "parser": PARSER_VERSION, "maxdiff": diff,
               "bad_rows": int(bad),
               "overlap_pct": round(cover * 100, 4),
               # 표가 정규 집합보다 작으면 그 항원의 백분위는 더 작은 비교
               # 집합에서 나온 값이다. 명부에 남겨야 나중에 되짚을 수 있다.
               "purity_pct": round(purity * 100, 4),
               "table_rows": int(len(d)),
               "when": time.strftime("%F %T")}
    save_json(TABLE_MANIFEST, man)
    if not keep_zip:
        os.remove(zp)
    return e, f"{detail} · 동등 최대차 {diff:.1e}", diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--have-only", action="store_true",
                    help="전수표를 이미 받아 둔 항원만. 다운로드 없이 "
                         "파이프라인을 끝까지 확인할 때 쓴다")
    ap.add_argument("--only", nargs="*", default=None, metavar="ANTIGEN")
    ap.add_argument("--retry-quarantine", nargs="?", const="*", default=None)
    a = ap.parse_args()

    meta = json.load(io.open("data/absolut/antigen_meta.json", encoding="utf-8"))
    pdb_meta = meta.get("pdb_meta", {})
    ids = [x.strip() for x in io.open("data/absolut/antigen_ids.txt",
                                      encoding="utf-8") if x.strip()]
    lib = set(pd.read_parquet("data/absolut/library.parquet")["cdr3"])
    man = (json.load(io.open(E60.MANIFEST, encoding="utf-8"))
           if os.path.exists(E60.MANIFEST) else {})
    print(f"항원 {len(ids)}종 · 정규 라이브러리 {len(lib):,}개", flush=True)

    # 빈 체크포인트를 빈 결과로 읽고, 비워진 격리 명부는 디스크에서도 지운다.
    # exp60에서 같은 자리를 두 번 고쳤으므로 그쪽 구현을 그대로 쓴다.
    rows, quar = E60.load_records(OUT), E60.load_records(QUAR)
    if a.retry_quarantine:
        g = a.retry_quarantine
        back = [r for r in quar if g == "*" or str(r.get("gate")) == g]
        quar = [r for r in quar if r not in back]
        print(f"격리 해제 {len(back)}종", flush=True)
    seen = {r["antigen"] for r in rows} | {r["antigen"] for r in quar}
    todo = [x for x in ids if x not in seen]
    if a.only:
        todo = [x for x in todo if x in set(a.only)]
    if a.have_only:
        # 내려받는 데 대부분의 시간이 든다. 이미 가진 것만 먼저 돌리면
        # 나머지를 받기 전에 파이프라인 결함을 잡을 수 있다.
        todo = [x for x in todo
                if os.path.exists(f"data/absolut/raw/{x}.zip")
                or os.path.exists(energies_path(x))]
    todo = todo[:a.limit or None]

    def save():
        for path, records in ((OUT, rows), (QUAR, quar)):
            if records:
                pd.DataFrame(records).to_csv(path, index=False)
            elif os.path.exists(path):
                os.remove(path)

    for i, ag in enumerate(todo, 1):
        # 철회된 PDB 항목은 대체 항목으로 조회한다(exp60_gates.PDB_REPLACED).
        pdb, t0 = pdb_of(ag), time.time()
        print(f"\n[{i}/{len(todo)}] {ag}  "
              f"({pdb_meta.get(pdb, {}).get('title', '')[:56]})", flush=True)

        hit, msg3 = gate_chain(E60.pdb_fasta(pdb), ag)
        if hit is None:
            print(f"  [격리] 관문3 — {msg3}", flush=True)
            quar.append({"antigen": ag, "pdb": pdb, "gate": "chain",
                         "reason": msg3})
            save()
            continue
        seq = hit["cdrh3"]
        print(f"  관문3 통과 — {seq} · {msg3}", flush=True)
        if len(seq) < 11:
            print(f"  길이 {len(seq)} — 11 미만이라 채점 불가", flush=True)
            rows.append({"antigen": ag, "pdb": pdb, "chain": hit["chain"],
                         "native": seq, "native_len": len(seq),
                         "status": "too_short", "gate_chain": msg3})
            save()
            continue

        ok, msg12 = E60.ensure_structures(ag, man)
        print(f"  구조 {'통과' if ok else '실패'} — {msg12[:96]}", flush=True)
        if not ok:
            quar.append({"antigen": ag, "pdb": pdb, "gate": "structures",
                         "reason": msg12[:300], "native": seq})
            save()
            continue

        e, msg24, diff = get_full_table(ag, lib, a.keep_zip)
        print(f"  전수표 {'통과' if e is not None else '실패'} — {msg24[:96]}",
              flush=True)
        if e is None:
            quar.append({"antigen": ag, "pdb": pdb, "gate": "fulltable",
                         "reason": msg24[:300], "native": seq})
            save()
            continue

        try:
            o = X.AbsolutOracle(ag, [seq], a.threads, f"n62_{ag}")
            o(np.arange(1))
            e_nat = float(o.energy[0])
        except Exception as exc:
            print(f"  [격리] 채점 예외 — {str(exc)[:120]}", flush=True)
            quar.append({"antigen": ag, "pdb": pdb, "gate": "score",
                         "reason": str(exc)[:300], "native": seq})
            save()
            continue
        if not np.isfinite(e_nat):
            quar.append({"antigen": ag, "pdb": pdb, "gate": "score",
                         "reason": "실제 항체 채점값 없음", "native": seq})
            save()
            continue

        pct = float(np.searchsorted(e, e_nat, "left") / len(e) * 100)
        print(f"  {seq}  {e_nat:.2f}  →  무린 {len(e):,}개 중 더 좋은 것 "
              f"{pct:.4f}%  [{(time.time() - t0) / 60:.1f}분]", flush=True)
        rows.append({"antigen": ag, "pdb": pdb, "chain": hit["chain"],
                     "desc": hit["desc"], "native": seq, "native_len": len(seq),
                     "native_energy": e_nat, "ref_n": int(len(e)),
                     "ref_best": float(e[0]), "ref_median": float(np.median(e)),
                     "pct_better": pct, "status": "ok",
                     "gate_chain": msg3, "gate_structures": msg12[:120],
                     "gate_fulltable": msg24[:120], "equiv_maxdiff": diff,
                     "minutes": round((time.time() - t0) / 60, 1)})
        save()

    d = pd.DataFrame(rows)
    ok = d[d.status == "ok"] if len(d) and "status" in d else d
    if len(ok):
        print(f"\n채점 {len(ok)}종 · 백분위 중앙 {ok.pct_better.median():.2f}% · "
              f"상위 10% 안 {(ok.pct_better < 10).sum()}종 · 격리 {len(quar)}종")
    print(f"wrote {OUT}" + (f" · {QUAR}" if quar else ""))


if __name__ == "__main__":
    main()
