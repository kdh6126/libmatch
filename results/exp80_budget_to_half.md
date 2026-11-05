# exp80 상위 1%의 절반을 찾는 데 드는 측정 횟수

무작위 대비 배수는 정합 격자에서 측정되지 않는다. 예산 1,000에서 유도 탐색의 회수율이 다섯 중 넷에서 0.95를 넘고, 예산을 500으로 낮춰도 0.80~0.94다. 분자가 천장에 닿으면 배수는 값이 아니라 하한이다.

천장이 없는 양으로 바꿔 읽는다. 회수율이 절반에 이르는 예산은 단조 증가하는 곡선 위의 한 점이고, '상위 1%의 절반을 찾으려면 몇 번 재야 하는가'로 바로 읽힌다. exp72 의 시드별 곡선에서 로그 예산 선형 보간으로 구했고, 시험한 범위(200~1,000) 안에서 절반에 이르지 못한 시드는 외삽하지 않고 따로 셌다.

| 팔 | 종류 | 전략 | 절반 도달 예산 (중앙) | 사분위 | 도달 시드 |
|---|---|---|---|---|---|
| matched:1NCA_N | matched lattice | active | 290 | 282 ~ 310 | 20/20 |
| matched:2Q8A_A | matched lattice | active | 292 | 280 ~ 301 | 20/20 |
| matched:1ZTX_E | matched lattice | active | 294 | 288 ~ 312 | 20/20 |
| matched:1OSP_O | matched lattice | active | 318 | 306 ~ 336 | 20/20 |
| matched:1ADQ_A | matched lattice | active | 366 | 354 ~ 391 | 20/20 |
| matched:3VRL_C | matched lattice | active | 370 | 314 ~ 426 | 20/20 |
| matched:2DD8_S | matched lattice | active | 500 | 463 ~ 645 | 18/20 |
| real:HER2 | measured | active | 799 | 735 ~ 902 | 18/20 |
| real:TIGIT | measured | active | 952 | 887 ~ 956 | 8/20 |
| real:CoV2-RBD | measured | active | &gt;1,000 | — | 0/20 |
| real:PD-1 | measured | active | &gt;1,000 | — | 0/20 |
| matched:1ADQ_A | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:1NCA_N | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:1OSP_O | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:1ZTX_E | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:2DD8_S | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:2Q8A_A | matched lattice | random | &gt;1,000 | — | 0/20 |
| matched:3VRL_C | matched lattice | random | &gt;1,000 | — | 0/20 |
| real:CoV2-RBD | measured | random | &gt;1,000 | — | 0/20 |
| real:HER2 | measured | random | &gt;1,000 | — | 0/20 |
| real:PD-1 | measured | random | &gt;1,000 | — | 0/20 |
| real:TIGIT | measured | random | &gt;1,000 | — | 0/20 |

유도 탐색으로 절반에 이르는 예산은 정합 격자가 290~500회, 실측이 799~952회다.

시험한 범위 안에서 절반에 이르지 못한 팔이 있다. 그 칸은 '>1,000'으로 적었고, 얼마나 더 필요한지는 이 자료로 말할 수 없다.

이 지표도 완전하지 않다. 예산을 셋만 재었으므로 보간의 해상도가 거기까지이고, 정합 격자의 값은 가장 낮은 예산(200)에 가까워 아래쪽 끝에 눌린다. 더 낮은 예산은 탐색 루틴의 초기 표본 하한(200) 때문에 잴 수 없다. 자릿수로 읽어야 한다.

