"""Absolut!(Ymir)의 격자 구조 문자열을 좌표로 되돌린다.

에피토프 대조를 하려면 "이 항체가 항원의 어디에 붙었는가"를 잔기 단위로 알아야
한다. 그 정보는 이미 출력에 들어 있다.

    항원   구조 파일 첫 줄   Ligand=<탭>133152<탭>SULSUURRUULL...
    항체   채점 결과 끝 열   124837-UDDLLDSLSD

둘 다 "시작 위치 + 이동열"이고, 이동 하나가 잔기 하나다(1FBI_X는 항원이 129
잔기, 이동열이 128자 -- 정확히 n-1). 그러니 둘을 같은 격자에 펼치고 인접한
자리를 세면 접촉면이 나온다.

규약은 추측하지 않는다. Absolut 소스가 tools/Absolut-main에 있고, 아래는 그
구현을 그대로 옮긴 것이다.

    lattice.h       XWidth=YWidth=ZWidth=64
    lattice.cpp     z = id/4096 · y = (id-4096z)/64 · x = 나머지
    compact.cpp     moveVector  S=(+1,0,0) B=(-1,0,0) L=(0,+1,0) R=(0,-1,0)
                                U=(0,0,+1) D=(0,0,-1)
                    첫 글자는 절대 방향, 그 뒤는 상대 이동
                    initialYaxis  S->L  B->R  R->S  L->B  U->L  D->L
                    movementMap   다음 방향 = mx*X + my*Y + mz*Z (Z=X×Y)
                                  좌/우로 꺾으면 newY = Z × newX
                                  위/아래로 꺾으면 newY = Y 유지

맞게 옮겼는지는 논증하지 않고 검사한다. 단백질 접힘은 자기 회피이므로, 규약을
하나라도 틀리면 조밀한 접힘에서 몇 걸음 안에 같은 자리를 두 번 밟는다.
`decode`는 그때 예외를 낸다.
"""
W = 64
SIZE = W * W * W

# compact.cpp moveVector()
VEC = {"S": (1, 0, 0), "B": (-1, 0, 0), "L": (0, 1, 0), "R": (0, -1, 0),
       "U": (0, 0, 1), "D": (0, 0, -1)}
# compact.cpp initialYaxis()
INIT_Y = {"R": "S", "L": "B", "U": "L", "D": "L", "S": "L", "B": "R"}


def pos_from_id(i):
    z, r = divmod(int(i), W * W)
    y, x = divmod(r, W)
    return x, y, z


def id_from_pos(p):
    x, y, z = p
    if not (0 <= x < W and 0 <= y < W and 0 <= z < W):
        raise ValueError(f"격자 밖: {p}")
    return x + W * y + W * W * z


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _name(v):
    for k, u in VEC.items():
        if u == v:
            return k
    raise ValueError(f"단위 방향이 아님: {v}")


def _next(cur_dir, cur_y, rel):
    """compact.cpp movementMap()의 한 칸을 그대로 계산한다."""
    X, Y = VEC[cur_dir], VEC[cur_y]
    Z = _cross(X, Y)
    m = VEC[rel]
    new_x = tuple(m[0] * X[i] + m[1] * Y[i] + m[2] * Z[i] for i in range(3))
    if rel in ("R", "L"):
        new_y = _cross(Z, new_x)
    else:                       # S, U, D, B -- Y축은 그대로
        new_y = Y
    return _name(new_x), _name(new_y)


def decode(start_id, moves, init_y=None):
    """시작 위치와 이동열을 잔기별 격자 좌표 목록으로 편다.

    반환값의 길이는 len(moves) + 1이다. 자기 회피를 어기면 ValueError를 낸다 --
    규약을 잘못 옮겼다는 뜻이므로 조용히 넘기면 안 된다.
    """
    moves = "".join(ch for ch in str(moves).strip() if ch in VEC)
    p = pos_from_id(start_id)
    out = [id_from_pos(p)]
    seen = {out[0]}
    if not moves:
        return out
    cur_dir = moves[0]
    cur_y = init_y or INIT_Y[cur_dir]
    for i, ch in enumerate(moves):
        if i == 0:
            d = VEC[cur_dir]
        else:
            cur_dir, cur_y = _next(cur_dir, cur_y, ch)
            d = VEC[cur_dir]
        p = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
        q = id_from_pos(p)
        if q in seen:
            raise ValueError(f"자기 회피 위반 — 잔기 {i + 1}에서 같은 자리를 "
                             f"두 번 밟았다. 규약 해석이 틀렸다.")
        seen.add(q)
        out.append(q)
    return out


def neighbours(i):
    """격자에서 면을 맞댄 여섯 자리."""
    x, y, z = pos_from_id(i)
    out = []
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                       (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        a, b, c = x + dx, y + dy, z + dz
        if 0 <= a < W and 0 <= b < W and 0 <= c < W:
            out.append(a + W * b + W * W * c)
    return out


def contacts(ag_path, ab_path):
    """항체 잔기와 면을 맞댄 항원 잔기의 번호를 낸다(0부터).

    돌려주는 것은 {항원 잔기 번호: 접촉 수}다. 번호는 항원 서열 순서와 같다 --
    Ligand= 줄이 잔기 순서대로 걷기 때문이다.
    """
    where = {pos: k for k, pos in enumerate(ag_path)}
    hit = {}
    for pos in ab_path:
        for nb in neighbours(pos):
            k = where.get(nb)
            if k is not None:
                hit[k] = hit.get(k, 0) + 1
    return hit
