"""Decode Absolut! lattice structures into 3D coordinates.

Absolut! stores a chain as a start position on a 64^3 lattice plus a string of
moves. The first move is absolute; every later move is relative to an observer
frame carried along the chain, the way a pilot's turns are relative to the
aircraft. The rules below are transcribed from Ymir's compact.cpp
(moveVector, initialYaxis and movementMap) so the geometry here is the same one
that produced the binding energies.

Nothing is inferred: if a decoded chain is not self-avoiding, or a decoded
CDRH3 does not touch the antigen it was scored against, the transcription is
wrong and check_decoding() says so.
"""
import numpy as np

W = 64                                   # XWidth = YWidth = ZWidth in lattice.h

# moveVector() in compact.cpp
VEC = {"S": (1, 0, 0), "B": (-1, 0, 0), "R": (0, -1, 0), "L": (0, 1, 0),
       "U": (0, 0, 1), "D": (0, 0, -1)}
# initialYaxis() in compact.cpp
INIT_Y = {"R": "S", "L": "B", "U": "L", "D": "L", "S": "L", "B": "R"}
NAME = {v: k for k, v in VEC.items()}


def pos_from_id(i):
    """lattice::positionFromID."""
    z = i // (W * W)
    y = (i - W * W * z) // W
    x = i - W * W * z - W * y
    return np.array([x, y, z], int)


def id_from_pos(p):
    return int(p[0] + W * p[1] + W * W * p[2])


def _cross(a, b):
    return np.cross(np.array(a), np.array(b))


def decode(start_id, moves):
    """Return (n+1, 3) integer coordinates for a chain of n moves."""
    pos = pos_from_id(int(start_id))
    out = [pos.copy()]

    first = moves[0]
    x_ax = np.array(VEC[first])                 # observer x = current heading
    y_ax = np.array(VEC[INIT_Y[first]])         # observer y
    pos = pos + x_ax
    out.append(pos.copy())

    for m in moves[1:]:
        z_ax = _cross(x_ax, y_ax)
        a, b, c = VEC[m]                        # move expressed in observer frame
        new_x = a * x_ax + b * y_ax + c * z_ax
        if m in ("R", "L"):                     # turning in the walking plane
            new_y = _cross(z_ax, new_x)
        else:                                   # S, U, D keep the y axis
            new_y = y_ax
        x_ax, y_ax = new_x, new_y
        pos = pos + x_ax
        out.append(pos.copy())
    return np.array(out)


def self_avoiding(c):
    return len({tuple(p) for p in c}) == len(c)


def contacts(a, b):
    """Pairs (i, j) of lattice-adjacent residues between two chains."""
    bset = {tuple(p): j for j, p in enumerate(b)}
    out = []
    for i, p in enumerate(a):
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            j = bset.get((p[0] + d[0], p[1] + d[1], p[2] + d[2]))
            if j is not None:
                out.append((i, j))
    return out


def check_decoding(ag_coords, ag_seq, cdr_list, hotspots=None, verbose=True):
    """Validate the transcription against properties the format guarantees."""
    ok = True

    def say(label, good, detail=""):
        nonlocal ok
        ok = ok and good
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {label}{'  ' + detail if detail else ''}")

    say("antigen length matches its sequence",
        len(ag_coords) == len(ag_seq),
        f"{len(ag_coords)} residues vs {len(ag_seq)} letters")
    say("antigen chain is self-avoiding", self_avoiding(ag_coords))
    say("antigen bonds are unit steps",
        bool(np.all(np.abs(np.diff(ag_coords, axis=0)).sum(1) == 1)))

    sa = sum(self_avoiding(c) for c in cdr_list)
    say("every CDRH3 pose is self-avoiding", sa == len(cdr_list),
        f"{sa}/{len(cdr_list)}")

    touching = [len(contacts(c, ag_coords)) for c in cdr_list]
    n_touch = sum(t > 0 for t in touching)
    say("every CDRH3 pose touches the antigen", n_touch == len(cdr_list),
        f"{n_touch}/{len(cdr_list)}, median {int(np.median(touching))} contacts")

    if hotspots:
        hs = set(hotspots)
        hit = [any(j in hs for _, j in contacts(c, ag_coords)) for c in cdr_list]
        frac = sum(hit) / max(len(hit), 1)
        say("poses concentrate on annotated hotspots", frac > 0.5,
            f"{frac:.0%} of poses touch a hotspot residue")
    return ok
