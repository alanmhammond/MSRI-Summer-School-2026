#!/usr/bin/env python3
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

SQRT2 = math.sqrt(2.0)

# ---------------- Min-cost max-flow ----------------

class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        fwd = [v, cap, cost, None]
        rev = [u, 0, -cost, fwd]
        fwd[3] = rev
        self.adj[u].append(fwd)
        self.adj[v].append(rev)

    def min_cost_flow(self, s, t, f):
        INF = 10**18
        flow = 0
        cost = 0

        while flow < f:
            dist = [INF] * self.n
            inq = [False] * self.n
            prev = [None] * self.n  # (u, edge)

            dist[s] = 0
            q = deque([s])
            inq[s] = True

            while q:
                u = q.popleft()
                inq[u] = False
                for e in self.adj[u]:
                    v, cap, w, rev = e
                    if cap > 0 and dist[u] + w < dist[v]:
                        dist[v] = dist[u] + w
                        prev[v] = (u, e)
                        if not inq[v]:
                            q.append(v)
                            inq[v] = True

            if prev[t] is None:
                break

            add = f - flow
            v = t
            while v != s:
                u, e = prev[v]
                add = min(add, e[1])
                v = u

            v = t
            while v != s:
                u, e = prev[v]
                e[1] -= add
                e[3][1] += add
                cost += add * e[2]
                v = u

            flow += add

        return flow, cost

# ---------------- Geometry ----------------

def rotate_45_ccw(x: np.ndarray, t: np.ndarray):
    u = (x - t) / SQRT2
    v = (x + t) / SQRT2
    return u, v

def pq_coords(x: float, t: float):
    # lightcone coords: increasing upgoing order becomes componentwise
    return (t + x, t - x)

# ---------------- Sampling: Poisson in cone ----------------

def sample_poisson_cone_xt(T: float, intensity: float, rng: np.random.Generator):
    """
    Sample Poisson PPP of rate 'intensity' in C_T = {(x,t): 0<=t<=T, |x|<=t}.
    Area(C_T) = int_0^T 2t dt = T^2, so N ~ Poisson(intensity * T^2).
    Conditional on N, sample t with density proportional to 2t, i.e. t = T*sqrt(U),
    then x | t uniform on [-t, t].
    """
    mean = intensity * (T**2)
    N = rng.poisson(mean)
    U = rng.random(N)
    t = T * np.sqrt(U)
    x = (2 * rng.random(N) - 1) * t
    return np.column_stack([x, t])

def sample_extra_point_cone_xt(T: float, rng: np.random.Generator):
    U = rng.random()
    t = T * math.sqrt(U)
    x = (2 * rng.random() - 1) * t
    return x, t

# ---------------- RSK: M_j via disjoint increasing paths ----------------

def build_poset(points_pq):
    n = len(points_pq)
    edges = [[] for _ in range(n)]
    for i in range(n):
        pi, qi = points_pq[i]
        for j in range(n):
            if i == j:
                continue
            pj, qj = points_pq[j]
            if pi < pj and qi < qj:
                edges[i].append(j)
    return edges

def M_j_to_endpoint(points_pq, endpoint_pq, j):
    P, Q = endpoint_pq
    eligible = [(p, q) for (p, q) in points_pq if p <= P and q <= Q]
    n = len(eligible)
    if n == 0:
        return 0

    edges = build_poset(eligible)

    S, T = 0, 1
    Nnodes = 2 + 2 * n
    mcmf = MinCostMaxFlow(Nnodes)

    # allow unused flow
    mcmf.add_edge(S, T, j, 0)

    for i, (p, q) in enumerate(eligible):
        IN = 2 + 2 * i
        OUT = IN + 1
        mcmf.add_edge(IN, OUT, 1, -1)   # selecting point contributes +1 length
        mcmf.add_edge(S, IN, 1, 0)
        for k in edges[i]:
            INk = 2 + 2 * k
            mcmf.add_edge(OUT, INk, 1, 0)
        mcmf.add_edge(OUT, T, 1, 0)

    flow, cost = mcmf.min_cost_flow(S, T, j)
    return -cost

def lines_from_M(Mvals):
    J = len(Mvals)
    L = np.zeros(J, dtype=int)
    L[0] = Mvals[0]
    for j in range(2, J + 1):
        L[j - 1] = Mvals[j - 1] - Mvals[j - 2] - (j - 1)
    return L

def compute_lines_profile_fixed_time(points_xt, T, nlines, x_grid):
    points_pq = [pq_coords(float(x), float(t)) for x, t in points_xt]
    Lprof = np.zeros((nlines, len(x_grid)), dtype=int)
    for jx, x in enumerate(x_grid):
        endpoint = (T + float(x), T - float(x))  # time slice t=T
        Mvals = [M_j_to_endpoint(points_pq, endpoint, j) for j in range(1, nlines + 1)]
        Lprof[:, jx] = lines_from_M(Mvals)
    return Lprof

# ---------------- Plot ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=12.0, help="final time")
    ap.add_argument("--intensity", type=float, default=1.0, help="Poisson intensity in cone")
    ap.add_argument("--nlines", type=int, default=6, help="how many lines to compute/plot")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xgrid", type=int, default=260, help="number of x samples on [-T,T]")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    pts_xt = sample_poisson_cone_xt(args.T, args.intensity, rng)
    xe, te = sample_extra_point_cone_xt(args.T, rng)
    pts_xt_aug = np.vstack([pts_xt, np.array([[xe, te]])])

    # top panel rotated
    u, v = rotate_45_ccw(pts_xt[:, 0], pts_xt[:, 1])
    ue, ve = rotate_45_ccw(np.array([xe]), np.array([te]))
    ue, ve = float(ue[0]), float(ve[0])
    v_final = args.T / SQRT2

    # bottom panel: bridges at time t=T over x in [-T,T]
    x_grid = np.linspace(-args.T, args.T, args.xgrid)
    L0 = compute_lines_profile_fixed_time(pts_xt, args.T, args.nlines, x_grid)
    L1 = compute_lines_profile_fixed_time(pts_xt_aug, args.T, args.nlines, x_grid)
    D = L1 - L0

    # pick single depth per x for shading (should already be unique in the ideal)
    pos = D > 0
    has = pos.any(axis=0)
    k_of_x = np.full(len(x_grid), -1, dtype=int)
    if has.any():
        k_of_x[has] = np.argmax(pos[:, has], axis=0)

    y_low = np.full(len(x_grid), np.nan, dtype=float)
    y_high = np.full(len(x_grid), np.nan, dtype=float)
    cols = np.where(k_of_x >= 0)[0]
    if cols.size:
        kk = k_of_x[cols]
        y0 = L0[kk, cols]
        y1 = L1[kk, cols]
        y_low[cols] = np.minimum(y0, y1)
        y_high[cols] = np.maximum(y0, y1)

    # --- plot ---
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 2.2], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    ax0.scatter(u, v, s=18, alpha=0.65, label=f"cloud (N={len(pts_xt)})")
    ax0.scatter([ue], [ve], s=120, marker="*", label="extra point")
    ax0.axhline(v_final, linewidth=1.6, alpha=0.8, label="t=T slice (v=T/√2)")
    ax0.set_xlim(-v_final * 1.05, v_final * 1.05)
    ax0.set_ylim(0.0, v_final * 1.02)
    ax0.set_xlabel("u = (x-t)/√2")
    ax0.set_ylabel("v = (x+t)/√2")
    ax0.set_title(f"Cone PPP, intensity={args.intensity}, T={args.T}, seed={args.seed}")
    ax0.legend(loc="upper right")

    for k in range(args.nlines):
        ax1.plot(x_grid, L0[k], linewidth=1.0, alpha=0.9)
        ax1.plot(x_grid, L1[k], linewidth=1.0, alpha=0.9)

    ax1.fill_between(x_grid, y_low, y_high, where=~np.isnan(y_low), alpha=0.35)
    ax1.set_xlim(-args.T, args.T)
    ax1.set_xlabel("x (space, fixed time t=T)")
    ax1.set_ylabel("line height")
    ax1.set_title("RSK multi-line PNG at time t=T (cone model); grey = water (difference)")
    ax1.grid(True, alpha=0.15)

    plt.show()

if __name__ == "__main__":
    main()