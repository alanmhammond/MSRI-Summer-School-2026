#!/usr/bin/env python3
import math
import numpy as np
from collections import deque

# ---------- Min-cost max-flow (successive SPFA; OK for N~20) ----------

class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        # forward
        self.adj[u].append([v, cap, cost, None])
        # backward
        self.adj[v].append([u, 0, -cost, None])
        self.adj[u][-1][3] = self.adj[v][-1]
        self.adj[v][-1][3] = self.adj[u][-1]

    def min_cost_flow(self, s, t, f):
        n = self.n
        flow = 0
        cost = 0
        INF = 10**18

        while flow < f:
            dist = [INF]*n
            inq = [False]*n
            prev = [None]*n  # (u, edge_ref)
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
                break  # cannot send more

            # augment by 1 (all caps are small)
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

# ---------- RSK via disjoint increasing paths on (p,q) ----------

def pq_coords(x, t):
    # lightcone coords
    return (t + x, t - x)

def build_poset(points_pq):
    # points_pq: list of (p,q)
    # edge i->j if pi<pj and qi<qj (strictly increasing)
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
    """
    Compute M_j(endpoint): max total points collected by j vertex-disjoint increasing paths
    from source (0,0) to endpoint (P,Q), visiting points <= endpoint.
    """
    P, Q = endpoint_pq
    eligible = [(p,q) for (p,q) in points_pq if p <= P and q <= Q]
    n = len(eligible)
    if n == 0:
        return 0

    edges = build_poset(eligible)

    # Build flow network:
    # source S, sink T
    # each point i split into i_in and i_out with cap 1, cost -1 (selecting point)
    # edges i_out -> j_in if i precedes j
    # S -> i_in if (0,0) precedes i  (always, since p,q>=0 in our region)
    # i_out -> T if i precedes endpoint
    #
    # Also allow "empty" paths: S->T edges to send flow that selects no points.
    #
    # Nodes: 0=S, 1=T, then for each i: in=2+2i, out=2+2i+1
    S, T = 0, 1
    Nnodes = 2 + 2*n
    mcmf = MinCostMaxFlow(Nnodes)

    # S->T to allow unused flow
    mcmf.add_edge(S, T, j, 0)

    for i, (p,q) in enumerate(eligible):
        IN = 2 + 2*i
        OUT = IN + 1
        mcmf.add_edge(IN, OUT, 1, -1)      # take point i yields reward 1
        mcmf.add_edge(S, IN, 1, 0)         # start at point i (or later via other points)
        # transitions
        for k in edges[i]:
            INk = 2 + 2*k
            mcmf.add_edge(OUT, INk, 1, 0)
        # can end at i if i precedes endpoint (always true by eligibility strictness not needed)
        mcmf.add_edge(OUT, T, 1, 0)

    flow, cost = mcmf.min_cost_flow(S, T, j)
    # cost is negative of collected points (since each selected point contributes -1)
    return -cost

def lines_from_M(Mvals):
    """
    Given M[1..J] (1-indexed list length J), return line heights L_0, L_-1, ..., L_-(J-1)
    using: L0=M1; L_{-(j-1)} = Mj - M_{j-1} - (j-1).
    """
    J = len(Mvals)
    L = [0]*J
    L[0] = Mvals[0]
    for j in range(2, J+1):
        L[j-1] = Mvals[j-1] - Mvals[j-2] - (j-1)
    return L

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # small demo: 20 points in triangle x>=0,t>=0,x+t<=T
    Ttri = 12.0
    pts = []
    while len(pts) < 20:
        x = rng.uniform(0, Ttri)
        t = rng.uniform(0, Ttri)
        if x + t <= Ttri:
            pts.append((x,t))

    points_pq = [pq_coords(x,t) for (x,t) in pts]

    # choose an endpoint: x = T/2, t = T/2 (on diagonal), in (p,q): (T,0)
    endpoint = pq_coords(Ttri/2, Ttri/2)  # (p,q)=(T,0)
    J = 6
    Mvals = [M_j_to_endpoint(points_pq, endpoint, j) for j in range(1, J+1)]
    Lvals = lines_from_M(Mvals)

    print("Endpoint (p,q):", endpoint)
    print("M_j:", Mvals)
    print("Lines L_0, L_-1, ...:", Lvals)