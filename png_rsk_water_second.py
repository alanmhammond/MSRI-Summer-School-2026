#!/usr/bin/env python3
"""
png_multiline_water.py

Visual-first simulation of multi-line PNG / RSK step dynamics driven by a Poisson cloud:

Upstairs: Poisson points in a triangular region, rotated 45° CCW:
    u = (x - t)/sqrt(2),   v = (x + t)/sqrt(2)
so that the boundary x+t = T is horizontal (v = const) and v increases upward.

Downstairs: multi-line PNG dynamics in (u,v)-coordinates:
- Each point (u0,v0) nucleates on line L0 at time v0:
    create an upstep and downstep at u0.
- Between events:
    upsteps drift left with speed 1, downsteps drift right with speed 1.
- When a right-moving downstep meets a left-moving upstep on the same line
  (an adjacent D then U in the step ordering), they annihilate and trigger a
  nucleation at the collision location/time on the next line below.

We run this for the original cloud and the augmented cloud (extra uniformly random point),
then plot:
- both line ensembles downstairs
- water (aug - orig) shaded in grey

Note: This is a faithful implementation of the step/collision/nucleation description,
but parameterized by v (the rotated “time-like” coordinate). This matches the tilted
picture you requested: anti-diagonal horizontal; “time” vertical.
"""

import argparse
import bisect
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

SQRT2 = math.sqrt(2.0)

# ----------------------------
# Sampling and rotation
# ----------------------------

def sample_uniform_triangle_xt(T: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample n points uniformly from triangle {x>=0, t>=0, x+t<=T}.
    """
    pts = []
    while len(pts) < n:
        x = rng.uniform(0.0, T)
        t = rng.uniform(0.0, T)
        if x + t <= T:
            pts.append((x, t))
    return np.array(pts, dtype=float)

def rotate_45_ccw(x: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    45° CCW rotation:
        u = (x - t)/sqrt(2)
        v = (x + t)/sqrt(2)
    """
    u = (x - t) / SQRT2
    v = (x + t) / SQRT2
    return u, v

def sample_extra_point_xt(T: float, rng: np.random.Generator) -> Tuple[float, float]:
    while True:
        x = rng.uniform(0.0, T)
        t = rng.uniform(0.0, T)
        if x + t <= T:
            return x, t

# ----------------------------
# Multi-line PNG dynamics in (u,v)
# ----------------------------

@dataclass
class LineState:
    # sorted positions of steps at current time
    ups: List[float]    # +1 jumps
    downs: List[float]  # -1 jumps

    def advance(self, dv: float) -> None:
        """Move steps forward in time by dv: upsteps left, downsteps right."""
        if dv == 0.0:
            return
        self.ups = [u - dv for u in self.ups]
        self.downs = [d + dv for d in self.downs]

    def add_nucleation(self, u0: float) -> None:
        """Add an upstep and a downstep at the same location u0."""
        bisect.insort(self.ups, u0)
        bisect.insort(self.downs, u0)

    def next_collision(self) -> Optional[Tuple[float, float, float]]:
        """
        Find earliest collision among adjacent (D, U) pairs in the merged step list.
        Returns (dt, d_pos, u_pos) where dt is time-to-collision from now,
        and d_pos/u_pos are their current positions (at current time).
        """
        # Merge steps into one ordered list with labels
        steps = []
        for u in self.ups:
            steps.append((u, "U"))
        for d in self.downs:
            steps.append((d, "D"))
        if len(steps) < 2:
            return None
        steps.sort(key=lambda x: x[0])

        best = None
        for (p1, s1), (p2, s2) in zip(steps[:-1], steps[1:]):
            if s1 == "D" and s2 == "U":
                gap = p2 - p1
                # They approach at relative speed 2, collide in gap/2
                dt = gap / 2.0
                if dt < 0:
                    # Shouldn't happen with sorted order, but guard anyway
                    continue
                if best is None or dt < best[0]:
                    best = (dt, p1, p2)
        return best

    def apply_collision(self, d_pos: float, u_pos: float) -> float:
        """
        Remove the downstep at d_pos and upstep at u_pos (they are at collision moment),
        return collision location (midpoint).
        """
        # Because of floating arithmetic, remove by nearest match.
        # With event-driven exact updates, these should be present.
        def remove_near(arr: List[float], target: float) -> None:
            # find insertion point then remove closest
            i = bisect.bisect_left(arr, target)
            candidates = []
            if 0 <= i < len(arr):
                candidates.append(i)
            if i-1 >= 0:
                candidates.append(i-1)
            if not candidates:
                raise RuntimeError("No candidates to remove.")
            j = min(candidates, key=lambda k: abs(arr[k] - target))
            arr.pop(j)

        remove_near(self.downs, d_pos)
        remove_near(self.ups, u_pos)
        return 0.5 * (d_pos + u_pos)

def simulate_multiline(
    points_uv: np.ndarray,
    v_final: float,
    n_lines: int
) -> List[LineState]:
    """
    Simulate multi-line PNG dynamics up to time v_final.
    points_uv: array shape (N,2) sorted or unsorted; each point nucleates on line 0 at its v.
    Returns list of LineState for lines 0..n_lines-1 at time v_final.
    """
    # Sort points by time v
    pts = points_uv[np.argsort(points_uv[:, 1])]
    idx = 0
    N = len(pts)

    lines = [LineState([], []) for _ in range(n_lines)]
    t = 0.0

    while True:
        # Next nucleation time on line 0
        next_nuc_v = pts[idx, 1] if idx < N else float("inf")

        # Next collision among all lines
        best_col = None  # (event_time, line_k, d_pos, u_pos)
        for k in range(n_lines):
            col = lines[k].next_collision()
            if col is None:
                continue
            dt, d_pos, u_pos = col
            ev_t = t + dt
            if ev_t <= v_final + 1e-12:
                if best_col is None or ev_t < best_col[0]:
                    best_col = (ev_t, k, d_pos, u_pos)

        # Decide next event time
        next_event_t = min(next_nuc_v, best_col[0] if best_col else float("inf"), v_final)

        if next_event_t == float("inf"):
            break

        # Advance all lines to next_event_t
        dv = next_event_t - t
        if dv < -1e-12:
            raise RuntimeError("Time went backwards.")
        for k in range(n_lines):
            lines[k].advance(dv)
        t = next_event_t

        # Stop if reached final
        if abs(t - v_final) < 1e-12 or t >= v_final:
            break

        # Process all nucleations at this exact time (could be multiple)
        while idx < N and abs(pts[idx, 1] - t) < 1e-12:
            u0 = float(pts[idx, 0])
            lines[0].add_nucleation(u0)
            idx += 1

        # Process collision if it occurs at this time (might be multiple; loop)
        # Recompute collisions at this same time repeatedly until none at t
        while True:
            triggered = False
            for k in range(n_lines):
                col = lines[k].next_collision()
                if col is None:
                    continue
                dt, d_pos, u_pos = col
                if dt < 1e-12:
                    # collision now
                    c_u = lines[k].apply_collision(d_pos, u_pos)
                    if k + 1 < n_lines:
                        lines[k+1].add_nucleation(c_u)
                    triggered = True
            if not triggered:
                break

    # Advance to v_final if not already there
    if t < v_final:
        dv = v_final - t
        for k in range(n_lines):
            lines[k].advance(dv)

    return lines

# ----------------------------
# Rendering lines and water
# ----------------------------

def eval_line_height(line: LineState, base: int, ugrid: np.ndarray) -> np.ndarray:
    """
    Evaluate piecewise-constant height on grid.
    height = base + (#ups<=u) - (#downs<=u)
    """
    ups = np.array(line.ups, dtype=float)
    downs = np.array(line.downs, dtype=float)
    # counts via searchsorted
    cu = np.searchsorted(ups, ugrid, side="right") if ups.size else np.zeros_like(ugrid, dtype=int)
    cd = np.searchsorted(downs, ugrid, side="right") if downs.size else np.zeros_like(ugrid, dtype=int)
    return base + (cu - cd)

def plot_snapshot(
    pts_uv: np.ndarray,
    extra_uv: Tuple[float, float],
    lines_orig: List[LineState],
    lines_aug: List[LineState],
    v_final: float,
    n_lines: int,
    u_margin: float = 1.0,
    grid_m: int = 1200,
    water_alpha: float = 0.35,
):
    """
    Two-panel plot:
    - Top: point cloud (rotated), extra point marked, anti-diagonal shown.
    - Bottom: line ensembles (orig + aug), water shaded.
    """
    u_pts, v_pts = pts_uv[:, 0], pts_uv[:, 1]
    u_extra, v_extra = extra_uv

    umin = min(u_pts.min(), u_extra) - u_margin
    umax = max(u_pts.max(), u_extra) + u_margin
    ugrid = np.linspace(umin, umax, grid_m)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 2.2], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # --- Top: Poisson picture ---
    ax0.scatter(u_pts, v_pts, s=18, alpha=0.65, label="cloud")
    ax0.scatter([u_extra], [v_extra], s=120, marker="*", label="extra point")
    ax0.axhline(v_final, linewidth=1.6, alpha=0.8, label="anti-diagonal (v = const)")
    ax0.set_xlim(umin, umax)
    ax0.set_ylim(0.0, v_final * 1.02)
    ax0.set_xlabel("u (horizontal)")
    ax0.set_ylabel("v (vertical time)")
    ax0.set_title(f"Rotated cloud (N≈{len(pts_uv)}), with extra marked point")
    ax0.legend(loc="upper right")

    # --- Bottom: Lines + water ---
    # Draw each line as a step profile; also shade water between orig and aug.
    for k in range(n_lines):
        base = -k  # L_{-k} baseline
        hO = eval_line_height(lines_orig[k], base, ugrid)
        hA = eval_line_height(lines_aug[k], base, ugrid)

        # plot the two ensembles (thin)
        ax1.step(ugrid, hO, where="post", linewidth=1.0, alpha=0.9)
        ax1.step(ugrid, hA, where="post", linewidth=1.0, alpha=0.9)

        # shade water where augmented above original
        above = hA >= hO
        # Fill in grey (water)
        ax1.fill_between(
            ugrid, hO, hA,
            where=above,
            step="post",
            alpha=water_alpha,
        )

    ax1.set_xlim(umin, umax)
    ax1.set_xlabel("u (matches top panel)")
    ax1.set_ylabel("line height (stacked baselines -k)")
    ax1.set_title(f"Multi-line PNG at time v = {v_final:.3f}: original vs augmented; grey = water (aug - orig)")
    ax1.grid(True, alpha=0.15)

    plt.show()

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=12.0, help="Triangle parameter in (x,t): x+t<=T")
    ap.add_argument("--npoints", type=int, default=20, help="Number of Poisson points (fixed, for small demos)")
    ap.add_argument("--nlines", type=int, default=6, help="How many lines to simulate/plot (L0..L_{-(nlines-1)})")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--umargin", type=float, default=1.0, help="Horizontal margin for plotting")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Sample small cloud in triangle x>=0,t>=0,x+t<=T
    pts_xt = sample_uniform_triangle_xt(args.T, args.npoints, rng)
    x, t = pts_xt[:, 0], pts_xt[:, 1]
    u, v = rotate_45_ccw(x, t)

    # Extra point uniform in same triangle
    xe, te = sample_extra_point_xt(args.T, rng)
    ue, ve = rotate_45_ccw(np.array([xe]), np.array([te]))
    ue, ve = float(ue[0]), float(ve[0])

    # Use v_final as the anti-diagonal height: x+t=T -> v = T/sqrt(2)
    v_final = args.T / SQRT2

    pts_uv = np.column_stack([u, v])
    pts_uv_aug = np.vstack([pts_uv, np.array([[ue, ve]], dtype=float)])

    # Simulate multi-line dynamics (orig and augmented) up to v_final
    lines_orig = simulate_multiline(pts_uv, v_final=v_final, n_lines=args.nlines)
    lines_aug = simulate_multiline(pts_uv_aug, v_final=v_final, n_lines=args.nlines)

    # Plot snapshot
    plot_snapshot(
        pts_uv=pts_uv,
        extra_uv=(ue, ve),
        lines_orig=lines_orig,
        lines_aug=lines_aug,
        v_final=v_final,
        n_lines=args.nlines,
        u_margin=args.umargin,
    )

if __name__ == "__main__":
    main()