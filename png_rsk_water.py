#!/usr/bin/env python3
"""
png_rsk_water.py

Poisson cloud in the triangle {x>=0, t>=0, x+t<=T}, rotated 45° CCW so that
the anti-diagonal x+t=T is horizontal. Add one extra marked point uniformly.

Compute a discrete last-passage percolation (LPP) / RSK-style height evolution
on a fine grid:
    G[i,j] = W[i,j] + max(G[i-1,j], G[i,j-1])

Compare original vs augmented (with extra point). The discrepancy is shaded
("water") as time advances. Can show a static frame or animate.

Usage examples:
  python png_rsk_water.py --T 60 --intensity 1.0 --grid 450 --seed 0 --mode animate
  python png_rsk_water.py --T 60 --intensity 1.0 --grid 450 --seed 0 --mode static --time_frac 0.7
  python png_rsk_water.py --T 60 --intensity 1.0 --grid 450 --seed 0 --mode animate --save out.mp4
"""

import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

SQRT2 = math.sqrt(2.0)

def sample_poisson_triangle(T: float, intensity: float, rng: np.random.Generator):
    """
    Triangle: x>=0, t>=0, x+t<=T. Area = T^2/2.
    N ~ Poisson(intensity * area), points uniform in triangle.

    Uniform sampling via rejection on [0,T]^2 is fine for moderate sizes.
    """
    area = 0.5 * T * T
    n = rng.poisson(intensity * area)

    # Rejection sampling
    pts = np.empty((n, 2), dtype=float)
    k = 0
    while k < n:
        # batch
        m = max(1024, n - k)
        x = rng.uniform(0.0, T, size=m)
        t = rng.uniform(0.0, T, size=m)
        ok = (x + t) <= T
        x = x[ok]
        t = t[ok]
        take = min(len(x), n - k)
        if take > 0:
            pts[k:k+take, 0] = x[:take]
            pts[k:k+take, 1] = t[:take]
            k += take
    return pts  # (x,t)

def rotate_45_ccw(x: np.ndarray, t: np.ndarray):
    """
    45° CCW rotation:
        u = (x - t)/sqrt(2)
        v = (x + t)/sqrt(2)
    Anti-diagonal x+t=T -> v = T/sqrt(2) (horizontal).
    """
    u = (x - t) / SQRT2
    v = (x + t) / SQRT2
    return u, v

def points_to_grid(u, v, T, gridN):
    """
    Map rotated coordinates (u,v) into a grid W[iv, iu].
    v in [0, Vmax], u in [-Vmax, Vmax], where Vmax=T/sqrt(2).

    We shift u by +Vmax to make it nonnegative.

    Returns:
      W: integer weights (counts) on grid
      meta: dict with coordinate mapping helpers
    """
    Vmax = T / SQRT2
    Umin, Umax = -Vmax, Vmax

    # We'll use a square-ish grid: v has gridN rows, u has 2*gridN cols
    Nv = gridN
    Nu = 2 * gridN

    dv = Vmax / (Nv - 1)
    du = (Umax - Umin) / (Nu - 1)

    iu = np.floor((u - Umin) / du).astype(int)
    iv = np.floor(v / dv).astype(int)

    # clip to be safe
    iu = np.clip(iu, 0, Nu - 1)
    iv = np.clip(iv, 0, Nv - 1)

    W = np.zeros((Nv, Nu), dtype=np.int16)
    for a, b in zip(iv, iu):
        W[a, b] += 1

    meta = dict(
        Vmax=Vmax, Umin=Umin, Umax=Umax,
        Nv=Nv, Nu=Nu, dv=dv, du=du
    )
    return W, meta

def lpp_dp(W: np.ndarray):
    """
    Compute last passage values G with:
      G[i,j] = W[i,j] + max(G[i-1,j], G[i,j-1])
    Boundary: missing indices treated as 0.
    """
    Nv, Nu = W.shape
    G = np.zeros((Nv, Nu), dtype=np.int32)

    # first cell
    G[0, 0] = W[0, 0]

    # first row
    for j in range(1, Nu):
        G[0, j] = W[0, j] + G[0, j-1]

    # first col
    for i in range(1, Nv):
        G[i, 0] = W[i, 0] + G[i-1, 0]

    # interior
    for i in range(1, Nv):
        # local var for speed
        Gi = G[i]
        Gim1 = G[i-1]
        Wi = W[i]
        for j in range(1, Nu):
            Gi[j] = Wi[j] + (Gim1[j] if Gim1[j] >= Gi[j-1] else Gi[j-1])

    return G

def grid_to_coords(meta):
    """
    Build coordinate arrays for plotting.
    """
    Umin, Umax = meta["Umin"], meta["Umax"]
    Vmax = meta["Vmax"]
    Nu, Nv = meta["Nu"], meta["Nv"]
    ugrid = np.linspace(Umin, Umax, Nu)
    vgrid = np.linspace(0.0, Vmax, Nv)
    return ugrid, vgrid

def make_static_plot(u_pts, v_pts, u_extra, v_extra,
                     G0, G1, meta, time_frac=0.7):
    """
    Static snapshot at a chosen time fraction (0..1).
    Shows:
      - point cloud (rotated)
      - step profiles at that time for original vs augmented
      - shaded discrepancy ("water")
    """
    ugrid, vgrid = grid_to_coords(meta)
    Nv = meta["Nv"]
    ti = int(np.clip(round(time_frac * (Nv - 1)), 0, Nv - 1))

    h0 = G0[ti, :]
    h1 = G1[ti, :]
    water = h1 - h0

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # Top: cloud + extra
    ax0.scatter(u_pts, v_pts, s=8, alpha=0.5, label="Poisson cloud")
    ax0.scatter([u_extra], [v_extra], s=60, marker="*", label="extra point")
    ax0.axhline(meta["Vmax"], linewidth=1.5, alpha=0.8, label="anti-diagonal (horizontal)")
    ax0.set_title("Rotated Poisson cloud (45° CCW): anti-diagonal is horizontal")
    ax0.set_xlabel("u")
    ax0.set_ylabel("v (time-like)")
    ax0.legend(loc="upper right")

    # Bottom: profiles + water
    ax1.step(ugrid, h0, where="post", linewidth=1.5, label="height (original)")
    ax1.step(ugrid, h1, where="post", linewidth=1.5, label="height (augmented)")
    # shade where discrepancy > 0
    ax1.fill_between(ugrid, h0, h1, step="post", alpha=0.3, label="water = augmented - original")

    ax1.set_title(f"Height profiles at v ≈ {vgrid[ti]:.3f}  (index {ti}/{Nv-1})")
    ax1.set_xlabel("u")
    ax1.set_ylabel("height")
    ax1.legend(loc="upper left")

    plt.show()

def make_animation(u_pts, v_pts, u_extra, v_extra,
                   G0, G1, meta, interval_ms=60, save_path=None):
    """
    Animation over time index i = 0..Nv-1.
    Top panel: points (static) + moving horizontal time line.
    Bottom: two step profiles + shaded water.
    """
    ugrid, vgrid = grid_to_coords(meta)
    Nv = meta["Nv"]

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])

    # Top: cloud
    ax0.scatter(u_pts, v_pts, s=8, alpha=0.5)
    ax0.scatter([u_extra], [v_extra], s=60, marker="*")
    ax0.axhline(meta["Vmax"], linewidth=1.5, alpha=0.8)
    time_line = ax0.axhline(vgrid[0], linewidth=2.0, alpha=0.8)
    ax0.set_title("Rotated cloud; moving time level (horizontal)")
    ax0.set_xlabel("u")
    ax0.set_ylabel("v (time-like)")

    # Bottom: initialize profiles
    h0 = G0[0, :]
    h1 = G1[0, :]
    line0, = ax1.step(ugrid, h0, where="post", linewidth=1.5, label="original")
    line1, = ax1.step(ugrid, h1, where="post", linewidth=1.5, label="augmented")
    water_poly = ax1.fill_between(ugrid, h0, h1, step="post", alpha=0.3)

    ax1.set_xlabel("u")
    ax1.set_ylabel("height")
    ax1.legend(loc="upper left")

    # set y-limits generously once
    ymax = int(max(G0.max(), G1.max()))
    ax1.set_ylim(-0.5, ymax + 1.0)

    def update(i):
        nonlocal water_poly
        time_line.set_ydata([vgrid[i], vgrid[i]])
        h0i = G0[i, :]
        h1i = G1[i, :]

        # Update step lines (matplotlib step returns a Line2D, just set ydata)
        line0.set_ydata(h0i)
        line1.set_ydata(h1i)

        # Rebuild the filled region
        water_poly.remove()
        water_poly = ax1.fill_between(ugrid, h0i, h1i, step="post", alpha=0.3)

        ax1.set_title(f"v ≈ {vgrid[i]:.3f}   frame {i+1}/{Nv}")
        return (time_line, line0, line1, water_poly)

    anim = FuncAnimation(fig, update, frames=Nv, interval=interval_ms, blit=False)

    if save_path is not None:
        # If you have ffmpeg installed, mp4 will work:
        #   brew install ffmpeg
        anim.save(save_path, dpi=150)
        print(f"Saved animation to {save_path}")

    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=60.0, help="Triangle parameter: x+t<=T")
    ap.add_argument("--intensity", type=float, default=1.0, help="Poisson intensity per unit area")
    ap.add_argument("--grid", type=int, default=450, help="Grid resolution for v; u uses 2*grid")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--mode", choices=["static", "animate"], default="animate")
    ap.add_argument("--time_frac", type=float, default=0.7, help="For static: time fraction in [0,1]")
    ap.add_argument("--interval_ms", type=int, default=60, help="Animation frame interval (ms)")
    ap.add_argument("--save", type=str, default=None, help="Optional path to save animation (e.g. out.mp4)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Sample base Poisson cloud
    pts = sample_poisson_triangle(args.T, args.intensity, rng)
    x, t = pts[:, 0], pts[:, 1]
    u_pts, v_pts = rotate_45_ccw(x, t)

    # Extra point uniform in the same triangle
    extra = sample_poisson_triangle(args.T, intensity=0.0, rng=rng)  # returns empty
    # we just sample 1 point by rejection directly
    while True:
        xe = rng.uniform(0.0, args.T)
        te = rng.uniform(0.0, args.T)
        if xe + te <= args.T:
            break
    u_extra, v_extra = rotate_45_ccw(np.array([xe]), np.array([te]))
    u_extra, v_extra = float(u_extra[0]), float(v_extra[0])

    # Discretize to grid weights
    W0, meta = points_to_grid(u_pts, v_pts, args.T, args.grid)
    # augmented weights
    W1 = W0.copy()
    # add extra to augmented grid
    Vmax = meta["Vmax"]
    Umin = meta["Umin"]
    dv = meta["dv"]
    du = meta["du"]
    Nu = meta["Nu"]
    Nv = meta["Nv"]
    iu = int(np.clip(math.floor((u_extra - Umin) / du), 0, Nu - 1))
    iv = int(np.clip(math.floor(v_extra / dv), 0, Nv - 1))
    W1[iv, iu] += 1

    # LPP / RSK heights
    G0 = lpp_dp(W0)
    G1 = lpp_dp(W1)

    if args.mode == "static":
        make_static_plot(u_pts, v_pts, u_extra, v_extra, G0, G1, meta, time_frac=args.time_frac)
    else:
        make_animation(u_pts, v_pts, u_extra, v_extra, G0, G1, meta,
                       interval_ms=args.interval_ms, save_path=args.save)

if __name__ == "__main__":
    main()