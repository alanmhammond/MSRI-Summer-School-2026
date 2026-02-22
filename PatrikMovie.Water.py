"""
PNG / multi-line PNG animation (pygame) with an "extra point coupling" and MULTI-LEVEL water.

What it does:
- Generates a base Poisson configuration (original).
- Optionally adds ONE extra point (augmented).
- Evolves both pictures in lockstep (same time, same rules).
- For each displayed level l=0..NLinee-1:
    original curve = blue
    augmented curve = red
    discrepancy (augmented above original) = grey "water"
  using an O(#jumps) method (merge breakpoints), not per-pixel scanning.

Controls:
  Space  : Start/Stop
  R      : Toggle time direction
  N      : New points (restart; refresh extra point if enabled)
  A      : Toggle extra point on/off (restart)
  1/2/3  : Geometry (Flat / Droplet / Droplet+Sources)
  Up/Down: Speed +/- (FPS)
  [ / ]  : NPunti -/+ (restarts)
  { / }  : NLinee -/+  (Shift+[ and Shift+])
  - / =  : DLinee -/+
  , / .  : LSource -/+ (restarts in sources geometry)
  K / L  : RSource -/+ (restarts in sources geometry)

Note:
- In droplet mode, the cone is symmetric: points appear on both sides of the centerline by design.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pygame


# -----------------------------
# Data structures (Java-like)
# -----------------------------

@dataclass
class Punto:
    x: int
    y: int
    next: Optional["Punto"] = None


class Linea:
    """
    dir:  1 = UP, -1 = DOWN, 0 = to cancel, 2 = unused sentinel
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.Njumps = 0
        self.NCancel = 0
        self.pos = [0] * capacity
        self.dir = [2] * capacity
        self.next: Optional["Linea"] = None
        self.prec: Optional["Linea"] = None

    def ensure_capacity_for(self, add: int) -> None:
        if self.Njumps + add <= self.capacity:
            return
        newcap = max(self.capacity * 2, self.Njumps + add)
        self.pos.extend([0] * (newcap - self.capacity))
        self.dir.extend([2] * (newcap - self.capacity))
        self.capacity = newcap


# -----------------------------
# Simulation core
# -----------------------------

class PNGSim:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.NPunti = 30
        self.Raggio = 3

        self.DLinee = 12
        self.NLinee = 10

        self.Speed = 30
        self.LSource = 10
        self.RSource = 5

        self.TypeOfPNG = 1      # 0 droplet, 1 flat, 2 droplet+sources

        self.freeze = True
        self.DirTempo = True

        self.Larghezza = 0
        self.Altezza = 0

        self.TotPunti = self.NPunti + self.LSource + self.RSource
        self.t = 0
        self.tOld = 0
        self.ComputedPoisson = False

        self.PuntoIniziale = Punto(0, 0)
        self._build_point_list()

        self.LineaIniziale: Optional[Linea] = None
        self.LineaFinale: Optional[Linea] = None
        self.InitLines()

    # ---- linked list helpers

    def _build_point_list(self) -> None:
        self.PuntoIniziale.next = None
        p = self.PuntoIniziale
        for _ in range(1, self.TotPunti):
            p.next = Punto(0, 0)
            p = p.next

    def set_points_from_list(self, pts: List[Tuple[int, int]]) -> None:
        self.TotPunti = len(pts)
        self._build_point_list()

        p = self.PuntoIniziale
        for (x, y) in pts:
            if p is None:
                break
            p.x, p.y = x, y
            p = p.next

        self.InitLines()

    def iter_points(self):
        p = self.PuntoIniziale
        while p is not None:
            yield p
            p = p.next

    def iter_lines(self):
        L = self.LineaIniziale
        while L is not None:
            yield L
            L = L.next

    def line_at_level(self, level: int) -> Optional[Linea]:
        L = self.LineaIniziale
        i = 0
        while L is not None and i < level:
            L = L.next
            i += 1
        return L

    # ---- init/reset

    def InitLines(self) -> None:
        cap = max(4, 2 * self.TotPunti)
        self.LineaIniziale = Linea(cap)
        self.LineaFinale = Linea(cap)

        for i in range(self.LineaIniziale.capacity):
            self.LineaIniziale.pos[i] = 0
            self.LineaIniziale.dir[i] = 2
        self.LineaIniziale.Njumps = 0
        self.LineaIniziale.NCancel = 0

        for i in range(self.LineaFinale.capacity):
            self.LineaFinale.pos[i] = 0
            self.LineaFinale.dir[i] = 2
        self.LineaFinale.Njumps = 0
        self.LineaFinale.NCancel = 0

        self.LineaIniziale.next = self.LineaFinale
        self.LineaIniziale.prec = None
        self.LineaFinale.prec = self.LineaIniziale
        self.LineaFinale.next = None

    def recompute_totpunti(self) -> None:
        if self.TypeOfPNG in (0, 1):
            self.TotPunti = self.NPunti
        else:
            self.TotPunti = self.NPunti + self.LSource + self.RSource
        self._build_point_list()
        self.InitLines()
        self.ComputedPoisson = False

    # ---- Poisson generation

    def sample_point_in_cone(self, width: int, height: int) -> Tuple[int, int]:
        cx = width // 2
        while True:
            x = 4 + int((width - 9) * random.random())
            y = 4 + int((height - 9) * random.random())
            if abs(x - cx) <= (height - 2 - y):
                y = ((x + y) % 2) + y
                if y > height - 5:
                    y = height - 5
                return x, y

    def ComputePoisson_points_list(self) -> List[Tuple[int, int]]:
        width = self.Larghezza
        height = self.Altezza
        if width <= 0 or height <= 0:
            return []

        pts: List[Tuple[int, int]] = []
        k = 0
        tot = self.TotPunti

        while len(pts) < tot:
            while True:
                nuovo = True

                if self.TypeOfPNG == 1:
                    x = 4 + int((width + 2 * height - 9) * random.random()) - height
                    y = 4 + int((height - 9) * random.random())

                if (self.TypeOfPNG == 0) or (self.TypeOfPNG == 2 and k >= self.LSource + self.RSource):
                    x = 4 + int((width - 9) * random.random())
                    y = 4 + int((height - 9) * random.random())
                    cx = width // 2
                    if abs(x - cx) > (height - 2 - y):
                        nuovo = False

                if (self.TypeOfPNG == 2) and (k < self.LSource + self.RSource):
                    y = 4 + int((height - 9) * random.random())
                    if k < self.LSource:
                        x = (width // 2) - height + y + 2
                    else:
                        x = (width // 2) + height - y - 2

                y = ((x + y) % 2) + y
                if y > height - 5:
                    y = height - 5

                if (x, y) in pts and width * height > 0:
                    nuovo = False

                if nuovo:
                    break

            pts.append((x, y))
            k += 1

        return pts

    # ---- dynamics

    def _add_nucleation_to_line(self, line: Linea, x: int, dir_pair=(1, -1)) -> None:
        line.ensure_capacity_for(2)
        k = line.Njumps
        line.dir[k] = dir_pair[0]
        line.pos[k] = x
        line.dir[k + 1] = dir_pair[1]
        line.pos[k + 1] = x
        line.Njumps += 2

    def ComputeAnimation(self, skip_time_update: bool = False) -> None:
        height = self.Altezza
        if height <= 0:
            return

        if (not skip_time_update) and (not self.freeze):
            self.tOld = self.t
            if self.DirTempo:
                self.t = min(self.t + 1, height - 3)
            else:
                self.t = max(self.t - 1, 0)

        if self.freeze or (self.tOld == self.t):
            return

        if self.DirTempo:
            # move jumps
            for L in self.iter_lines():
                for i in range(L.Njumps):
                    L.pos[i] = L.pos[i] - L.dir[i]

            # cancellations + nucleations below
            LineaTemp = self.LineaFinale
            while LineaTemp is not None and LineaTemp.prec is not None:
                upper = LineaTemp.prec
                for n in range(1, max(1, self.TotPunti - 1)):
                    limit = upper.Njumps - n
                    for k in range(0, max(0, limit)):
                        if (
                            upper.pos[k] == upper.pos[k + n]
                            and upper.dir[k] == -1
                            and upper.dir[k + n] == 1
                        ):
                            upper.dir[k] = 0
                            upper.dir[k + n] = 0
                            upper.NCancel += 2

                            LineaTemp.ensure_capacity_for(2)
                            m = LineaTemp.Njumps
                            LineaTemp.dir[m] = 1
                            LineaTemp.pos[m] = upper.pos[k + n]
                            LineaTemp.dir[m + 1] = -1
                            LineaTemp.pos[m + 1] = upper.pos[k]
                            LineaTemp.Njumps += 2

                LineaTemp = LineaTemp.prec

            # nucleations on level 0
            base = self.LineaIniziale
            if base is not None:
                for p in self.iter_points():
                    if (height - p.y - 2) == self.t:
                        self._add_nucleation_to_line(base, p.x, dir_pair=(1, -1))

        else:
            # backwards: mark cancellations & create on upper
            for L in self.iter_lines():
                for n in range(1, max(1, self.TotPunti - 1)):
                    limit = L.Njumps - n
                    for k in range(0, max(0, limit)):
                        if (
                            L.pos[k + n] == L.pos[k]
                            and L.dir[k] == 1
                            and L.dir[k + n] == -1
                        ):
                            L.dir[k] = 0
                            L.dir[k + n] = 0
                            L.NCancel += 2
                            if L.prec is not None:
                                upper = L.prec
                                upper.ensure_capacity_for(2)
                                m = upper.Njumps
                                upper.dir[m] = -1
                                upper.pos[m] = L.pos[k + n]
                                upper.dir[m + 1] = 1
                                upper.pos[m + 1] = L.pos[k]
                                upper.Njumps += 2

            # move jumps backwards
            for L in self.iter_lines():
                for i in range(L.Njumps):
                    L.pos[i] = L.pos[i] + L.dir[i]

        # compact
        for L in self.iter_lines():
            m = 0
            for i in range(L.Njumps):
                if L.dir[i] == 0:
                    m += 1
                else:
                    if L.dir[i] != 2:
                        L.pos[i - m] = L.pos[i]
                        L.dir[i - m] = L.dir[i]
                        if m > 0:
                            L.pos[i] = 0
                            L.dir[i] = 2
            L.Njumps -= L.NCancel
            L.NCancel = 0

        # add new last line if needed
        if self.LineaFinale is not None and self.LineaFinale.Njumps > 0:
            old_last = self.LineaFinale
            new_last = Linea(max(4, 2 * self.TotPunti))
            old_last.next = new_last
            new_last.prec = old_last
            self.LineaFinale = new_last

        # delete empty lines except last
        if self.LineaFinale is not None and self.LineaFinale.prec is not None:
            prev = self.LineaFinale.prec
            if prev.Njumps == 0 and prev is not self.LineaIniziale and prev.prec is not None:
                prevprec = prev.prec
                prevprec.next = self.LineaFinale
                self.LineaFinale.prec = prevprec

        # reorder jumps
        for L in self.iter_lines():
            pairs = list(zip(L.pos[:L.Njumps], L.dir[:L.Njumps]))
            pairs.sort(key=lambda z: z[0])
            for i, (px, dr) in enumerate(pairs):
                L.pos[i] = px
                L.dir[i] = dr
            for i in range(L.Njumps):
                for j in range(i + 1, L.Njumps):
                    if L.pos[i] == L.pos[j] and L.dir[i] == -1 and L.dir[j] == 1:
                        L.dir[i] = 1
                        L.dir[j] = -1


# -----------------------------
# Drawing helpers
# -----------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def draw_text(surface, text, x, y, font, color=(0, 0, 0)):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def draw_step_line(screen, rect, line: Optional[Linea], H0: int, D: int, color, thickness=2):
    """
    Draw the step line. If line is None => draw flat baseline.
    """
    W = rect.width
    if line is None or line.Njumps == 0:
        pygame.draw.line(screen, color,
                         (rect.left + 0, rect.top + H0),
                         (rect.left + W - 1, rect.top + H0), thickness)
        return

    n = 0
    m = 0
    x0 = clamp(line.pos[0], 0, W - 1)
    pygame.draw.line(screen, color,
                     (rect.left + 0, rect.top + H0),
                     (rect.left + x0, rect.top + H0), thickness)

    for k in range(line.Njumps - 1):
        m = m + line.dir[k]
        xk = clamp(line.pos[k], 0, W - 1)
        xk1 = clamp(line.pos[k + 1], 0, W - 1)

        y_from = H0 - n * D
        y_to = H0 - m * D

        pygame.draw.line(screen, color,
                         (rect.left + xk, rect.top + y_from),
                         (rect.left + xk, rect.top + y_to), thickness)
        pygame.draw.line(screen, color,
                         (rect.left + xk, rect.top + y_to),
                         (rect.left + xk1, rect.top + y_to), thickness)
        n = n + line.dir[k]

    k = line.Njumps - 1
    m = m + line.dir[k]
    xk = clamp(line.pos[k], 0, W - 1)
    y_from = H0 - n * D
    y_to = H0 - m * D
    pygame.draw.line(screen, color,
                     (rect.left + xk, rect.top + y_from),
                     (rect.left + xk, rect.top + y_to), thickness)
    pygame.draw.line(screen, color,
                     (rect.left + xk, rect.top + y_to),
                     (rect.left + W - 1, rect.top + y_to), thickness)


def draw_water_gap(screen, rect, H0: int, D: int, base_line: Optional[Linea], aug_line: Optional[Linea]):
    """
    Efficient: merge breakpoints and fill rectangles where augmented height exceeds base height.
    If a line is None, treat it as flat (no jumps).
    """
    WATER_UNIT_ONLY = True  # you said “unit shaded region”; set False for “all positive discrepancy”

    # gather jumps (empty if None)
    base_jumps = [] if (base_line is None) else [(base_line.pos[i], base_line.dir[i]) for i in range(base_line.Njumps)]
    aug_jumps = [] if (aug_line is None) else [(aug_line.pos[i], aug_line.dir[i]) for i in range(aug_line.Njumps)]
    base_jumps.sort(key=lambda z: z[0])
    aug_jumps.sort(key=lambda z: z[0])

    bp = {0, rect.width}
    for x, _ in base_jumps:
        bp.add(clamp(x, 0, rect.width))
    for x, _ in aug_jumps:
        bp.add(clamp(x, 0, rect.width))
    breakpoints = sorted(bp)

    hb = 0
    ha = 0
    ib = 0
    ia = 0

    for idx in range(len(breakpoints) - 1):
        x_left = breakpoints[idx]
        x_right = breakpoints[idx + 1]
        if x_right <= x_left:
            continue

        while ib < len(base_jumps) and clamp(base_jumps[ib][0], 0, rect.width) == x_left:
            hb += base_jumps[ib][1]
            ib += 1
        while ia < len(aug_jumps) and clamp(aug_jumps[ia][0], 0, rect.width) == x_left:
            ha += aug_jumps[ia][1]
            ia += 1

        ok = (ha == hb + 1) if WATER_UNIT_ONLY else (ha > hb)
        if ok:
            y_top = H0 - ha * D
            y_bot = H0 - hb * D
            if y_bot > y_top:
                pygame.draw.rect(
                    screen,
                    (150, 150, 150),
                    pygame.Rect(rect.left + x_left, rect.top + y_top, x_right - x_left, y_bot - y_top),
                )


# -----------------------------
# Main loop (coupled sims)
# -----------------------------

def main():
    pygame.init()
    pygame.display.set_caption("PNG coupling (multi-level water)")

    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

    font = pygame.font.SysFont(None, 18)
    font2 = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    sim = PNGSim()
    sim_aug = PNGSim()

    augmented_on = True
    extra_point: Optional[Tuple[int, int]] = None

    def sync_aug_from_base():
        nonlocal extra_point

        # copy parameters
        for s in (sim_aug,):
            s.NPunti = sim.NPunti
            s.Raggio = sim.Raggio
            s.DLinee = sim.DLinee
            s.NLinee = sim.NLinee
            s.Speed = sim.Speed
            s.LSource = sim.LSource
            s.RSource = sim.RSource
            s.TypeOfPNG = sim.TypeOfPNG
            s.freeze = sim.freeze
            s.DirTempo = sim.DirTempo
            s.Larghezza = sim.Larghezza
            s.Altezza = sim.Altezza

        # build base points
        sim.recompute_totpunti()
        base_points = sim.ComputePoisson_points_list()
        sim.set_points_from_list(base_points)

        # build augmented points
        if augmented_on:
            if sim.TypeOfPNG in (0, 2):
                extra_point = sim.sample_point_in_cone(sim.Larghezza, sim.Altezza)
            else:
                x = 4 + int((sim.Larghezza + 2 * sim.Altezza - 9) * random.random()) - sim.Altezza
                y = 4 + int((sim.Altezza - 9) * random.random())
                y = ((x + y) % 2) + y
                if y > sim.Altezza - 5:
                    y = sim.Altezza - 5
                extra_point = (x, y)
            aug_points = base_points + [extra_point]
        else:
            extra_point = None
            aug_points = base_points

        sim_aug.set_points_from_list(aug_points)

        # reset time + lines
        sim.t = sim.tOld = 0
        sim.InitLines()
        sim_aug.t = sim_aug.tOld = 0
        sim_aug.InitLines()

        sim.ComputedPoisson = True
        sim_aug.ComputedPoisson = True

    running = True
    while running:
        W, H = screen.get_size()
        CONTROL_W = min(380, max(240, W // 4))
        ANIM_W = max(10, W - CONTROL_W)
        LINES_H = H // 2
        POINTS_H = H - LINES_H

        sim.Larghezza = ANIM_W
        sim.Altezza = POINTS_H
        sim_aug.Larghezza = ANIM_W
        sim_aug.Altezza = POINTS_H

        # keep spacing reasonable
        sim.DLinee = clamp(sim.DLinee, 2, max(2, LINES_H // 2))
        sim_aug.DLinee = sim.DLinee

        if not sim.ComputedPoisson:
            sync_aug_from_base()

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.freeze = not sim.freeze
                    sim_aug.freeze = sim.freeze
                elif event.key == pygame.K_r:
                    sim.DirTempo = not sim.DirTempo
                    sim_aug.DirTempo = sim.DirTempo
                elif event.key == pygame.K_n:
                    sync_aug_from_base()
                elif event.key == pygame.K_a:
                    augmented_on = not augmented_on
                    sync_aug_from_base()
                elif event.key == pygame.K_1:
                    sim.TypeOfPNG = 1
                    sim_aug.TypeOfPNG = 1
                    sync_aug_from_base()
                elif event.key == pygame.K_2:
                    sim.TypeOfPNG = 0
                    sim_aug.TypeOfPNG = 0
                    sync_aug_from_base()
                elif event.key == pygame.K_3:
                    sim.TypeOfPNG = 2
                    sim_aug.TypeOfPNG = 2
                    sync_aug_from_base()
                elif event.key == pygame.K_UP:
                    sim.Speed = clamp(sim.Speed + 5, 1, 240)
                    sim_aug.Speed = sim.Speed
                elif event.key == pygame.K_DOWN:
                    sim.Speed = clamp(sim.Speed - 5, 1, 240)
                    sim_aug.Speed = sim.Speed
                elif event.key == pygame.K_LEFTBRACKET:
                    if mods & pygame.KMOD_SHIFT:
                        sim.NLinee = clamp(sim.NLinee - 1, 1, 200)
                        sim_aug.NLinee = sim.NLinee
                    else:
                        sim.NPunti = clamp(sim.NPunti - 5, 1, 10000)
                        sim_aug.NPunti = sim.NPunti
                        sync_aug_from_base()
                elif event.key == pygame.K_RIGHTBRACKET:
                    if mods & pygame.KMOD_SHIFT:
                        sim.NLinee = clamp(sim.NLinee + 1, 1, 200)
                        sim_aug.NLinee = sim.NLinee
                    else:
                        sim.NPunti = clamp(sim.NPunti + 5, 1, 10000)
                        sim_aug.NPunti = sim.NPunti
                        sync_aug_from_base()
                elif event.key == pygame.K_MINUS:
                    sim.DLinee = clamp(sim.DLinee - 1, 1, 80)
                    sim_aug.DLinee = sim.DLinee
                elif event.key == pygame.K_EQUALS:
                    sim.DLinee = clamp(sim.DLinee + 1, 1, 80)
                    sim_aug.DLinee = sim.DLinee
                elif event.key == pygame.K_COMMA:
                    sim.LSource = clamp(sim.LSource - 1, 0, 2000)
                    sim_aug.LSource = sim.LSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()
                elif event.key == pygame.K_PERIOD:
                    sim.LSource = clamp(sim.LSource + 1, 0, 2000)
                    sim_aug.LSource = sim.LSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()
                elif event.key == pygame.K_k:
                    sim.RSource = clamp(sim.RSource - 1, 0, 2000)
                    sim_aug.RSource = sim.RSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()
                elif event.key == pygame.K_l:
                    sim.RSource = clamp(sim.RSource + 1, 0, 2000)
                    sim_aug.RSource = sim.RSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()

        # advance both in lockstep time
        if not sim.freeze:
            sim.tOld = sim.t
            if sim.DirTempo:
                sim.t = min(sim.t + 1, POINTS_H - 3)
            else:
                sim.t = max(sim.t - 1, 0)

            sim_aug.tOld = sim.tOld
            sim_aug.t = sim.t
            sim_aug.DirTempo = sim.DirTempo
            sim_aug.freeze = sim.freeze

        sim.ComputeAnimation(skip_time_update=True)
        sim_aug.ComputeAnimation(skip_time_update=True)

        # auto-stop endpoints like Java
        if sim.t == POINTS_H - 3 and not sim.freeze:
            sim.freeze = True
            sim_aug.freeze = True
            sim.DirTempo = False
            sim_aug.DirTempo = False
        if sim.t == 0 and not sim.freeze:
            sim.freeze = True
            sim_aug.freeze = True
            sim.DirTempo = True
            sim_aug.DirTempo = True

        # draw
        screen.fill((235, 235, 235))
        control_rect = pygame.Rect(0, 0, CONTROL_W, H)
        lines_rect = pygame.Rect(CONTROL_W, 0, ANIM_W, LINES_H)
        points_rect = pygame.Rect(CONTROL_W, LINES_H, ANIM_W, POINTS_H)

        pygame.draw.rect(screen, (245, 245, 245), control_rect)
        pygame.draw.rect(screen, (200, 200, 200), control_rect, 1)
        pygame.draw.rect(screen, (255, 255, 255), lines_rect)
        pygame.draw.rect(screen, (220, 220, 220), lines_rect, 1)
        pygame.draw.rect(screen, (255, 255, 255), points_rect)
        pygame.draw.rect(screen, (220, 220, 220), points_rect, 1)

        # cone + points
        cx = CONTROL_W + ANIM_W // 2
        bottom_y = lines_rect.bottom + POINTS_H - 2
        top_y = lines_rect.bottom + 1
        if sim.TypeOfPNG in (0, 2):
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx + POINTS_H - 3, top_y), 2)
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx - POINTS_H + 3, top_y), 2)

        for p in sim.iter_points():
            pygame.draw.circle(screen, (0, 70, 200), (CONTROL_W + p.x, lines_rect.bottom + p.y), sim.Raggio)

        if augmented_on and extra_point is not None:
            ex, ey = extra_point
            pygame.draw.circle(screen, (210, 0, 0), (CONTROL_W + ex, lines_rect.bottom + ey), sim.Raggio + 1)

        # time line
        time_y = lines_rect.bottom + (POINTS_H - sim.t - 2)
        pygame.draw.line(screen, (0, 0, 0), (CONTROL_W, time_y), (CONTROL_W + ANIM_W - 1, time_y), 2)

        # multi-level water + curves
        H0_base = max(LINES_H // 2, LINES_H - sim.DLinee * sim.NLinee)
        for level in range(sim.NLinee):
            H0 = H0_base + level * sim.DLinee
            baseL = sim.line_at_level(level)
            augL = sim_aug.line_at_level(level)

            draw_water_gap(screen, lines_rect, H0, sim.DLinee, baseL, augL)
            draw_step_line(screen, lines_rect, baseL, H0, sim.DLinee, (0, 70, 200), thickness=2 if level == 0 else 1)
            draw_step_line(screen, lines_rect, augL,  H0, sim.DLinee, (210, 0, 0), thickness=2 if level == 0 else 1)

        # control panel
        geom_name = {0: "Droplet", 1: "Flat", 2: "Droplet+Sources"}[sim.TypeOfPNG]
        draw_text(screen, "PNG coupling (multi-level water)", 12, 10, font2)

        y = 44
        draw_text(screen, f"Status: {'RUNNING' if not sim.freeze else 'PAUSED'} (Space)", 12, y, font); y += 20
        draw_text(screen, f"Time dir: {'FORWARD' if sim.DirTempo else 'BACKWARD'} (R)", 12, y, font); y += 20
        draw_text(screen, f"Geometry: {geom_name} (1/2/3)", 12, y, font); y += 20
        draw_text(screen, f"Extra point: {augmented_on} (A)   New points: (N)", 12, y, font); y += 20
        draw_text(screen, "Blue=original, Red=augmented, Grey=unit water", 12, y, font); y += 20

        pygame.display.flip()
        clock.tick(sim.Speed)

    pygame.quit()


if __name__ == "__main__":
    main()