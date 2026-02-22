"""
Weighted PNG coupling (pygame): original vs augmented with ONE extra nucleation of weight w.

Key idea:
- Each jump has direction (+1/-1) and weight w (float).
- Heights are real-valued.
- When a DOWN jump meets an UP jump at the same position, they partially cancel:
    transferred = min(w_down, w_up)
    residual stays on the same line, transferred falls down as a weighted nucleation.

Controls:
  Space  : Start/Stop
  R      : Toggle time direction
  N      : New base configuration (restart; refresh extra point if enabled)
  A      : Toggle extra point on/off (restart)
  1/2/3  : Geometry (Flat / Droplet / Droplet+Sources)
  Up/Down: Speed +/- (FPS)
  [ / ]  : NPunti -/+ (restarts)
  { / }  : NLinee -/+  (Shift+[ and Shift+])
  - / =  : DLinee -/+
  , / .  : LSource -/+ (restarts in sources geometry)
  Q / W  : RSource -/+ (restarts in sources geometry)
  J / K  : Extra point weight w  down/up (restarts)

Notes:
- In droplet mode, the cone is symmetric: points appear on both sides of the centerline by design.
- Grey shading is the positive discrepancy (augmented height minus base height), per level.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pygame


@dataclass
class Punto:
    x: int
    y: int
    next: Optional["Punto"] = None


class Linea:
    """
    A line is a multiset of jumps, each with:
      pos[i] : x-coordinate (int)
      dir[i] : +1 (UP), -1 (DOWN)
      w[i]   : weight (float > 0)
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.Njumps = 0
        self.pos = [0] * capacity
        self.dir = [0] * capacity
        self.w = [0.0] * capacity
        self.next: Optional["Linea"] = None
        self.prec: Optional["Linea"] = None

    def ensure_capacity_for(self, add: int) -> None:
        if self.Njumps + add <= self.capacity:
            return
        newcap = max(self.capacity * 2, self.Njumps + add)
        self.pos.extend([0] * (newcap - self.capacity))
        self.dir.extend([0] * (newcap - self.capacity))
        self.w.extend([0.0] * (newcap - self.capacity))
        self.capacity = newcap


class PNGSim:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

        self.NPunti = 200
        self.Raggio = 3
        self.DLinee = 12
        self.NLinee = 10

        self.Speed = 30
        self.LSource = 10
        self.RSource = 5

        self.TypeOfPNG = 0  # start in droplet by default now

        self.freeze = True
        self.DirTempo = True

        self.Larghezza = 0
        self.Altezza = 0

        self.TotPunti = self.NPunti
        self.t = 0
        self.tOld = 0
        self.ComputedPoisson = False

        self.PuntoIniziale = Punto(0, 0)
        self._build_point_list()

        self.LineaIniziale: Optional[Linea] = None
        self.LineaFinale: Optional[Linea] = None
        self.InitLines()

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

    def line_at_level(self, level: int) -> Optional[Linea]:
        L = self.LineaIniziale
        i = 0
        while L is not None and i < level:
            L = L.next
            i += 1
        return L

    def iter_lines(self):
        L = self.LineaIniziale
        while L is not None:
            yield L
            L = L.next

    def InitLines(self) -> None:
        cap = max(8, 2 * self.TotPunti)
        self.LineaIniziale = Linea(cap)
        self.LineaFinale = Linea(cap)
        self.LineaIniziale.next = self.LineaFinale
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

    # ---------- Poisson sampling

    def sample_point_in_cone(self, width: int, height: int) -> Tuple[int, int]:
        cx = width // 2
        while True:
            x = 4 + int((width - 9) * random.random())
            y = 4 + int((height - 9) * random.random())
            if abs(x - cx) <= (height - 2 - y):
                y = ((x + y) % 2) + y
                y = min(y, height - 5)
                return x, y

    def ComputePoisson_points_list(self) -> List[Tuple[int, int]]:
        width, height = self.Larghezza, self.Altezza
        if width <= 0 or height <= 0:
            return []
        pts: List[Tuple[int, int]] = []
        k = 0
        tot = self.TotPunti

        while len(pts) < tot:
            while True:
                nuovo = True

                if self.TypeOfPNG == 1:  # flat
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
                y = min(y, height - 5)

                if (x, y) in pts:
                    nuovo = False
                if nuovo:
                    break

            pts.append((x, y))
            k += 1

        return pts

    # ---------- weighted jump insertion

    def add_nucleation(self, line: Linea, x: int, weight: float) -> None:
        """
        Add a weighted nucleation: an UP jump and a DOWN jump at same position.
        """
        if weight <= 0:
            return
        line.ensure_capacity_for(2)
        k = line.Njumps
        line.pos[k] = x
        line.dir[k] = +1
        line.w[k] = weight
        line.pos[k + 1] = x
        line.dir[k + 1] = -1
        line.w[k + 1] = weight
        line.Njumps += 2

    # ---------- core dynamics

    def ComputeAnimation(self, skip_time_update: bool = False) -> None:
        height = self.Altezza
        if height <= 0:
            return

        if (not skip_time_update) and (not self.freeze):
            self.tOld = self.t
            self.t = min(self.t + 1, height - 3) if self.DirTempo else max(self.t - 1, 0)

        if self.freeze or (self.tOld == self.t):
            return

        if self.DirTempo:
            # move jumps
            for L in self.iter_lines():
                for i in range(L.Njumps):
                    L.pos[i] -= L.dir[i]

            # collisions + fall-down (process from bottom up, like Java structure)
            LineaTemp = self.LineaFinale
            while LineaTemp is not None and LineaTemp.prec is not None:
                upper = LineaTemp.prec

                # ensure upper jumps are sorted
                self._sort_line_by_pos(upper)

                # scan for same-position DOWN/UP collisions and partially cancel
                i = 0
                while i < upper.Njumps - 1:
                    x = upper.pos[i]
                    # collect all jumps at this x
                    j = i
                    downs: List[int] = []
                    ups: List[int] = []
                    while j < upper.Njumps and upper.pos[j] == x:
                        if upper.dir[j] == -1:
                            downs.append(j)
                        elif upper.dir[j] == +1:
                            ups.append(j)
                        j += 1

                    # cancel greedily between DOWN and UP at this position
                    di = 0
                    ui = 0
                    while di < len(downs) and ui < len(ups):
                        d_idx = downs[di]
                        u_idx = ups[ui]
                        a = upper.w[d_idx]
                        b = upper.w[u_idx]
                        transfer = min(a, b)
                        if transfer > 0:
                            upper.w[d_idx] -= transfer
                            upper.w[u_idx] -= transfer
                            # fall-down nucleation of weight=transfer on the next line
                            LineaTemp.ensure_capacity_for(2)
                            self.add_nucleation(LineaTemp, x, transfer)

                        # advance whichever is exhausted
                        if upper.w[d_idx] <= 1e-12:
                            upper.w[d_idx] = 0.0
                            di += 1
                        if upper.w[u_idx] <= 1e-12:
                            upper.w[u_idx] = 0.0
                            ui += 1

                    i = j

                # remove zero-weight jumps from upper
                self._compact_line(upper)

                LineaTemp = LineaTemp.prec

            # nucleations on level 0 from points at current time (unit weight = 1 here)
            base = self.LineaIniziale
            if base is not None:
                for p in self.iter_points():
                    if (height - p.y - 2) == self.t:
                        self.add_nucleation(base, p.x, 1.0)

        else:
            # Backward-time dynamics for weighted partial cancellation is subtle.
            # For now, keep backward as a "visual rewind" disabled by freezing at endpoints,
            # matching your main use (forward evolution / tracking defect).
            # You can still toggle R, but you should treat it as experimental.
            pass

        # add new last line if last got excited
        if self.LineaFinale is not None and self.LineaFinale.Njumps > 0:
            old_last = self.LineaFinale
            new_last = Linea(max(8, 2 * self.TotPunti))
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

        # sort all lines (stable display + collision logic)
        for L in self.iter_lines():
            self._sort_line_by_pos(L)

    def _sort_line_by_pos(self, L: Linea) -> None:
        if L.Njumps <= 1:
            return
        items = [(L.pos[i], L.dir[i], L.w[i]) for i in range(L.Njumps)]
        items.sort(key=lambda z: z[0])
        for i, (x, d, w) in enumerate(items):
            L.pos[i], L.dir[i], L.w[i] = x, d, w

    def _compact_line(self, L: Linea) -> None:
        m = 0
        for i in range(L.Njumps):
            if L.w[i] <= 1e-12:
                m += 1
            else:
                if m > 0:
                    L.pos[i - m] = L.pos[i]
                    L.dir[i - m] = L.dir[i]
                    L.w[i - m] = L.w[i]
        L.Njumps -= m
        # (no need to wipe tail)


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
    Draw weighted step line: height changes by +/- w at each jump.
    """
    W = rect.width
    if line is None or line.Njumps == 0:
        pygame.draw.line(screen, color,
                         (rect.left + 0, rect.top + H0),
                         (rect.left + W - 1, rect.top + H0), thickness)
        return

    # Ensure sorted by position
    items = [(line.pos[i], line.dir[i], line.w[i]) for i in range(line.Njumps)]
    items.sort(key=lambda z: z[0])

    h = 0.0
    x_prev = 0
    y_prev = H0 - h * D

    # draw from left to first jump
    x0 = clamp(items[0][0], 0, W - 1)
    pygame.draw.line(screen, color,
                     (rect.left + 0, rect.top + y_prev),
                     (rect.left + x0, rect.top + y_prev), thickness)

    for (x, d, w) in items:
        x = clamp(x, 0, W - 1)
        # vertical at x: from current height to updated height
        y_from = H0 - h * D
        h = h + d * w
        y_to = H0 - h * D
        pygame.draw.line(screen, color,
                         (rect.left + x, rect.top + y_from),
                         (rect.left + x, rect.top + y_to), thickness)
        # horizontal until next jump handled by next iteration (we draw when we know next x)
        x_prev = x

        # find next x in items (inefficient but fine at these sizes)
        # we’ll instead do it in a second pass by drawing between successive jumps:
    # second pass for horizontals
    h = 0.0
    for idx in range(len(items) - 1):
        x, d, w = items[idx]
        x_next = items[idx + 1][0]
        x = clamp(x, 0, W - 1)
        x_next = clamp(x_next, 0, W - 1)
        h = h + d * w
        y = H0 - h * D
        pygame.draw.line(screen, color,
                         (rect.left + x, rect.top + y),
                         (rect.left + x_next, rect.top + y), thickness)

    # tail to right edge
    x_last, d_last, w_last = items[-1]
    x_last = clamp(x_last, 0, W - 1)
    h = 0.0
    for (x, d, w) in items:
        h += d * w
    y_tail = H0 - h * D
    pygame.draw.line(screen, color,
                     (rect.left + x_last, rect.top + y_tail),
                     (rect.left + W - 1, rect.top + y_tail), thickness)


def draw_water_gap(screen, rect, H0: int, D: int, base_line: Optional[Linea], aug_line: Optional[Linea]):
    """
    Efficient water shading between two weighted step functions using merged breakpoints.
    Grey rectangles on intervals where aug height > base height.
    """
    base_jumps = [] if base_line is None else [(base_line.pos[i], base_line.dir[i], base_line.w[i]) for i in range(base_line.Njumps)]
    aug_jumps = [] if aug_line is None else [(aug_line.pos[i], aug_line.dir[i], aug_line.w[i]) for i in range(aug_line.Njumps)]
    base_jumps.sort(key=lambda z: z[0])
    aug_jumps.sort(key=lambda z: z[0])

    bp = {0, rect.width}
    for x, _, _ in base_jumps:
        bp.add(clamp(x, 0, rect.width))
    for x, _, _ in aug_jumps:
        bp.add(clamp(x, 0, rect.width))
    breakpoints = sorted(bp)

    hb = 0.0
    ha = 0.0
    ib = 0
    ia = 0

    for idx in range(len(breakpoints) - 1):
        x_left = breakpoints[idx]
        x_right = breakpoints[idx + 1]
        if x_right <= x_left:
            continue

        while ib < len(base_jumps) and clamp(base_jumps[ib][0], 0, rect.width) == x_left:
            _, d, w = base_jumps[ib]
            hb += d * w
            ib += 1
        while ia < len(aug_jumps) and clamp(aug_jumps[ia][0], 0, rect.width) == x_left:
            _, d, w = aug_jumps[ia]
            ha += d * w
            ia += 1

        if ha > hb + 1e-12:
            y_top = H0 - ha * D
            y_bot = H0 - hb * D
            if y_bot > y_top:
                pygame.draw.rect(
                    screen,
                    (150, 150, 150),
                    pygame.Rect(rect.left + x_left, rect.top + int(y_top), x_right - x_left, int(y_bot - y_top)),
                )


# -----------------------------
# Main loop
# -----------------------------

def main():
    pygame.init()
    pygame.display.set_caption("Weighted PNG defect: original vs augmented")

    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

    font = pygame.font.SysFont(None, 18)
    font2 = pygame.font.SysFont(None, 22)
    clock = pygame.time.Clock()

    sim = PNGSim()
    sim_aug = PNGSim()

    augmented_on = True
    extra_point: Optional[Tuple[int, int]] = None
    w_extra = math.sqrt(2.0)

    def sync_aug_from_base():
        nonlocal extra_point

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

        sim.recompute_totpunti()
        base_points = sim.ComputePoisson_points_list()
        sim.set_points_from_list(base_points)

        if augmented_on:
            if sim.TypeOfPNG in (0, 2):
                extra_point = sim.sample_point_in_cone(sim.Larghezza, sim.Altezza)
            else:
                # flat: just add a flat-sampled point
                x = 4 + int((sim.Larghezza + 2 * sim.Altezza - 9) * random.random()) - sim.Altezza
                y = 4 + int((sim.Altezza - 9) * random.random())
                y = ((x + y) % 2) + y
                y = min(y, sim.Altezza - 5)
                extra_point = (x, y)
            aug_points = base_points + [extra_point]
        else:
            extra_point = None
            aug_points = base_points

        sim_aug.set_points_from_list(aug_points)

        # reset time and lines
        sim.t = sim.tOld = 0
        sim.InitLines()
        sim_aug.t = sim_aug.tOld = 0
        sim_aug.InitLines()

        sim.ComputedPoisson = True
        sim_aug.ComputedPoisson = True

    running = True
    while running:
        W, H = screen.get_size()
        CONTROL_W = min(420, max(260, W // 4))
        ANIM_W = max(10, W - CONTROL_W)
        LINES_H = H // 2
        POINTS_H = H - LINES_H

        sim.Larghezza = ANIM_W
        sim.Altezza = POINTS_H
        sim_aug.Larghezza = ANIM_W
        sim_aug.Altezza = POINTS_H

        sim.DLinee = clamp(sim.DLinee, 2, max(2, LINES_H // 2))
        sim_aug.DLinee = sim.DLinee

        if not sim.ComputedPoisson:
            sync_aug_from_base()

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
                    # backward dynamics is not implemented for weighted partial-cancel
                    sim.DirTempo = True
                    sim_aug.DirTempo = True
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

                elif event.key == pygame.K_q:
                    sim.RSource = clamp(sim.RSource - 1, 0, 2000)
                    sim_aug.RSource = sim.RSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()
                elif event.key == pygame.K_w:
                    sim.RSource = clamp(sim.RSource + 1, 0, 2000)
                    sim_aug.RSource = sim.RSource
                    if sim.TypeOfPNG == 2:
                        sync_aug_from_base()

                elif event.key == pygame.K_j:
                    w_extra = max(0.05, w_extra / math.sqrt(2.0))
                    sync_aug_from_base()
                elif event.key == pygame.K_k:
                    w_extra = min(50.0, w_extra * math.sqrt(2.0))
                    sync_aug_from_base()

        # advance (forward only)
        if not sim.freeze:
            sim.tOld = sim.t
            sim.t = min(sim.t + 1, POINTS_H - 3)

            sim_aug.tOld = sim.tOld
            sim_aug.t = sim.t
            sim_aug.freeze = sim.freeze
            sim_aug.DirTempo = True

        # Inject the weighted extra point as a nucleation WHEN ITS TIME ARRIVES (on level 0)
        # We do this by adding it to the augmented point list already, but base nucleations are unit weight.
        # So we override: if the extra point is present, add nucleation weight w_extra when it triggers.
        #
        # Implementation trick:
        # - Run sim normally.
        # - Run sim_aug normally, but when iterating points at phase 3, the extra point will be treated as unit.
        # To fix that, we detect when a point equals extra_point and add with weight w_extra.
        #
        # Simplest: temporarily monkeypatch the point weights by duplicating logic for sim_aug here.
        sim.ComputeAnimation(skip_time_update=True)

        # custom step for sim_aug forward:
        if not sim_aug.freeze and sim_aug.DirTempo:
            # move jumps
            for L in sim_aug.iter_lines():
                for i in range(L.Njumps):
                    L.pos[i] -= L.dir[i]

            # collisions + fall-down
            LineaTemp = sim_aug.LineaFinale
            while LineaTemp is not None and LineaTemp.prec is not None:
                upper = LineaTemp.prec
                sim_aug._sort_line_by_pos(upper)

                i = 0
                while i < upper.Njumps - 1:
                    x = upper.pos[i]
                    j = i
                    downs, ups = [], []
                    while j < upper.Njumps and upper.pos[j] == x:
                        if upper.dir[j] == -1:
                            downs.append(j)
                        elif upper.dir[j] == +1:
                            ups.append(j)
                        j += 1
                    di = 0
                    ui = 0
                    while di < len(downs) and ui < len(ups):
                        d_idx = downs[di]
                        u_idx = ups[ui]
                        a = upper.w[d_idx]
                        b = upper.w[u_idx]
                        transfer = min(a, b)
                        if transfer > 0:
                            upper.w[d_idx] -= transfer
                            upper.w[u_idx] -= transfer
                            sim_aug.add_nucleation(LineaTemp, x, transfer)
                        if upper.w[d_idx] <= 1e-12:
                            upper.w[d_idx] = 0.0
                            di += 1
                        if upper.w[u_idx] <= 1e-12:
                            upper.w[u_idx] = 0.0
                            ui += 1
                    i = j

                sim_aug._compact_line(upper)
                LineaTemp = LineaTemp.prec

            # nucleations on level 0 from points at current time:
            base = sim_aug.LineaIniziale
            if base is not None:
                for p in sim_aug.iter_points():
                    if (POINTS_H - p.y - 2) == sim_aug.t:
                        if augmented_on and extra_point is not None and (p.x, p.y) == extra_point:
                            sim_aug.add_nucleation(base, p.x, w_extra)
                        else:
                            sim_aug.add_nucleation(base, p.x, 1.0)

            # line housekeeping
            if sim_aug.LineaFinale is not None and sim_aug.LineaFinale.Njumps > 0:
                old_last = sim_aug.LineaFinale
                new_last = Linea(max(8, 2 * sim_aug.TotPunti))
                old_last.next = new_last
                new_last.prec = old_last
                sim_aug.LineaFinale = new_last

            if sim_aug.LineaFinale is not None and sim_aug.LineaFinale.prec is not None:
                prev = sim_aug.LineaFinale.prec
                if prev.Njumps == 0 and prev is not sim_aug.LineaIniziale and prev.prec is not None:
                    prevprec = prev.prec
                    prevprec.next = sim_aug.LineaFinale
                    sim_aug.LineaFinale.prec = prevprec

            for L in sim_aug.iter_lines():
                sim_aug._sort_line_by_pos(L)

        # auto-stop endpoints
        if sim.t == POINTS_H - 3 and not sim.freeze:
            sim.freeze = True
            sim_aug.freeze = True
        if sim.t == 0 and not sim.freeze:
            sim.freeze = True
            sim_aug.freeze = True

        # ----- draw
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

        # cone
        cx = CONTROL_W + ANIM_W // 2
        bottom_y = lines_rect.bottom + POINTS_H - 2
        top_y = lines_rect.bottom + 1
        if sim.TypeOfPNG in (0, 2):
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx + POINTS_H - 3, top_y), 2)
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx - POINTS_H + 3, top_y), 2)

        # points
        for p in sim.iter_points():
            pygame.draw.circle(screen, (0, 70, 200), (CONTROL_W + p.x, lines_rect.bottom + p.y), sim.Raggio)

        if augmented_on and extra_point is not None:
            ex, ey = extra_point
            pygame.draw.circle(screen, (210, 0, 0), (CONTROL_W + ex, lines_rect.bottom + ey), sim.Raggio + 1)

        # time line
        time_y = lines_rect.bottom + (POINTS_H - sim.t - 2)
        pygame.draw.line(screen, (0, 0, 0), (CONTROL_W, time_y), (CONTROL_W + ANIM_W - 1, time_y), 2)

        # multilayer water + lines
        H0_base = max(LINES_H // 2, LINES_H - sim.DLinee * sim.NLinee)
        for level in range(sim.NLinee):
            H0 = H0_base + level * sim.DLinee
            baseL = sim.line_at_level(level)
            augL = sim_aug.line_at_level(level)

            draw_water_gap(screen, lines_rect, H0, sim.DLinee, baseL, augL)
            draw_step_line(screen, lines_rect, baseL, H0, sim.DLinee, (0, 70, 200), thickness=2 if level == 0 else 1)
            draw_step_line(screen, lines_rect, augL,  H0, sim.DLinee, (210, 0, 0), thickness=2 if level == 0 else 1)

        # control panel text
        geom_name = {0: "Droplet", 1: "Flat", 2: "Droplet+Sources"}[sim.TypeOfPNG]
        draw_text(screen, "Weighted PNG defect (multi-level water)", 12, 10, font2)
        y = 44
        draw_text(screen, f"{'RUNNING' if not sim.freeze else 'PAUSED'}  (Space)", 12, y, font); y += 18
        draw_text(screen, f"Geometry: {geom_name} (1/2/3)", 12, y, font); y += 18
        draw_text(screen, f"Extra point: {augmented_on} (A)  weight w={w_extra:.4g} (J/K)", 12, y, font); y += 18
        draw_text(screen, "Blue=original, Red=augmented, Grey=water (aug>base)", 12, y, font); y += 18
        draw_text(screen, f"NPunti={sim.NPunti}  NLinee={sim.NLinee}  DLinee={sim.DLinee}", 12, y, font); y += 18
        draw_text(screen, f"Speed={sim.Speed} FPS  (Up/Down)", 12, y, font)

        pygame.display.flip()
        clock.tick(sim.Speed)

    pygame.quit()


if __name__ == "__main__":
    main()