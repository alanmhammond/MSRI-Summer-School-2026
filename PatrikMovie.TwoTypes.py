from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pygame


# -----------------------------
# Poisson sampler (no numpy)
# -----------------------------
def poisson_sample(mean: float) -> int:
    """Poisson(mean) using Knuth for small mean, normal approx for large mean."""
    if mean <= 0:
        return 0
    if mean < 50.0:
        L = math.exp(-mean)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
    # Normal approximation for speed when mean is large (simulation-grade)
    val = int(random.gauss(mean, math.sqrt(mean)) + 0.5)
    return max(0, val)


# -----------------------------
# Point / line structures
# -----------------------------
@dataclass
class PointW:
    x: int
    y: int
    w: float
    color: Tuple[int, int, int]


class Linea:
    """
    Weighted jumps:
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


# -----------------------------
# Geometry helpers
# -----------------------------
def droplet_allowed_interval(width: int, height: int, y: int) -> Optional[Tuple[int, int]]:
    """
    Droplet cone: apex at (cx, height-2), rays slope 1.
    Condition: abs(x-cx) <= height-2-y
    Also keep within margins: x in [4, width-5], y in [4, height-5]
    """
    if y < 4 or y > height - 5:
        return None
    cx = width // 2
    r = (height - 2 - y)
    lo = max(4, cx - r)
    hi = min(width - 5, cx + r)
    if hi < lo:
        return None
    return lo, hi


def droplet_area(width: int, height: int) -> int:
    """Discrete area = number of integer lattice sites (x,y) allowed (with margins)."""
    total = 0
    for y in range(4, height - 4):
        iv = droplet_allowed_interval(width, height, y)
        if iv is None:
            continue
        lo, hi = iv
        total += (hi - lo + 1)
    return total


def flat_area(width: int, height: int) -> int:
    """Discrete area for the 'flat' sampling rectangle used by the Java logic."""
    return max(0, (height - 9)) * max(0, (width + 2 * height - 9))


# -----------------------------
# Simulation
# -----------------------------
class PNGWeighted:
    def __init__(self):
        # display / dynamics parameters
        self.freeze = True
        self.Speed = 30
        self.DLinee = 12
        self.NLinee = 14

        # geometry: 0 droplet, 1 flat
        self.TypeOfPNG = 0

        # intensities per lattice site
        self.lam1 = 0.020  # weight 1 (blue)
        self.lam2 = 0.012  # weight sqrt2 (red)

        self.w1 = 1.0
        self.w2 = math.sqrt(2.0)

        # panel sizes set from main loop
        self.W = 0
        self.H_points = 0

        self.t = 0
        self.tOld = 0

        # points list
        self.points: List[PointW] = []
        self.last_counts = (0, 0)  # (n1, n2)

        # lines as linked list
        self.LineaIniziale: Optional[Linea] = None
        self.LineaFinale: Optional[Linea] = None

    def init_lines(self, cap: int = 256) -> None:
        self.LineaIniziale = Linea(cap)
        self.LineaFinale = Linea(cap)
        self.LineaIniziale.next = self.LineaFinale
        self.LineaFinale.prec = self.LineaIniziale
        self.LineaFinale.next = None

    def reset(self) -> None:
        self.t = 0
        self.tOld = 0
        self.init_lines()

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

    def add_nucleation(self, line: Linea, x: int, weight: float) -> None:
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

    def _sort_line(self, L: Linea) -> None:
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

    def generate_cloud(self) -> None:
        """
        Two independent Poisson clouds:
          Blue: weight 1, intensity lam1
          Red : weight sqrt2, intensity lam2
        Domain depends on geometry and current panel size.
        """
        width, height = self.W, self.H_points
        if width <= 0 or height <= 0:
            self.points = []
            self.last_counts = (0, 0)
            return

        if self.TypeOfPNG == 0:
            area = droplet_area(width, height)
        else:
            area = flat_area(width, height)

        n1 = poisson_sample(self.lam1 * area)
        n2 = poisson_sample(self.lam2 * area)
        self.last_counts = (n1, n2)

        pts: List[PointW] = []
        used = set()  # avoid duplicates

        def sample_droplet() -> Tuple[int, int]:
            cx = width // 2
            while True:
                x = 4 + int((width - 9) * random.random())
                y = 4 + int((height - 9) * random.random())
                if abs(x - cx) <= (height - 2 - y):
                    # parity snap like the Java code
                    y = ((x + y) % 2) + y
                    y = min(y, height - 5)
                    return x, y

        def sample_flat() -> Tuple[int, int]:
            x = 4 + int((width + 2 * height - 9) * random.random()) - height
            y = 4 + int((height - 9) * random.random())
            y = ((x + y) % 2) + y
            y = min(y, height - 5)
            return x, y

        sampler = sample_droplet if self.TypeOfPNG == 0 else sample_flat

        # blue cloud
        for _ in range(n1):
            while True:
                x, y = sampler()
                key = (x, y)
                if key not in used:
                    used.add(key)
                    pts.append(PointW(x, y, self.w1, (0, 70, 200)))
                    break

        # red cloud
        for _ in range(n2):
            while True:
                x, y = sampler()
                key = (x, y)
                if key not in used:
                    used.add(key)
                    pts.append(PointW(x, y, self.w2, (210, 0, 0)))
                    break

        self.points = pts

    def step_forward_one_tick(self) -> None:
        """Forward-only evolution, weighted collisions with partial fall-down."""
        height = self.H_points
        if height <= 0 or self.LineaIniziale is None or self.LineaFinale is None:
            return

        self.tOld = self.t
        self.t = min(self.t + 1, height - 3)
        if self.tOld == self.t:
            return

        # 1) move jumps
        for L in self.iter_lines():
            for i in range(L.Njumps):
                L.pos[i] -= L.dir[i]

        # 2) collisions + fall-down (bottom-up)
        LineaTemp = self.LineaFinale
        while LineaTemp is not None and LineaTemp.prec is not None:
            upper = LineaTemp.prec
            self._sort_line(upper)

            i = 0
            while i < upper.Njumps - 1:
                x = upper.pos[i]
                j = i
                downs: List[int] = []
                ups: List[int] = []
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
                        self.add_nucleation(LineaTemp, x, transfer)

                    if upper.w[d_idx] <= 1e-12:
                        upper.w[d_idx] = 0.0
                        di += 1
                    if upper.w[u_idx] <= 1e-12:
                        upper.w[u_idx] = 0.0
                        ui += 1

                i = j

            self._compact_line(upper)
            LineaTemp = LineaTemp.prec

        # 3) nucleations on level 0 from points at current time
        base = self.LineaIniziale
        for p in self.points:
            if (height - p.y - 2) == self.t:
                self.add_nucleation(base, p.x, p.w)

        # 4) extend tail if excited
        if self.LineaFinale is not None and self.LineaFinale.Njumps > 0:
            old_last = self.LineaFinale
            new_last = Linea(max(64, old_last.capacity))
            old_last.next = new_last
            new_last.prec = old_last
            self.LineaFinale = new_last

        # 5) prune empty lines except last
        if self.LineaFinale is not None and self.LineaFinale.prec is not None:
            prev = self.LineaFinale.prec
            if prev.Njumps == 0 and prev is not self.LineaIniziale and prev.prec is not None:
                prevprec = prev.prec
                prevprec.next = self.LineaFinale
                self.LineaFinale.prec = prevprec

        # 6) sort for drawing
        for L in self.iter_lines():
            self._sort_line(L)


# -----------------------------
# Drawing helpers
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def draw_step_line(screen, rect, line: Optional[Linea], H0: int, D: int, color, thickness=2):
    W = rect.width
    if line is None or line.Njumps == 0:
        pygame.draw.line(screen, color,
                         (rect.left + 0, rect.top + H0),
                         (rect.left + W - 1, rect.top + H0), thickness)
        return

    items = [(line.pos[i], line.dir[i], line.w[i]) for i in range(line.Njumps)]
    items.sort(key=lambda z: z[0])

    h = 0.0
    y0 = H0 - h * D
    x_first = clamp(items[0][0], 0, W - 1)
    pygame.draw.line(screen, color,
                     (rect.left + 0, rect.top + y0),
                     (rect.left + x_first, rect.top + y0), thickness)

    for idx, (x, d, w) in enumerate(items):
        x = clamp(x, 0, W - 1)
        y_from = H0 - h * D
        h = h + d * w
        y_to = H0 - h * D
        pygame.draw.line(screen, color,
                         (rect.left + x, rect.top + y_from),
                         (rect.left + x, rect.top + y_to), thickness)
        x_next = clamp(items[idx + 1][0], 0, W - 1) if idx + 1 < len(items) else (W - 1)
        pygame.draw.line(screen, color,
                         (rect.left + x, rect.top + y_to),
                         (rect.left + x_next, rect.top + y_to), thickness)


def draw_text(surface, text, x, y, font, color=(0, 0, 0)):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


# -----------------------------
# Main
# -----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("Two-colour weighted Poisson cloud → multi-line PNG (resamples on resize)")

    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 18)
    font2 = pygame.font.SysFont(None, 22)

    sim = PNGWeighted()
    sim.init_lines()

    # track last panel size; on resize we resample cloud + reset lines/time
    last_anim_size: Optional[Tuple[int, int]] = None

    def restart_cloud(pause: bool = True) -> None:
        if pause:
            sim.freeze = True
        sim.reset()
        sim.generate_cloud()

    running = True
    first = True

    while running:
        W, H = screen.get_size()
        CONTROL_W = min(420, max(260, W // 4))
        ANIM_W = max(10, W - CONTROL_W)
        LINES_H = H // 2
        POINTS_H = H - LINES_H

        sim.W = ANIM_W
        sim.H_points = POINTS_H
        sim.DLinee = clamp(sim.DLinee, 2, max(2, LINES_H // 2))

        # Resample on first draw or on resize of the animation panel
        anim_size = (ANIM_W, POINTS_H)
        if first or (last_anim_size is not None and anim_size != last_anim_size):
            last_anim_size = anim_size
            restart_cloud(pause=True)
            first = False
        elif last_anim_size is None:
            last_anim_size = anim_size

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim.freeze = not sim.freeze
                elif event.key == pygame.K_n:
                    restart_cloud(pause=True)
                elif event.key == pygame.K_UP:
                    sim.Speed = clamp(sim.Speed + 5, 1, 240)
                elif event.key == pygame.K_DOWN:
                    sim.Speed = clamp(sim.Speed - 5, 1, 240)
                elif event.key == pygame.K_MINUS:
                    sim.DLinee = clamp(sim.DLinee - 1, 1, 80)
                elif event.key == pygame.K_EQUALS:
                    sim.DLinee = clamp(sim.DLinee + 1, 1, 80)

                # NLinee: Shift+[ / Shift+]
                elif event.key == pygame.K_LEFTBRACKET and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    sim.NLinee = clamp(sim.NLinee - 1, 1, 300)
                elif event.key == pygame.K_RIGHTBRACKET and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    sim.NLinee = clamp(sim.NLinee + 1, 1, 300)

                # Geometry
                elif event.key == pygame.K_1:
                    sim.TypeOfPNG = 1
                    restart_cloud(pause=True)
                elif event.key == pygame.K_2:
                    sim.TypeOfPNG = 0
                    restart_cloud(pause=True)

                # Intensities (resample)
                elif event.key == pygame.K_a:
                    sim.lam1 = max(0.0, sim.lam1 / math.sqrt(2.0))
                    restart_cloud(pause=True)
                elif event.key == pygame.K_s:
                    sim.lam1 = min(1.0, sim.lam1 * math.sqrt(2.0))
                    restart_cloud(pause=True)
                elif event.key == pygame.K_k:
                    sim.lam2 = max(0.0, sim.lam2 / math.sqrt(2.0))
                    restart_cloud(pause=True)
                elif event.key == pygame.K_l:
                    sim.lam2 = min(1.0, sim.lam2 * math.sqrt(2.0))
                    restart_cloud(pause=True)

        if not sim.freeze:
            sim.step_forward_one_tick()
            if sim.t == POINTS_H - 3:
                sim.freeze = True  # stop at top

        # layout
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

        # droplet cone boundary
        cx = CONTROL_W + ANIM_W // 2
        bottom_y = lines_rect.bottom + POINTS_H - 2
        top_y = lines_rect.bottom + 1
        if sim.TypeOfPNG == 0:
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx + POINTS_H - 3, top_y), 2)
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx - POINTS_H + 3, top_y), 2)

        # points
        for p in sim.points:
            pygame.draw.circle(
                screen,
                p.color,
                (CONTROL_W + p.x, lines_rect.bottom + p.y),
                3
            )

        # time line
        time_y = lines_rect.bottom + (POINTS_H - sim.t - 2)
        pygame.draw.line(screen, (0, 0, 0), (CONTROL_W, time_y), (CONTROL_W + ANIM_W - 1, time_y), 2)

        # PNG lines (weighted)
        H0_base = max(LINES_H // 2, LINES_H - sim.DLinee * sim.NLinee)
        for level in range(sim.NLinee):
            H0 = H0_base + level * sim.DLinee
            L = sim.line_at_level(level)
            draw_step_line(screen, lines_rect, L, H0, sim.DLinee, (0, 0, 0), thickness=2 if level == 0 else 1)

        # control panel
        geom_name = "Droplet (cone)" if sim.TypeOfPNG == 0 else "Flat"
        n1, n2 = sim.last_counts
        draw_text(screen, "Two independent Poisson clouds: weights 1 and √2", 12, 10, font2)
        y = 44
        draw_text(screen, f"{'RUNNING' if not sim.freeze else 'PAUSED'} (Space)    New cloud (N)", 12, y, font); y += 18
        draw_text(screen, f"Geometry: {geom_name} (2 droplet / 1 flat)   Resamples on resize", 12, y, font); y += 18
        draw_text(screen, f"Blue: w=1  λ1={sim.lam1:.4g} (A/S)   sampled N1={n1}", 12, y, font); y += 18
        draw_text(screen, f"Red : w=√2 λ2={sim.lam2:.4g} (K/L)   sampled N2={n2}", 12, y, font); y += 18
        draw_text(screen, f"Visible lines NLinee={sim.NLinee} ({{ / }})   spacing DLinee={sim.DLinee} (-/=)", 12, y, font); y += 18
        draw_text(screen, f"Speed={sim.Speed} FPS (Up/Down)", 12, y, font); y += 18
        draw_text(screen, "Lines use weighted partial cancellation: min(a,b) falls down, |a-b| stays.", 12, y, font)

        pygame.display.flip()
        clock.tick(sim.Speed)

    pygame.quit()


if __name__ == "__main__":
    main()