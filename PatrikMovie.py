"""
PNG / multi-line PNG animation (Python counterpart to the provided Java Swing applet)

Dependencies:
  python3 -m pip install pygame

Run:
  python3 png_animation.py

Controls:
  Space  : Start/Stop (toggle paused/running)
  R      : Toggle time direction (forward/backward)
  N      : New configuration of Poisson points (restart)
  M      : Toggle multilines vs single line display (only level 0 stepped)
  1      : Flat PNG
  2      : PNG droplet
  3      : PNG droplet with sources
  Up/Down: Speed +/- (FPS)
  [ / ]  : Number of points -/+ (NPunti)
  { / }  : Number of lines -/+ (NLinee)  (Shift+[ and Shift+])
  - / =  : Distance between lines (DLinee) -/+
  , / .  : Left sources (LSource) -/+
  K / L  : Right sources (RSource) -/+

Key fixes vs earlier draft:
- Droplet cone rejection now matches the *drawn* cone apex (y = height-2), reducing “outside cone” points.
- Default line spacing increased (DLinee) so lines aren’t cramped.
- ComputePoisson() is complete and correctly indented/connected.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pygame


# -----------------------------
# Data structures (like Java)
# -----------------------------

@dataclass
class Punto:
    x: int
    y: int
    next: Optional["Punto"] = None


class Linea:
    """
    Mimics Java's Linea with fixed-ish arrays pos/dir and linked-list pointers.
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

        # Parameters (mirroring Java, but with a better-looking default spacing)
        self.NPunti = 30
        self.Raggio = 3

        # Increased default so lines are not too close on modern screens
        self.DLinee = 12

        self.NLinee = 10
        self.Speed = 30  # FPS-ish
        self.delay_ms = int(1000 / max(1, self.Speed))

        self.LSource = 10
        self.RSource = 5

        self.TypeOfPNG = 1      # 0 droplet, 1 flat, 2 droplet with sources
        self.Multilinee = True  # show all lines stepped vs only level 0 stepped

        self.freeze = True
        self.DirTempo = True    # True forward, False backward

        self.Larghezza = 0
        self.Altezza = 0

        self.TotPunti = self.NPunti + self.LSource + self.RSource
        self.t = 0
        self.tOld = 0
        self.ComputedPoisson = False

        # Linked list of points
        self.PuntoIniziale = Punto(0, 0)
        self._build_point_list()

        # Linked list of lines
        self.LineaIniziale: Optional[Linea] = None
        self.LineaFinale: Optional[Linea] = None
        self.InitLines()

    def _build_point_list(self) -> None:
        self.PuntoIniziale.next = None
        p = self.PuntoIniziale
        for _ in range(1, self.TotPunti):
            p.next = Punto(0, 0)
            p = p.next

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

    def InitLines(self) -> None:
        cap = max(4, 2 * self.TotPunti)
        self.LineaIniziale = Linea(cap)
        self.LineaFinale = Linea(cap)

        # initialize arrays to sentinel
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

    def InitPoints(self) -> None:
        self._build_point_list()

    def recompute_totpunti_and_rebuild_points(self) -> None:
        if self.TypeOfPNG in (0, 1):
            self.TotPunti = self.NPunti
        else:
            self.TotPunti = self.NPunti + self.LSource + self.RSource

        self.InitPoints()
        self.InitLines()
        self.ComputedPoisson = False

    def riparti(self) -> None:
        self.DirTempo = True
        self.freeze = True
        self.t = 0
        self.tOld = 0
        self.InitLines()
        self.InitPoints()
        self.ComputePoisson()
        self.ComputedPoisson = True

    def ComputePoisson(self) -> None:
        """
        Recompute the Poisson point configuration according to TypeOfPNG.

        Fix applied: droplet rejection uses apex y = height-2 (matching the drawn rays),
        which reduces points that appear outside the visible cone.
        """
        width = self.Larghezza
        height = self.Altezza
        if width <= 0 or height <= 0:
            return

        p_list: List[Tuple[int, int]] = []
        k = 0

        p = self.PuntoIniziale
        while p is not None:
            while True:
                nuovo = True

                # Flat PNG
                if self.TypeOfPNG == 1:
                    p.x = 4 + int((width + 2 * height - 9) * random.random()) - height
                    p.y = 4 + int((height - 9) * random.random())

                # Droplet interior (or droplet-with-sources interior points)
                if (self.TypeOfPNG == 0) or (self.TypeOfPNG == 2 and k >= self.LSource + self.RSource):
                    p.x = 4 + int((width - 9) * random.random())
                    p.y = 4 + int((height - 9) * random.random())

                    # Match drawn rays: apex at y = height-2
                    if (height - 2 - p.y) < abs(p.x - width // 2):
                        nuovo = False

                # Droplet with sources: boundary points on the two rays
                if (self.TypeOfPNG == 2) and (k < self.LSource + self.RSource):
                    p.y = 4 + int((height - 9) * random.random())
                    if k < self.LSource:
                        p.x = (width // 2) - height + p.y + 2
                    else:
                        p.x = (width // 2) + height - p.y - 2

                # Parity adjustment like Java
                p.y = ((p.x + p.y) % 2) + p.y

                # Keep within margins after parity tweak (helps avoid edge artifacts)
                if p.y > height - 5:
                    p.y = height - 5

                # Uniqueness check
                if (p.x, p.y) in p_list and width * height > 0:
                    nuovo = False

                if nuovo:
                    break

            p_list.append((p.x, p.y))
            k += 1
            p = p.next

    def _add_nucleation_to_line(self, line: Linea, x: int, dir_pair=(1, -1)) -> None:
        line.ensure_capacity_for(2)
        k = line.Njumps
        line.dir[k] = dir_pair[0]
        line.pos[k] = x
        line.dir[k + 1] = dir_pair[1]
        line.pos[k + 1] = x
        line.Njumps += 2

    def ComputeAnimation(self) -> None:
        height = self.Altezza
        if height <= 0:
            return

        # Update time
        if not self.freeze:
            self.tOld = self.t
            if self.DirTempo:
                self.t = min(self.t + 1, height - 3)
            else:
                self.t = max(self.t - 1, 0)

        if self.freeze or (self.tOld == self.t):
            return

        # FORWARDS
        if self.DirTempo:
            # Phase 1: move jumps
            for L in self.iter_lines():
                for k in range(L.Njumps):
                    L.pos[k] = L.pos[k] - L.dir[k]

            # Phase 2: cancellations + nucleations below
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
                            # match Java ordering
                            LineaTemp.dir[m] = 1
                            LineaTemp.pos[m] = upper.pos[k + n]
                            LineaTemp.dir[m + 1] = -1
                            LineaTemp.pos[m + 1] = upper.pos[k]
                            LineaTemp.Njumps += 2

                LineaTemp = LineaTemp.prec

            # Phase 3: nucleations at level 0
            base = self.LineaIniziale
            if base is not None:
                for p in self.iter_points():
                    if (height - p.y - 2) == self.t:
                        self._add_nucleation_to_line(base, p.x, dir_pair=(1, -1))

        # BACKWARDS
        else:
            # Phase 1: cancellations that will mix; add nucleations to upper lines
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

            # Phase 2: move jumps back
            for L in self.iter_lines():
                for k in range(L.Njumps):
                    L.pos[k] = L.pos[k] + L.dir[k]

        # Delete canceled jumps (compact)
        for L in self.iter_lines():
            m = 0
            for k in range(L.Njumps):
                if L.dir[k] == 0:
                    m += 1
                else:
                    if L.dir[k] != 2:
                        L.pos[k - m] = L.pos[k]
                        L.dir[k - m] = L.dir[k]
                        if m > 0:
                            L.pos[k] = 0
                            L.dir[k] = 2
            L.Njumps -= L.NCancel
            L.NCancel = 0

        # Create extra line if last is excited
        if self.LineaFinale is not None and self.LineaFinale.Njumps > 0:
            old_last = self.LineaFinale
            new_last = Linea(max(4, 2 * self.TotPunti))
            old_last.next = new_last
            new_last.prec = old_last
            self.LineaFinale = new_last

        # Delete empty lines except last
        if self.LineaFinale is not None and self.LineaFinale.prec is not None:
            prev = self.LineaFinale.prec
            if prev.Njumps == 0 and prev is not self.LineaIniziale and prev.prec is not None:
                prevprec = prev.prec
                prevprec.next = self.LineaFinale
                self.LineaFinale.prec = prevprec

        # Reorder jumps by position; then swap mixed equal-position pairs
        for L in self.iter_lines():
            pairs = list(zip(L.pos[:L.Njumps], L.dir[:L.Njumps]))
            pairs.sort(key=lambda x: x[0])
            for i, (px, dr) in enumerate(pairs):
                L.pos[i] = px
                L.dir[i] = dr

            for k in range(L.Njumps):
                for n in range(k + 1, L.Njumps):
                    if L.pos[k] == L.pos[n] and L.dir[k] == -1 and L.dir[n] == 1:
                        L.dir[k] = 1
                        L.dir[n] = -1


# -----------------------------
# Pygame UI / drawing
# -----------------------------

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def draw_text(surface, text, x, y, font, color=(0, 0, 0)):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


def main():
    pygame.init()
    pygame.display.set_caption("PNG / multi-line PNG (pygame)")

    W, H = 1100, 750
    screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)

    font = pygame.font.SysFont(None, 18)
    font2 = pygame.font.SysFont(None, 22)

    clock = pygame.time.Clock()
    sim = PNGSim()

    running = True
    while running:
        # Handle resize
        W, H = screen.get_size()
        CONTROL_W = min(360, max(220, W // 4))
        ANIM_W = max(10, W - CONTROL_W)
        ANIM_H = H
        LINES_H = ANIM_H // 2
        POINTS_H = ANIM_H - LINES_H

        # Points panel dimensions (used for poisson and time line)
        sim.Larghezza = ANIM_W
        sim.Altezza = POINTS_H

        # Optional: keep spacing sensible if you resize very small/large
        sim.DLinee = clamp(sim.DLinee, 2, max(2, LINES_H // 2))

        if not sim.ComputedPoisson:
            sim.recompute_totpunti_and_rebuild_points()
            sim.ComputePoisson()
            sim.ComputedPoisson = True

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_SPACE:
                    sim.freeze = not sim.freeze

                elif event.key == pygame.K_r:
                    sim.DirTempo = not sim.DirTempo

                elif event.key == pygame.K_n:
                    sim.riparti()

                elif event.key == pygame.K_m:
                    sim.Multilinee = not sim.Multilinee

                elif event.key == pygame.K_1:
                    if sim.TypeOfPNG != 1:
                        sim.TypeOfPNG = 1
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_2:
                    if sim.TypeOfPNG != 0:
                        sim.TypeOfPNG = 0
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_3:
                    if sim.TypeOfPNG != 2:
                        sim.TypeOfPNG = 2
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_UP:
                    sim.Speed = clamp(sim.Speed + 5, 1, 240)
                    sim.delay_ms = int(1000 / max(1, sim.Speed))
                elif event.key == pygame.K_DOWN:
                    sim.Speed = clamp(sim.Speed - 5, 1, 240)
                    sim.delay_ms = int(1000 / max(1, sim.Speed))

                elif event.key == pygame.K_LEFTBRACKET:  # [
                    if mods & pygame.KMOD_SHIFT:
                        sim.NLinee = clamp(sim.NLinee - 1, 1, 200)
                    else:
                        sim.NPunti = clamp(sim.NPunti - 5, 1, 10000)
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_RIGHTBRACKET:  # ]
                    if mods & pygame.KMOD_SHIFT:
                        sim.NLinee = clamp(sim.NLinee + 1, 1, 200)
                    else:
                        sim.NPunti = clamp(sim.NPunti + 5, 1, 10000)
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_MINUS:
                    sim.DLinee = clamp(sim.DLinee - 1, 1, 80)
                elif event.key == pygame.K_EQUALS:
                    sim.DLinee = clamp(sim.DLinee + 1, 1, 80)

                elif event.key == pygame.K_COMMA:
                    sim.LSource = clamp(sim.LSource - 1, 0, 2000)
                    if sim.TypeOfPNG == 2:
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()
                elif event.key == pygame.K_PERIOD:
                    sim.LSource = clamp(sim.LSource + 1, 0, 2000)
                    if sim.TypeOfPNG == 2:
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

                elif event.key == pygame.K_k:
                    sim.RSource = clamp(sim.RSource - 1, 0, 2000)
                    if sim.TypeOfPNG == 2:
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()
                elif event.key == pygame.K_l:
                    sim.RSource = clamp(sim.RSource + 1, 0, 2000)
                    if sim.TypeOfPNG == 2:
                        sim.recompute_totpunti_and_rebuild_points()
                        sim.riparti()

        # Update simulation one tick per frame
        sim.ComputeAnimation()

        # Clear screen
        screen.fill((235, 235, 235))

        # Regions
        control_rect = pygame.Rect(0, 0, CONTROL_W, H)
        anim_rect = pygame.Rect(CONTROL_W, 0, ANIM_W, H)
        lines_rect = pygame.Rect(CONTROL_W, 0, ANIM_W, LINES_H)
        points_rect = pygame.Rect(CONTROL_W, LINES_H, ANIM_W, POINTS_H)

        pygame.draw.rect(screen, (245, 245, 245), control_rect)
        pygame.draw.rect(screen, (200, 200, 200), control_rect, 1)
        pygame.draw.rect(screen, (255, 255, 255), anim_rect)
        pygame.draw.rect(screen, (200, 200, 200), anim_rect, 1)
        pygame.draw.rect(screen, (220, 220, 220), lines_rect, 1)
        pygame.draw.rect(screen, (220, 220, 220), points_rect, 1)

        # -----------------------------
        # Draw lines (top panel)
        # -----------------------------
        if sim.LineaIniziale is not None:
            line_color0 = (210, 0, 0)   # red
            line_color = (0, 70, 200)   # blue

            H0_base = max(LINES_H // 2, LINES_H - sim.DLinee * sim.NLinee)
            L = sim.LineaIniziale
            for l in range(sim.NLinee):
                if L is None:
                    break
                color = line_color0 if l == 0 else line_color
                H0 = H0_base + l * sim.DLinee

                do_draw = sim.Multilinee or (l == 0)

                if not do_draw:
                    pygame.draw.line(
                        screen, color,
                        (lines_rect.left + 0, lines_rect.top + H0),
                        (lines_rect.left + ANIM_W - 1, lines_rect.top + H0),
                        2
                    )
                else:
                    if L.Njumps == 0:
                        pygame.draw.line(
                            screen, color,
                            (lines_rect.left + 0, lines_rect.top + H0),
                            (lines_rect.left + ANIM_W - 1, lines_rect.top + H0),
                            2
                        )
                    else:
                        n = 0
                        m = 0
                        x0 = clamp(L.pos[0], 0, ANIM_W - 1)
                        pygame.draw.line(
                            screen, color,
                            (lines_rect.left + 0, lines_rect.top + H0),
                            (lines_rect.left + x0, lines_rect.top + H0),
                            2
                        )

                        for k in range(L.Njumps - 1):
                            m = m + L.dir[k]
                            xk = clamp(L.pos[k], 0, ANIM_W - 1)
                            xk1 = clamp(L.pos[k + 1], 0, ANIM_W - 1)

                            y_from = H0 - n * sim.DLinee
                            y_to = H0 - m * sim.DLinee
                            pygame.draw.line(
                                screen, color,
                                (lines_rect.left + xk, lines_rect.top + y_from),
                                (lines_rect.left + xk, lines_rect.top + y_to),
                                2
                            )
                            pygame.draw.line(
                                screen, color,
                                (lines_rect.left + xk, lines_rect.top + y_to),
                                (lines_rect.left + xk1, lines_rect.top + y_to),
                                2
                            )
                            n = n + L.dir[k]

                        k = L.Njumps - 1
                        m = m + L.dir[k]
                        xk = clamp(L.pos[k], 0, ANIM_W - 1)
                        y_from = H0 - n * sim.DLinee
                        y_to = H0 - m * sim.DLinee
                        pygame.draw.line(
                            screen, color,
                            (lines_rect.left + xk, lines_rect.top + y_from),
                            (lines_rect.left + xk, lines_rect.top + y_to),
                            2
                        )
                        pygame.draw.line(
                            screen, color,
                            (lines_rect.left + xk, lines_rect.top + y_to),
                            (lines_rect.left + ANIM_W - 1, lines_rect.top + y_to),
                            2
                        )

                L = L.next

        # -----------------------------
        # Draw points + cone + time line (bottom panel)
        # -----------------------------
        cx = CONTROL_W + ANIM_W // 2
        bottom_y = lines_rect.bottom + POINTS_H - 2
        top_y = lines_rect.bottom + 1

        # Draw droplet cone for droplet geometries
        if sim.TypeOfPNG in (0, 2):
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx + POINTS_H - 3, top_y), 2)
            pygame.draw.line(screen, (0, 0, 0), (cx, bottom_y), (cx - POINTS_H + 3, top_y), 2)
        elif sim.TypeOfPNG == 1:
            # flat: faint cone guides (optional)
            pygame.draw.line(screen, (235, 235, 235), (cx, bottom_y), (cx + POINTS_H - 3, top_y), 1)
            pygame.draw.line(screen, (235, 235, 235), (cx, bottom_y), (cx - POINTS_H + 3, top_y), 1)

        # Points
        for p in sim.iter_points():
            pygame.draw.circle(
                screen, (0, 70, 200),
                (CONTROL_W + p.x, lines_rect.bottom + p.y),
                sim.Raggio
            )

        # Time line
        time_y = lines_rect.bottom + (POINTS_H - sim.t - 2)
        pygame.draw.line(
            screen, (0, 0, 0),
            (CONTROL_W + 0, time_y),
            (CONTROL_W + ANIM_W - 1, time_y),
            2
        )

        pygame.draw.line(
            screen, (160, 160, 160),
            (CONTROL_W, lines_rect.bottom),
            (CONTROL_W + ANIM_W, lines_rect.bottom),
            1
        )

        # -----------------------------
        # Control panel text
        # -----------------------------
        geom_name = {0: "Droplet", 1: "Flat", 2: "Droplet+Sources"}[sim.TypeOfPNG]
        draw_text(screen, "PNG / multi-line PNG (pygame)", 12, 10, font2, (0, 0, 0))

        y = 44
        draw_text(screen, f"Status: {'RUNNING' if not sim.freeze else 'PAUSED'} (Space)", 12, y, font); y += 20
        draw_text(screen, f"Time dir: {'FORWARD' if sim.DirTempo else 'BACKWARD'} (R)", 12, y, font); y += 20
        draw_text(screen, f"Geometry: {geom_name} (1/2/3)", 12, y, font); y += 20
        draw_text(screen, f"Multilines: {sim.Multilinee} (M)", 12, y, font); y += 20

        y += 10
        draw_text(screen, f"Speed: {sim.Speed} (Up/Down)", 12, y, font); y += 20
        draw_text(screen, f"NPunti: {sim.NPunti} ([ / ])", 12, y, font); y += 20
        draw_text(screen, f"NLinee: {sim.NLinee} ({{ / }})", 12, y, font); y += 20
        draw_text(screen, f"DLinee: {sim.DLinee} (- / =)", 12, y, font); y += 20

        y += 10
        draw_text(screen, f"LSource: {sim.LSource} (, / .)", 12, y, font); y += 20
        draw_text(screen, f"RSource: {sim.RSource} (K / L)", 12, y, font); y += 20

        y += 10
        draw_text(screen, "N: new points    Esc: quit", 12, y, font)

        pygame.display.flip()

        # frame cap
        clock.tick(sim.Speed)

        # Auto-stop at endpoints (like Java)
        if sim.t == POINTS_H - 3 and not sim.freeze:
            sim.freeze = True
            sim.DirTempo = False
        if sim.t == 0 and not sim.freeze:
            sim.freeze = True
            sim.DirTempo = True

    pygame.quit()


if __name__ == "__main__":
    main()