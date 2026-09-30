"""Top-down Pygame renderer with a broadcast-style scoreboard.

Blue alliance wall on the left, red on the right, field +y pointing up the screen.
``Viewer(headless=True)`` renders off-screen (for ``render_mode="rgb_array"``).
With the high-fidelity physics it also draws the bumper rectangles, the swerve modules'
wheel directions, the HUB openings, and a thin outline where the highlighted robot
believes it is (its pose estimate).
"""

from __future__ import annotations

import math
import os

import numpy as np

from . import constants as C
from .ballistics import APOTHEM
from .constants import Alliance
from .drivetrain import module_positions
from .field import FIELD
from .robot import ClimbState
from .rules import Period, display_clock, period_bounds

SCALE = 78  # pixels per meter
MARGIN = 18
HUD = 96
BLUE = (40, 110, 230)
RED = (225, 45, 55)
BLUE_DIM = (28, 48, 84)
RED_DIM = (84, 30, 34)
FUEL_COLOR = (250, 215, 30)
CARPET = (58, 60, 66)
LINE = (210, 210, 215)
TEXT = (235, 235, 240)
PANEL = (22, 23, 28)

_PERIOD_LABEL = {
    Period.AUTO: "AUTO", Period.PAUSE: "AUTO over", Period.TRANSITION: "TRANSITION", Period.SHIFT1: "SHIFT 1",
    Period.SHIFT2: "SHIFT 2", Period.SHIFT3: "SHIFT 3", Period.SHIFT4: "SHIFT 4", Period.ENDGAME: "END GAME",
    Period.POST: "MATCH OVER", Period.DONE: "FINAL",
}
_TELEOP_INDEX = {Period.TRANSITION: 1, Period.SHIFT1: 2, Period.SHIFT2: 3, Period.SHIFT3: 4, Period.SHIFT4: 5,
                 Period.ENDGAME: 6}


class Viewer:
    def __init__(self, headless: bool = False, title: str = "REBUILT simulator") -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame

        self.pg = pygame
        pygame.init()
        self.w = int(C.FIELD_LENGTH * SCALE) + 2 * MARGIN
        self.h = int(C.FIELD_WIDTH * SCALE) + 2 * MARGIN + HUD
        self.headless = headless
        if headless:
            self.screen = pygame.Surface((self.w, self.h))
        else:
            self.screen = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("arial", 16, bold=True)
        self.small = pygame.font.SysFont("arial", 12, bold=True)
        self.big = pygame.font.SysFont("arial", 34, bold=True)
        self.clock = pygame.time.Clock()
        self._field = self._draw_static()

    # ---------------------------------------------------------------- coordinates
    def px(self, x: float, y: float) -> tuple[int, int]:
        return int(MARGIN + x * SCALE), int(HUD + MARGIN + (C.FIELD_WIDTH - y) * SCALE)

    def rect(self, box) -> "pygame.Rect":
        x0, y1 = self.px(box.x0, box.y0)
        x1, y0 = self.px(box.x1, box.y1)
        return self.pg.Rect(x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    # ---------------------------------------------------------------- static field
    def _draw_static(self):
        pg = self.pg
        surf = pg.Surface((self.w, self.h))
        surf.fill(PANEL)
        pg.draw.rect(surf, CARPET, pg.Rect(MARGIN, HUD + MARGIN, int(C.FIELD_LENGTH * SCALE), int(C.FIELD_WIDTH * SCALE)))
        for a, tint in ((Alliance.BLUE, (48, 58, 82)), (Alliance.RED, (82, 52, 56))):
            pg.draw.rect(surf, tint, self.rect(FIELD.alliance_zones[a]))
        cx0, cy0 = self.px(C.CENTER_X, 0)
        cx1, cy1 = self.px(C.CENTER_X, C.FIELD_WIDTH)
        pg.draw.line(surf, LINE, (cx0, cy0), (cx1, cy1), 2)
        for a, col in ((Alliance.BLUE, BLUE), (Alliance.RED, RED)):
            for pair in FIELD.bumps:
                r = self.rect(pair[a])
                pg.draw.rect(surf, tuple(int(c * 0.55) for c in col), r)
                pg.draw.rect(surf, col, r, 2)
            for pair in FIELD.trench_openings:
                r = self.rect(pair[a])
                pg.draw.rect(surf, (90, 92, 98), r)
                for k in range(r.left, r.right, 10):
                    pg.draw.line(surf, (200, 180, 40), (k, r.top), (min(k + 6, r.right), r.top + 6), 2)
                pg.draw.rect(surf, (200, 180, 40), r, 1)
            for pair in FIELD.trench_columns:
                pg.draw.rect(surf, (30, 30, 34), self.rect(pair[a]))
            tower = self.rect(FIELD.towers[a])
            pg.draw.rect(surf, (35, 35, 40), tower)
            for k in range(3):
                y = tower.top + (k + 1) * tower.height // 4
                pg.draw.line(surf, col, (tower.left + 3, y), (tower.right - 3, y), 3)
            pg.draw.rect(surf, col, tower, 2)
            pg.draw.rect(surf, col, self.rect(FIELD.depots[a]), 2)
            oy = C.OUTPOST_CENTER_Y_BLUE if a == Alliance.BLUE else C.FIELD_WIDTH - C.OUTPOST_CENTER_Y_BLUE
            ox = 0.0 if a == Alliance.BLUE else C.FIELD_LENGTH
            p0 = self.px(ox, oy - C.OUTPOST_FEED_HALF_WIDTH)
            p1 = self.px(ox, oy + C.OUTPOST_FEED_HALF_WIDTH)
            pg.draw.line(surf, col, p0, p1, 8)
        return surf

    # ---------------------------------------------------------------- per frame
    def draw(self, match, highlight: int | None = None, return_array: bool = False, message: str | None = None):
        pg = self.pg
        if not self.headless:
            for event in pg.event.get(pg.QUIT):
                self.close()
                raise SystemExit
        s = self.screen
        s.blit(self._field, (0, 0))
        self._draw_hubs(match)
        g = match.ground_fuel()
        rad = max(3, int(C.FUEL_RADIUS * SCALE))
        for x, y in g:
            pg.draw.circle(s, FUEL_COLOR, self.px(x, y), rad)
        for x, y, h in match.fuel_in_flight():
            pg.draw.circle(s, (255, 240, 120), self.px(x, y), int(rad * (1 + 1.2 * h)))
            pg.draw.circle(s, (120, 100, 20), self.px(x, y), int(rad * (1 + 1.2 * h)), 1)
        for a in Alliance:
            ox = 0.35 if a == Alliance.BLUE else C.FIELD_LENGTH - 0.35
            oy = C.OUTPOST_CENTER_Y_BLUE if a == Alliance.BLUE else C.FIELD_WIDTH - C.OUTPOST_CENTER_Y_BLUE
            lbl = self.small.render(str(match.chute_count[a]), True, FUEL_COLOR)
            s.blit(lbl, lbl.get_rect(center=self.px(ox, oy)))
        for r in match.robots:
            self._draw_robot(r, r.index == highlight, match)
        if highlight is not None and match.sensors is not None:  # where the robot believes it is
            x, y, h, _, _ = match.perceived(highlight)
            r = match.robots[highlight]
            pg.draw.polygon(s, (255, 255, 255), self._footprint(r, x, y, h), 1)
        self._draw_hud(match)
        if message:
            lbl = self.font.render(message, True, TEXT)
            s.blit(lbl, lbl.get_rect(midbottom=(self.w // 2, self.h - 4)))
        if return_array:
            return np.transpose(pg.surfarray.array3d(s), (1, 0, 2))
        if not self.headless:
            pg.display.flip()
        return None

    def _draw_hubs(self, m):
        pg = self.pg
        t = m.t
        for a, col, dim in ((Alliance.BLUE, BLUE, BLUE_DIM), (Alliance.RED, RED, RED_DIM)):
            r = self.rect(FIELD.hubs[a])
            active = m.hub_active(a)
            ttg = m.time_to_toggle(a)
            warn = active and ttg is not None and ttg <= C.GRACE
            color = col if active else dim
            if warn and int(t * 4) % 2 == 0:  # 3 s warning before the HUB switches off
                color = (240, 240, 240)
            s = self.screen
            pg.draw.rect(s, (30, 30, 34), r)
            pg.draw.rect(s, color, r.inflate(-10, -10), 0 if active else 3)
            pg.draw.rect(s, color, r, 3)
            if m.shooters is not None:  # the hexagonal opening FUEL has to drop through
                cx, cy = FIELD.hub_centers[a]
                corner = APOTHEM / math.cos(math.pi / 6)
                pts = [self.px(cx + corner * math.cos(math.pi / 6 + k * math.pi / 3),
                               cy + corner * math.sin(math.pi / 6 + k * math.pi / 3)) for k in range(6)]
                pg.draw.polygon(s, (30, 30, 34), pts, 2)

    def _footprint(self, r, x: float, y: float, heading: float) -> list[tuple[int, int]]:
        """Screen corners of a robot's bumpers at pose (x, y, heading): front-left, front-right, ..."""
        if r.rect:
            hl, hw = r.spec.length / 2, r.spec.width / 2
        else:
            hl = hw = r.spec.radius * 0.93
        ch, sh = math.cos(heading), math.sin(heading)
        return [self.px(x + dx * hl * ch - dy * hw * sh, y + dx * hl * sh + dy * hw * ch)
                for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1))]

    def _draw_robot(self, r, highlight: bool, match=None):
        pg = self.pg
        s = self.screen
        col = BLUE if r.alliance == Alliance.BLUE else RED
        half = r.spec.radius * SCALE * 0.93
        corners = self._footprint(r, r.x, r.y, r.heading)
        body = (70, 72, 78) if r.mobile or r.climb_state != ClimbState.GROUND else (40, 40, 44)
        pg.draw.polygon(s, body, corners)
        pg.draw.polygon(s, col, corners, 5)
        # intake edge
        pg.draw.line(s, FUEL_COLOR, corners[0], corners[1], 3)
        if match is not None and match.drive is not None:  # swerve modules: which way each wheel points
            ch, sh = math.cos(r.heading), math.sin(r.heading)
            for (mx, my), a in zip(module_positions(r.spec), match.drive.angle[r.index]):
                wx, wy = r.x + mx * ch - my * sh, r.y + mx * sh + my * ch
                dx, dy = 0.07 * math.cos(r.heading + a), 0.07 * math.sin(r.heading + a)
                pg.draw.line(s, (20, 20, 22), self.px(wx - dx, wy - dy), self.px(wx + dx, wy + dy), 4)
        if highlight:
            pg.draw.circle(s, (255, 255, 255), self.px(r.x, r.y), int(half * 1.45), 2)
        label = self.small.render(f"{r.name[0].upper()}{r.slot + 1}:{r.fuel}", True, TEXT)
        s.blit(label, label.get_rect(center=self.px(r.x, r.y)))
        if r.climb_state != ClimbState.GROUND:
            tag = {ClimbState.CLIMBING: "climbing", ClimbState.CLIMBED: f"L{r.climb_level}",
                   ClimbState.DESCENDING: "down"}[r.climb_state]
            lbl = self.small.render(tag, True, (255, 255, 255))
            s.blit(lbl, lbl.get_rect(midbottom=self.px(r.x, r.y + r.spec.radius + 0.15)))
        if r.jam_timer > 0:
            lbl = self.small.render("JAM", True, (255, 150, 60))
            s.blit(lbl, lbl.get_rect(midtop=self.px(r.x, r.y - r.spec.radius - 0.05)))

    def _draw_hud(self, m):
        pg = self.pg
        s = self.screen
        pg.draw.rect(s, PANEL, pg.Rect(0, 0, self.w, HUD))
        mid = self.w // 2
        blue, red = m.scores
        # scores
        pg.draw.rect(s, BLUE, pg.Rect(mid - 250, 12, 160, 56))
        pg.draw.rect(s, RED, pg.Rect(mid + 90, 12, 160, 56))
        for val, x in ((blue.total, mid - 170), (red.total, mid + 170)):
            lbl = self.big.render(str(val), True, TEXT)
            s.blit(lbl, lbl.get_rect(center=(x, 40)))
        # clock
        pg.draw.rect(s, (245, 245, 245), pg.Rect(mid - 88, 12, 176, 56))
        secs = int(math.ceil(display_clock(m.t) - 1e-9))
        lbl = self.big.render(f"{secs // 60}:{secs % 60:02d}", True, (20, 20, 20))
        s.blit(lbl, lbl.get_rect(center=(mid, 36)))
        p = m.period
        sub = _PERIOD_LABEL[p]
        if p in _TELEOP_INDEX:
            left = period_bounds(p)[1] - m.t
            sub = f"{_TELEOP_INDEX[p]}/6 :{int(math.ceil(left)):02d}  {sub}"
        lbl = self.small.render(sub, True, (20, 20, 20))
        s.blit(lbl, lbl.get_rect(center=(mid, 60)))
        # FUEL counters toward the ranking points, and active-HUB arrows
        for a, x, col in ((Alliance.BLUE, 110, BLUE), (Alliance.RED, self.w - 110, RED)):
            sc = m.scores[a]
            target = 100 if sc.fuel < 100 else 360
            pg.draw.rect(s, col, pg.Rect(x - 80, 14, 160, 30))
            lbl = self.font.render(f"FUEL {sc.fuel} / {target}", True, TEXT)
            s.blit(lbl, lbl.get_rect(center=(x, 29)))
            info = f"tower {sc.tower}  fouls rcvd {sc.penalty_points}"
            lbl = self.small.render(info, True, TEXT)
            s.blit(lbl, lbl.get_rect(center=(x, 58)))
            if m.hub_active(a):  # like the broadcast: a yellow arrow beside an alliance whose HUB counts
                d = -1 if a == Alliance.BLUE else 1
                ax = x - d * 100
                pg.draw.polygon(s, (250, 230, 40), [(ax + d * 12, 29), (ax - d * 8, 16), (ax - d * 8, 42)])
        first = m.schedule.first_inactive
        if first is not None and m.t < C.ENDGAME_START:
            who = "BLUE" if first == Alliance.BLUE else "RED"
            lbl = self.small.render(f"{who} won AUTO: its HUB is off in SHIFTS 1 and 3", True, TEXT)
            s.blit(lbl, lbl.get_rect(center=(mid, 84)))
        if m.done and m.rp:
            rp = m.rp
            lbl = self.small.render(
                f"FINAL  blue {blue.total} ({rp[0].total} RP)  -  red {red.total} ({rp[1].total} RP)", True, TEXT)
            s.blit(lbl, lbl.get_rect(center=(mid, 84)))

    def tick(self, fps: float) -> None:
        self.clock.tick(fps)

    def close(self) -> None:
        self.pg.quit()
