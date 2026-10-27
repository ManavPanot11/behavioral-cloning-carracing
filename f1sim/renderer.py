"""
f1sim/renderer.py - Broadcast-quality 1920x1080 Pygame Dashboard Renderer.

Features:
- Full 1080p broadcast canvas (1920x1080) with responsive DPI scaling (720p, 1080p, 1440p).
- Track Rendering with SINGLE authentic checkered start/finish line and staggered starting grid.
- Follow-camera tracking with smooth rotation or track-aligned view.
- Broadcast Timing Tower with position pills, driver codes, team badges, intervals, and sector status.
- Minimap with full track silhouette, sector boundaries, and live pulsing car blips.
- Live Telemetry Gauges: Speedometer, horizontal steering dial, throttle/brake LED bars, lateral Gs.
- Driver Cam Inset: Live 84x96 neural observation upscaled with team accent border.
- Floating Race Control notifications banner.
"""

import math
import numpy as np
import pygame

from f1sim.drivers import TEAMS


class DashboardRenderer:
    """
    Renders the broadcast telemetry dashboard at target resolution (default 1920x1080).
    """

    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.scale_factor = width / 1920.0

        if not pygame.get_init():
            pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        # Canvas surface
        self.surface = pygame.Surface((self.width, self.height))

        # Fonts (scalable)
        def get_font(size, bold=False):
            scaled_size = max(10, int(size * self.scale_factor))
            try:
                return pygame.font.SysFont("Segoe UI", scaled_size, bold=bold)
            except Exception:
                return pygame.font.SysFont(None, scaled_size, bold=bold)

        self.font_title = get_font(22, bold=True)
        self.font_header = get_font(18, bold=True)
        self.font_body = get_font(15, bold=False)
        self.font_bold = get_font(15, bold=True)
        self.font_small = get_font(12, bold=False)
        self.font_huge = get_font(44, bold=True)
        self.font_speed = get_font(56, bold=True)

        # Active camera target
        self.selected_car_id = None

    def _scale(self, val):
        return int(val * self.scale_factor)

    def render_frame(
        self,
        env,
        cars_state,
        leaderboard,
        selected_car_id,
        current_step=0,
        total_laps=10,
        race_events=None,
        active_cam_obs=None,
    ):
        """
        Main draw call.
        cars_state: dict mapping car_id -> {
            'pos': (x, y), 'heading': rad, 'speed': float, 'steer': float,
            'gas': float, 'brake': float, 'team_code': str, 'code': str,
            'offroad': bool, 'centre_dist': float, 'lateral_g': float
        }
        """
        self.selected_car_id = selected_car_id
        self.surface.fill((14, 18, 26))  # Deep broadcast dark backdrop

        uw = env.unwrapped
        track = getattr(uw, "track", [])

        # 1. World View (Follow camera)
        self._render_world(track, cars_state, selected_car_id)

        # 2. Timing Tower Overlay (Left side)
        self._render_timing_tower(leaderboard, total_laps)

        # 3. Minimap Overlay (Top right)
        self._render_minimap(track, cars_state)

        # 4. Driver Cam Inset (Middle right)
        if active_cam_obs is not None:
            self._render_driver_cam(active_cam_obs, cars_state.get(selected_car_id))

        # 5. Live Telemetry Gauges (Bottom right)
        self._render_telemetry_hud(cars_state.get(selected_car_id))

        # 6. Race Control Banner (Top center)
        if race_events:
            self._render_race_control_banner(race_events, current_step)

        return self.surface

    def _render_world(self, track, cars_state, selected_car_id):
        """Render the 2D circuit and cars centered on selected car."""
        target_car = cars_state.get(selected_car_id)
        if not target_car:
            target_pos = (0.0, 0.0)
            target_heading = 0.0
        else:
            target_pos = target_car["pos"]
            target_heading = target_car["heading"]

        center_x = self.width * 0.52
        center_y = self.height * 0.50
        zoom = 14.0 * self.scale_factor

        # Helper to transform world coords (x, y) to screen coords
        def world_to_screen(wx, wy):
            dx = wx - target_pos[0]
            dy = wy - target_pos[1]
            # Rotate with camera so car faces upward (F1 T-cam style)
            cos_a = math.cos(-target_heading - math.pi / 2)
            sin_a = math.sin(-target_heading - math.pi / 2)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            sx = int(center_x + rx * zoom)
            sy = int(center_y - ry * zoom)
            return sx, sy

        # Draw grass background texture
        pygame.draw.rect(self.surface, (30, 48, 32), (0, 0, self.width, self.height))

        # Draw Track tiles
        if len(track) > 1:
            # Draw tarmac polygons
            track_width_world = 6.67
            half_w = track_width_world / 2.0

            for i in range(len(track)):
                p1 = track[i]
                p2 = track[(i + 1) % len(track)]
                # Normal vector
                dx = p2[2] - p1[2]
                dy = p2[3] - p1[3]
                seg_len = math.hypot(dx, dy)
                if seg_len < 1e-4:
                    continue
                nx = -dy / seg_len * half_w
                ny = dx / seg_len * half_w

                # 4 corners of tarmac quad
                c1 = world_to_screen(p1[2] + nx, p1[3] + ny)
                c2 = world_to_screen(p1[2] - nx, p1[3] - ny)
                c3 = world_to_screen(p2[2] - nx, p2[3] - ny)
                c4 = world_to_screen(p2[2] + nx, p2[3] + ny)

                # Alternating asphalt shade
                shade = 42 if (i % 2 == 0) else 48
                pygame.draw.polygon(self.surface, (shade, shade, shade + 4), [c1, c2, c3, c4])

                # Red/White Kerbs on edges
                kerb_w = 0.8
                knx = -dy / seg_len * (half_w + kerb_w)
                kny = dx / seg_len * (half_w + kerb_w)
                kc1 = world_to_screen(p1[2] + knx, p1[3] + kny)
                kc4 = world_to_screen(p2[2] + knx, p2[3] + kny)
                kerb_color = (220, 20, 30) if (i % 4 < 2) else (240, 240, 240)
                pygame.draw.polygon(self.surface, kerb_color, [c1, kc1, kc4, c4])

            # Draw Start/Finish Line (SINGLE checkered strip at tile 0)
            p0 = track[0]
            p1 = track[1]
            dx = p1[2] - p0[2]
            dy = p1[3] - p0[3]
            seg_len = math.hypot(dx, dy)
            if seg_len > 1e-4:
                nx = -dy / seg_len * half_w
                ny = dx / seg_len * half_w
                tx = dx / seg_len * 1.5
                ty = dy / seg_len * 1.5

                # Render 6 checkered patches across width
                num_patches = 8
                for pi in range(num_patches):
                    frac1 = pi / num_patches
                    frac2 = (pi + 1) / num_patches
                    w1x = p0[2] + nx * (1 - 2 * frac1)
                    w1y = p0[3] + ny * (1 - 2 * frac1)
                    w2x = p0[2] + nx * (1 - 2 * frac2)
                    w2y = p0[3] + ny * (1 - 2 * frac2)

                    pt1 = world_to_screen(w1x, w1y)
                    pt2 = world_to_screen(w2x, w2y)
                    pt3 = world_to_screen(w2x + tx, w2y + ty)
                    pt4 = world_to_screen(w1x + tx, w1y + ty)

                    col = (255, 255, 255) if (pi % 2 == 0) else (15, 15, 15)
                    pygame.draw.polygon(self.surface, col, [pt1, pt2, pt3, pt4])

        # Draw Cars
        for car_id, cdata in cars_state.items():
            cx, cy = cdata["pos"]
            head = cdata["heading"]
            team_code = cdata.get("team_code", "APX")
            team = TEAMS.get(team_code, TEAMS["APX"])
            color = team["color"]

            scx, scy = world_to_screen(cx, cy)

            # Compute car rotation relative to camera
            rel_head = head - target_heading - math.pi / 2
            car_w = 2.4 * zoom
            car_l = 4.8 * zoom

            # Draw car body (F1 silhouette)
            car_surf = pygame.Surface((int(car_l * 1.5), int(car_l * 1.5)), pygame.SRCALPHA)
            mid = int(car_l * 0.75)

            # Main Chassis
            chassis_rect = pygame.Rect(mid - car_w * 0.35, mid - car_l * 0.45, car_w * 0.7, car_l * 0.9)
            pygame.draw.rect(car_surf, color, chassis_rect, border_radius=int(car_w * 0.2))

            # Cockpit / Halo (Dark)
            cockpit_rect = pygame.Rect(mid - car_w * 0.2, mid - car_l * 0.1, car_w * 0.4, car_l * 0.35)
            pygame.draw.rect(car_surf, (15, 20, 25), cockpit_rect, border_radius=int(car_w * 0.15))

            # Front & Rear Wings
            fw_rect = pygame.Rect(mid - car_w * 0.5, mid - car_l * 0.55, car_w, car_l * 0.15)
            rw_rect = pygame.Rect(mid - car_w * 0.5, mid + car_l * 0.38, car_w, car_l * 0.15)
            pygame.draw.rect(car_surf, (20, 20, 20), fw_rect, border_radius=2)
            pygame.draw.rect(car_surf, (20, 20, 20), rw_rect, border_radius=2)

            # Wheels (4 black rounded rectangles)
            wh_w = car_w * 0.25
            wh_l = car_l * 0.25
            for wx_sign in [-1, 1]:
                # Front wheels
                pygame.draw.rect(car_surf, (10, 10, 10), (mid + wx_sign * car_w * 0.42 - wh_w/2, mid - car_l * 0.38, wh_w, wh_l), border_radius=2)
                # Rear wheels
                pygame.draw.rect(car_surf, (10, 10, 10), (mid + wx_sign * car_w * 0.45 - wh_w/2, mid + car_l * 0.2, wh_w, wh_l), border_radius=2)

            # Rotate car surface
            rot_surf = pygame.transform.rotate(car_surf, math.degrees(-rel_head - math.pi / 2))
            rw_rect = rot_surf.get_rect(center=(scx, scy))
            self.surface.blit(rot_surf, rw_rect)

            # Overhead tag (Pill with code)
            code_text = cdata.get("code", "CAR")
            tag_surf = self.font_small.render(code_text, True, (255, 255, 255))
            tag_w, tag_h = tag_surf.get_size()
            pill_rect = pygame.Rect(scx - tag_w / 2 - 4, scy - car_l * 0.7 - tag_h - 2, tag_w + 8, tag_h + 4)
            pygame.draw.rect(self.surface, (10, 15, 25), pill_rect, border_radius=4)
            pygame.draw.rect(self.surface, color, pill_rect, width=1, border_radius=4)
            self.surface.blit(tag_surf, (scx - tag_w / 2, scy - car_l * 0.7 - tag_h))

    def _render_timing_tower(self, leaderboard, total_laps):
        """Render F1 Broadcast Timing Tower on left margin."""
        panel_w = self._scale(340)
        panel_h = self._scale(480)
        margin_x = self._scale(24)
        margin_y = self._scale(24)

        # Translucent glass panel
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((10, 15, 25, 215))
        pygame.draw.rect(panel_surf, (0, 229, 255, 80), (0, 0, panel_w, panel_h), width=1, border_radius=8)
        self.surface.blit(panel_surf, (margin_x, margin_y))

        # Title Header
        title = self.font_header.render("AUTONOMOUS GP", True, (255, 255, 255))
        self.surface.blit(title, (margin_x + self._scale(16), margin_y + self._scale(14)))

        # Subtitle / Lap counter
        lap_info = self.font_small.render(f"LEADERBOARD · {total_laps} LAPS", True, (0, 229, 255))
        self.surface.blit(lap_info, (margin_x + self._scale(16), margin_y + self._scale(38)))

        # Divider line
        pygame.draw.line(
            self.surface, (40, 50, 70),
            (margin_x + self._scale(16), margin_y + self._scale(56)),
            (margin_x + panel_w - self._scale(16), margin_y + self._scale(56)),
            1
        )

        # Leaderboard Rows
        row_y = margin_y + self._scale(66)
        row_h = self._scale(72)

        for entry in leaderboard:
            is_selected = (entry["car_id"] == self.selected_car_id)
            team = TEAMS.get(entry["team_code"], TEAMS["APX"])

            # Row background
            row_rect = pygame.Rect(margin_x + self._scale(10), row_y, panel_w - self._scale(20), row_h - self._scale(6))
            if is_selected:
                pygame.draw.rect(self.surface, (20, 35, 55, 230), row_rect, border_radius=6)
                pygame.draw.rect(self.surface, team["color"], row_rect, width=2, border_radius=6)
            else:
                pygame.draw.rect(self.surface, (16, 22, 34, 180), row_rect, border_radius=6)

            # Team Accent Bar
            pygame.draw.rect(
                self.surface, team["color"],
                (row_rect.x, row_rect.y, self._scale(5), row_rect.height),
                border_top_left_radius=6, border_bottom_left_radius=6
            )

            # Pos Pill
            pos_text = self.font_bold.render(str(entry["pos"]), True, (255, 255, 255))
            self.surface.blit(pos_text, (row_rect.x + self._scale(14), row_rect.y + self._scale(10)))

            # Driver Code
            code_text = self.font_bold.render(entry["code"], True, (255, 255, 255))
            self.surface.blit(code_text, (row_rect.x + self._scale(38), row_rect.y + self._scale(10)))

            # Gap / Interval
            gap_col = (0, 229, 255) if entry["pos"] == 1 else (200, 210, 225)
            gap_text = self.font_small.render(entry["gap"], True, gap_col)
            self.surface.blit(gap_text, (row_rect.right - self._scale(80), row_rect.y + self._scale(12)))

            # Driver Name & Team
            name_text = self.font_small.render(entry["name"], True, (160, 175, 195))
            self.surface.blit(name_text, (row_rect.x + self._scale(14), row_rect.y + self._scale(34)))

            # Sector Pills (S1, S2, S3)
            sec_x = row_rect.right - self._scale(78)
            sec_y = row_rect.y + self._scale(34)
            for s_idx, sec_key in enumerate(["s1", "s2", "s3"]):
                sec_obj = entry.get(sec_key)
                col = sec_obj.color if sec_obj else (45, 55, 70)
                pill = pygame.Rect(sec_x + s_idx * self._scale(24), sec_y, self._scale(20), self._scale(12))
                pygame.draw.rect(self.surface, col, pill, border_radius=3)

            row_y += row_h

    def _render_minimap(self, track, cars_state):
        """Render Circuit Minimap at top right."""
        box_w = self._scale(280)
        box_h = self._scale(220)
        pos_x = self.width - box_w - self._scale(24)
        pos_y = self._scale(24)

        panel_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel_surf.fill((10, 15, 25, 215))
        pygame.draw.rect(panel_surf, (0, 229, 255, 80), (0, 0, box_w, box_h), width=1, border_radius=8)
        self.surface.blit(panel_surf, (pos_x, pos_y))

        # Title
        m_title = self.font_small.render("CIRCUIT MAP", True, (0, 229, 255))
        self.surface.blit(m_title, (pos_x + self._scale(14), pos_y + self._scale(10)))

        if len(track) < 2:
            return

        # Find bounds of track
        nodes = np.array([[p[2], p[3]] for p in track])
        min_x, min_y = np.min(nodes, axis=0)
        max_x, max_y = np.max(nodes, axis=0)
        span_x = max(max_x - min_x, 1e-4)
        span_y = max(max_y - min_y, 1e-4)

        pad = self._scale(30)
        draw_w = box_w - 2 * pad
        draw_h = box_h - 2 * pad - self._scale(15)
        scale = min(draw_w / span_x, draw_h / span_y)

        def norm_pt(wx, wy):
            nx = pos_x + pad + (wx - min_x) * scale
            ny = pos_y + pad + self._scale(15) + (max_y - wy) * scale
            return int(nx), int(ny)

        # Draw track silhouette
        pts = [norm_pt(p[2], p[3]) for p in track]
        pygame.draw.lines(self.surface, (50, 65, 85), True, pts, width=3)

        # Start/Finish tick mark
        s0 = norm_pt(track[0][2], track[0][3])
        pygame.draw.circle(self.surface, (255, 255, 255), s0, 4)

        # Draw car positions
        for car_id, cdata in cars_state.items():
            cx, cy = cdata["pos"]
            team = TEAMS.get(cdata.get("team_code", "APX"), TEAMS["APX"])
            m_pos = norm_pt(cx, cy)
            is_sel = (car_id == self.selected_car_id)
            if is_sel:
                pygame.draw.circle(self.surface, (255, 255, 255), m_pos, 7, width=2)
            pygame.draw.circle(self.surface, team["color"], m_pos, 5)

    def _render_driver_cam(self, obs, car_data):
        """Render the 84x96 neural camera observation inset on right side."""
        box_w = self._scale(280)
        box_h = self._scale(210)
        pos_x = self.width - box_w - self._scale(24)
        pos_y = self._scale(260)

        panel_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel_surf.fill((10, 15, 25, 215))
        pygame.draw.rect(panel_surf, (0, 229, 255, 80), (0, 0, box_w, box_h), width=1, border_radius=8)
        self.surface.blit(panel_surf, (pos_x, pos_y))

        # Title
        team = TEAMS.get(car_data.get("team_code", "APX"), TEAMS["APX"]) if car_data else TEAMS["APX"]
        c_title = self.font_small.render(f"ON-BOARD · {team['driver'].upper()}", True, team["color"])
        self.surface.blit(c_title, (pos_x + self._scale(14), pos_y + self._scale(10)))

        # Upscale 84x96 crop to fit inner box
        crop = obs[:84, :, :]
        # Convert numpy RGB to Pygame surface (96 wide, 84 high)
        cam_surf = pygame.surfarray.make_surface(np.transpose(crop, (1, 0, 2)))
        scaled_cam = pygame.transform.scale(cam_surf, (box_w - self._scale(28), box_h - self._scale(48)))
        self.surface.blit(scaled_cam, (pos_x + self._scale(14), pos_y + self._scale(34)))

    def _render_telemetry_hud(self, car_data):
        """Render Live Telemetry Gauges (Speedometer, Steer, Throttle, Brake) at bottom right."""
        box_w = self._scale(520)
        box_h = self._scale(220)
        pos_x = self.width - box_w - self._scale(24)
        pos_y = self.height - box_h - self._scale(24)

        panel_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        panel_surf.fill((10, 15, 25, 225))
        pygame.draw.rect(panel_surf, (0, 229, 255, 80), (0, 0, box_w, box_h), width=1, border_radius=8)
        self.surface.blit(panel_surf, (pos_x, pos_y))

        speed = car_data.get("speed", 0.0) if car_data else 0.0
        steer = car_data.get("steer", 0.0) if car_data else 0.0
        gas = car_data.get("gas", 0.0) if car_data else 0.0
        brake = car_data.get("brake", 0.0) if car_data else 0.0
        cdist = car_data.get("centre_dist", 0.0) if car_data else 0.0
        lat_g = car_data.get("lateral_g", 0.0) if car_data else 0.0

        # Speed Readout
        speed_kmh = int(speed * 1.6)
        spd_text = self.font_speed.render(f"{speed_kmh:3d}", True, (255, 255, 255))
        self.surface.blit(spd_text, (pos_x + self._scale(24), pos_y + self._scale(20)))

        kmh_label = self.font_small.render("KM / H", True, (0, 229, 255))
        self.surface.blit(kmh_label, (pos_x + self._scale(32), pos_y + self._scale(82)))

        # Throttle & Brake Pedals (Vertical Bars)
        bar_x = pos_x + self._scale(160)
        bar_y = pos_y + self._scale(30)
        bar_w = self._scale(22)
        bar_h = self._scale(120)

        # Throttle (Green)
        pygame.draw.rect(self.surface, (25, 40, 30), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
        t_fill = int(bar_h * min(1.0, max(0.0, gas)))
        if t_fill > 0:
            pygame.draw.rect(self.surface, (0, 230, 118), (bar_x, bar_y + bar_h - t_fill, bar_w, t_fill), border_radius=4)
        t_lbl = self.font_small.render("THR", True, (0, 230, 118))
        self.surface.blit(t_lbl, (bar_x, bar_y + bar_h + self._scale(6)))

        # Brake (Red)
        b_x = bar_x + self._scale(32)
        pygame.draw.rect(self.surface, (45, 20, 25), (b_x, bar_y, bar_w, bar_h), border_radius=4)
        b_fill = int(bar_h * min(1.0, max(0.0, brake)))
        if b_fill > 0:
            pygame.draw.rect(self.surface, (255, 23, 68), (b_x, bar_y + bar_h - b_fill, bar_w, b_fill), border_radius=4)
        b_lbl = self.font_small.render("BRK", True, (255, 23, 68))
        self.surface.blit(b_lbl, (b_x, bar_y + bar_h + self._scale(6)))

        # Steering Horizontal Dial
        st_x = pos_x + self._scale(250)
        st_y = pos_y + self._scale(50)
        st_w = self._scale(230)
        st_h = self._scale(14)
        pygame.draw.rect(self.surface, (30, 40, 55), (st_x, st_y, st_w, st_h), border_radius=7)
        pygame.draw.line(self.surface, (100, 120, 150), (st_x + st_w / 2, st_y - 4), (st_x + st_w / 2, st_y + st_h + 4), 2)

        # Steering Marker
        clamped_steer = min(1.0, max(-1.0, steer))
        marker_x = int(st_x + st_w / 2 + clamped_steer * (st_w / 2 - 8))
        pygame.draw.circle(self.surface, (0, 229, 255), (marker_x, int(st_y + st_h / 2)), self._scale(9))

        st_lbl = self.font_small.render(f"STEERING: {steer:+.2f}", True, (200, 215, 230))
        self.surface.blit(st_lbl, (st_x, st_y - self._scale(22)))

        # Metrics Readouts: Centre-line & Lateral G
        cd_text = self.font_small.render(f"CENTRE DIST: {cdist:.2f} m", True, (160, 180, 205))
        self.surface.blit(cd_text, (st_x, st_y + self._scale(40)))

        g_text = self.font_small.render(f"LATERAL G: {lat_g:.2f} G", True, (160, 180, 205))
        self.surface.blit(g_text, (st_x, st_y + self._scale(62)))

        autonomy_text = self.font_bold.render("AUTONOMY: 100.0%", True, (0, 230, 118))
        self.surface.blit(autonomy_text, (st_x, st_y + self._scale(88)))

    def _render_race_control_banner(self, race_events, current_step):
        """Render Floating Race Control Notice at top center."""
        if not race_events:
            return
        latest = race_events[-1]
        msg = latest.message
        level = latest.level

        color = (0, 229, 255)
        if level == "PENALTY" or level == "DNF":
            color = (255, 23, 68)
        elif level == "WARNING" or level == "FLAG":
            color = (255, 214, 0)

        banner_text = self.font_bold.render(f"RACE CONTROL: {msg}", True, (255, 255, 255))
        bw, bh = banner_text.get_size()
        bx = (self.width - bw) // 2
        by = self._scale(20)

        pill_rect = pygame.Rect(bx - self._scale(16), by - self._scale(6), bw + self._scale(32), bh + self._scale(12))
        pygame.draw.rect(self.surface, (10, 15, 25, 240), pill_rect, border_radius=8)
        pygame.draw.rect(self.surface, color, pill_rect, width=2, border_radius=8)
        self.surface.blit(banner_text, (bx, by))
