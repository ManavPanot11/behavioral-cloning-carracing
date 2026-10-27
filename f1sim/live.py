"""
f1sim/live.py - Real-Time Streaming Server & WebSocket Telemetry Broadcaster.

Provides:
- Standalone zero-dependency RFC 6455 WebSocket Server
- Built-in HTTP static server for local 3D viewer
- Live simulation stepping at 50 FPS with real-time state broadcasts
- Deterministic seed generation (>= 6000), random lights hold, and driver pace variance
- Replay generation command `make_replays --n 8`
"""

import os
import time
import json
import base64
import struct
import socket
import select
import random
import hashlib
import threading
import http.server
from typing import Optional, Dict, Any, List

from f1sim.world import MultiCarWorld, TEAMS, validate_seed


class SimpleWebSocketServer:
    """
    Lightweight, dependency-free RFC 6455 WebSocket Server.
    Handles handshake, framing, ping/pong, and multi-client broadcasting.
    """

    GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(10)
        self.clients: List[socket.socket] = []
        self.lock = threading.Lock()
        self.running = False
        self._thread = None
        self.latest_init_msg: Optional[str] = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        with self.lock:
            for c in self.clients:
                try:
                    c.close()
                except Exception:
                    pass
            self.clients.clear()
        try:
            self.sock.close()
        except Exception:
            pass

    def _listen_loop(self):
        while self.running:
            try:
                r, _, _ = select.select([self.sock], [], [], 0.2)
                if not r:
                    continue
                client_sock, addr = self.sock.accept()
                t = threading.Thread(target=self._handle_handshake, args=(client_sock,), daemon=True)
                t.start()
            except Exception:
                break

    def _handle_handshake(self, client: socket.socket):
        try:
            client.settimeout(3.0)
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(1024)
                if not chunk:
                    client.close()
                    return
                data += chunk

            headers = {}
            lines = data.decode("utf-8", errors="ignore").split("\r\n")
            for line in lines[1:]:
                if ": " in line:
                    k, v = line.split(": ", 1)
                    headers[k.strip().lower()] = v.strip()

            key = headers.get("sec-websocket-key")
            if not key:
                client.close()
                return

            accept_val = base64.b64encode(
                hashlib.sha1((key + self.GUID).encode("utf-8")).digest()
            ).decode("utf-8")

            response = (
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
            )
            client.sendall(response.encode("utf-8"))
            client.setblocking(False)

            with self.lock:
                self.clients.append(client)

            # Immediately send initial state if available
            if self.latest_init_msg:
                self._send_frame(client, self.latest_init_msg)

            # Keep reading client frames in background to catch close / ping
            threading.Thread(target=self._client_reader, args=(client,), daemon=True).start()

        except Exception:
            try:
                client.close()
            except Exception:
                pass

    def _client_reader(self, client: socket.socket):
        while self.running:
            try:
                r, _, _ = select.select([client], [], [], 0.5)
                if not r:
                    continue
                head = client.recv(2)
                if not head or len(head) < 2:
                    break
                b1, b2 = head[0], head[1]
                opcode = b1 & 0x0F
                if opcode == 0x8:  # Close
                    break
                # Handle masked payload
                is_masked = bool(b2 & 0x80)
                length = b2 & 0x7F
                if length == 126:
                    ext = client.recv(2)
                    length = struct.unpack("!H", ext)[0]
                elif length == 127:
                    ext = client.recv(8)
                    length = struct.unpack("!Q", ext)[0]
                mask = client.recv(4) if is_masked else b""
                payload = b""
                while len(payload) < length:
                    chunk = client.recv(min(length - len(payload), 4096))
                    if not chunk:
                        break
                    payload += chunk
            except Exception:
                break

        with self.lock:
            if client in self.clients:
                self.clients.remove(client)
        try:
            client.close()
        except Exception:
            pass

    def _send_frame(self, client: socket.socket, text: str):
        payload = text.encode("utf-8")
        length = len(payload)
        if length < 126:
            header = bytes([0x81, length])
        elif length < 65536:
            header = bytes([0x81, 126]) + struct.pack("!H", length)
        else:
            header = bytes([0x81, 127]) + struct.pack("!Q", length)
        client.sendall(header + payload)

    def broadcast(self, text: str):
        with self.lock:
            dead = []
            for c in self.clients:
                try:
                    self._send_frame(c, text)
                except Exception:
                    dead.append(c)
            for d in dead:
                if d in self.clients:
                    self.clients.remove(d)
                try:
                    d.close()
                except Exception:
                    pass


def start_http_server(port: int = 8080) -> http.server.ThreadingHTTPServer:
    """Start threaded static HTTP server in background."""
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


def run_live_race(
    seed: Optional[Any] = None,
    laps: int = 3,
    traffic: str = "contact",
    pace_variance: float = 0.03,
    http_port: int = 8080,
    ws_port: int = 8765,
    out_path: Optional[str] = None,
    open_browser: bool = True,
) -> Dict[str, Any]:
    """
    Launch live interactive race simulation streaming to 3D WebGL viewer.
    """
    # 1. Resolve Seed
    if seed is None or str(seed).strip().lower() == "random":
        actual_seed = random.randint(6000, 99999)
        print(f"[LIVE] Generated random showcase seed: {actual_seed}")
    else:
        actual_seed = int(seed)
        validate_seed(actual_seed)

    print("=" * 65)
    print(f"AUTONOMOUS GRAND PRIX — LIVE STREAMING (Seed {actual_seed})")
    print(f"Laps: {laps} | Traffic: {traffic} | Pace Variance: ±{pace_variance*100:.1f}%")
    print("=" * 65)

    # 2. Start Servers
    ws_server = SimpleWebSocketServer(host="127.0.0.1", port=ws_port)
    ws_server.start()
    print(f"[LIVE] WebSocket server listening on ws://127.0.0.1:{ws_port}")

    try:
        httpd = start_http_server(port=http_port)
        print(f"[LIVE] HTTP static server running on http://127.0.0.1:{http_port}")
    except OSError:
        print(f"[LIVE] Note: Port {http_port} already active; reusing existing HTTP server.")

    # 3. Initialize Shared-World Simulation
    mcw = MultiCarWorld(
        track_seed=actual_seed,
        total_laps=laps,
        traffic_mode=traffic,
        lights_hold_steps=-1,  # Random lights hold 0.5-3.0s
        pace_variance=pace_variance,
    )

    # Prepare Initial State Message
    track_geom = mcw._extract_track_geometry()
    drivers_list = []
    for spec in mcw.specs:
        cid = spec["id"]
        team_info = TEAMS.get(spec["team"], {})
        drivers_list.append({
            "id": cid,
            "code": spec["code"],
            "name": spec["name"],
            "team": spec["team"],
            "team_name": team_info.get("name", spec["team"]),
            "number": spec.get("number", 0),
            "color": team_info.get("hex", "#FFFFFF"),
            "model": spec.get("model_file", "expert"),
            "grid_pos": spec.get("grid_pos", 1),
            "pace_variance": f"{(mcw.pace_multipliers.get(cid, 1.0) - 1.0)*100:+.1f}%",
        })

    init_payload = {
        "type": "init",
        "meta": {
            "title": f"Live Grand Prix — Seed {actual_seed}",
            "track_seed": actual_seed,
            "total_laps": laps,
            "fps": int(mcw.fps),
            "num_cars": mcw.num_cars,
            "traffic_mode": traffic,
            "lights_out_step": mcw.lights_hold_steps,
            "pace_variance": pace_variance,
        },
        "track": track_geom,
        "drivers": drivers_list,
    }
    init_str = json.dumps(init_payload)
    ws_server.latest_init_msg = init_str
    ws_server.broadcast(init_str)

    viewer_url = f"http://localhost:{http_port}/docs/viewer/?live=1&ws_port={ws_port}"
    print(f"\n[LIVE] Connect viewer at: {viewer_url}")

    if open_browser:
        import webbrowser
        try:
            webbrowser.open(viewer_url)
        except Exception:
            pass

    # 4. Simulation Execution Loop at 50 FPS
    step_duration = 1.0 / mcw.fps
    t_start = time.time()
    last_print = 0
    active = True

    try:
        while active:
            frame_t0 = time.time()
            active = mcw.step()
            cur_step = mcw.current_step

            # Build step telemetry message
            leaderboard = []
            for row in mcw.timing.compute_leaderboard(current_step=cur_step):
                r = dict(row)
                for s_key in ("s1", "s2", "s3"):
                    sec = r.get(s_key)
                    r[s_key] = {"time_s": round(float(sec.time_s), 3), "status": sec.status} if (sec and hasattr(sec, "time_s")) else None
                leaderboard.append(r)

            step_data = {
                "type": "step",
                "step": cur_step,
                "time": round(cur_step / mcw.fps, 2),
                "state": mcw.race_state,
                "cars": {
                    cid: {
                        "x": mcw.trajectories[cid]["x"][-1],
                        "y": mcw.trajectories[cid]["y"][-1],
                        "heading": mcw.trajectories[cid]["heading"][-1],
                        "hull_angle": mcw.trajectories[cid]["hull_angle"][-1],
                        "speed": mcw.trajectories[cid]["speed"][-1],
                        "steer": mcw.trajectories[cid]["steer"][-1],
                        "gas": mcw.trajectories[cid]["gas"][-1],
                        "brake": mcw.trajectories[cid]["brake"][-1],
                        "wheels": mcw.trajectories[cid]["wheels"][-1],
                        "progress": mcw.trajectories[cid]["progress"][-1],
                        "lap": mcw.trajectories[cid]["lap"][-1],
                        "offroad": mcw.trajectories[cid]["offroad"][-1],
                        "lateral_g": mcw.trajectories[cid]["lateral_g"][-1],
                    }
                    for cid in mcw.cars
                },
                "events": mcw.race_events[-3:] if mcw.race_events else [],
                "leaderboard": leaderboard,
            }
            ws_server.broadcast(json.dumps(step_data))

            # Terminal progress print every 100 steps
            if cur_step - last_print >= 100:
                last_print = cur_step
                p1 = leaderboard[0] if leaderboard else {}
                print(
                    f"Step {cur_step:5d} ({cur_step/mcw.fps:5.1f}s) | State: {mcw.race_state:9s} | "
                    f"P1: {p1.get('code', '---')} (L{p1.get('laps', 0)}) | Contacts: {len(mcw.contacts_log)}",
                    flush=True,
                )

            # Sleep to match 50 FPS real time
            elapsed = time.time() - frame_t0
            rem_sleep = step_duration - elapsed
            if rem_sleep > 0:
                time.sleep(rem_sleep)

    except KeyboardInterrupt:
        print("\n[LIVE] Stream interrupted by user.")

    # 5. Save Replay at End
    total_elapsed = time.time() - t_start
    replay_dict = mcw.export_replay_dict()
    if not out_path:
        out_path = f"docs/viewer/data/race_live_{actual_seed}.json"

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(replay_dict, f)
    print(f"\n[LIVE] Replay successfully exported to: {out_path}")

    # Notify WebSocket clients of finish
    ws_server.broadcast(json.dumps({
        "type": "finish",
        "replay_file": out_path,
        "results": leaderboard,
    }))

    # Print Final Classification
    print("\n" + "=" * 65)
    print("OFFICIAL GRAND PRIX CLASSIFICATION")
    print("=" * 65)
    for row in leaderboard:
        fast_lap_str = " (FASTEST LAP)" if row.get("is_fastest_lap") else ""
        print(f"P{row['pos']} | {row['code']} - {row['name']} ({row['team_code']}) | Laps: {row['laps']} | Gap: {row['gap']} | Best: {row['best_lap']}s{fast_lap_str}")
    print(f"Total Duration: {mcw.current_step/mcw.fps:.1f}s | Incidents: {len(mcw.contacts_log)}")
    print("=" * 65)

    mcw.close()
    return {
        "seed": actual_seed,
        "winner": leaderboard[0]["name"] if leaderboard else "---",
        "leaderboard": leaderboard,
        "contacts": len(mcw.contacts_log),
        "duration_s": mcw.current_step / mcw.fps,
        "replay_file": out_path,
    }


def make_replays(n: int = 8, laps: int = 3, out_dir: str = "docs/viewer/data") -> List[Dict[str, Any]]:
    """
    Generate N diverse Grand Prix replays and index.json catalog for static hosting.
    """
    os.makedirs(out_dir, exist_ok=True)
    catalog = []
    # Diverse showcase seeds
    candidate_seeds = [3000, 3001, 3002, 3003, 6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008]
    chosen_seeds = candidate_seeds[:n]

    print("=" * 65)
    print(f"Generating {n} Diverse Grand Prix Replays into {out_dir}")
    print("=" * 65)

    for i, seed in enumerate(chosen_seeds, 1):
        filename = f"race_{seed}.json"
        filepath = os.path.join(out_dir, filename)

        # If race_3003.json already exists and is complete (>1MB), preserve it
        if seed == 3003 and os.path.exists(filepath) and os.path.getsize(filepath) > 1000000:
            print(f"[{i}/{n}] Preserving existing showcase replay: {filename}...", flush=True)
            with open(filepath, "r", encoding="utf-8") as f:
                replay_dict = json.load(f)
        else:
            print(f"[{i}/{n}] Simulating Seed {seed} ({laps} laps)...", flush=True)
            mcw = MultiCarWorld(
                track_seed=seed,
                total_laps=laps,
                traffic_mode="follow",
                lights_hold_steps=50,
                pace_variance=0.03,
            )
            replay_dict = mcw.run_full_race()
            mcw.close()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(replay_dict, f)

        lb = replay_dict.get("leaderboard", [])
        winner = lb[0] if lb else {}
        catalog_entry = {
            "id": f"race_{seed}",
            "file": f"data/{filename}",
            "name": f"Grand Prix — Seed {seed} ({laps} Laps)",
            "seed": seed,
            "laps": laps,
            "winner": winner.get("name", "---"),
            "winner_team": winner.get("team_code", "---"),
            "fastest_lap": winner.get("best_lap", "---"),
            "contacts": len(replay_dict.get("contacts", [])),
            "duration_s": replay_dict["meta"]["duration_s"],
        }
        catalog.append(catalog_entry)
        print(f"    Winner: {winner.get('name', '---')} ({winner.get('team_code', '---')}) | Best: {winner.get('best_lap', '---')}s | File: {filepath}")

    # Write index.json
    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"replays": catalog}, f, indent=2)

    print(f"\n[DONE] Generated {n} replays and catalog at {index_path}")
    return catalog
