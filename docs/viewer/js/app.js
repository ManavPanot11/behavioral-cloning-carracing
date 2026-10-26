/**
 * docs/viewer/js/app.js - Main Application Orchestrator for 3D Replay Viewer
 */

import { TrackScene } from './scene.js';
import { F1Car } from './car_model.js';
import { CameraDirector } from './cameras.js';

class ReplayApp {
  constructor() {
    this.container = document.getElementById('canvas-container');
    this.scene = new TrackScene(this.container);
    this.cameraDirector = new CameraDirector(this.container, this.scene);

    this.replayData = null;
    this.cars = {}; // id -> F1Car
    this.focusedCarId = null;

    // Playback state
    this.isPlaying = false;
    this.currentStep = 0.0;
    this.maxSteps = 0;
    this.playbackSpeed = 1.0;
    this.isManualStartPending = true; // Wait for manual start
    this.lightsSequenceActive = false;

    // HUD Elements
    this.initDOMElements();
    this.setupEventListeners();

    // Check URL parameters (?capture=1&frame=N&camera=3, ?replay=...)
    this.urlParams = new URLSearchParams(window.location.search);
    const replayFile = this.urlParams.get('replay') || 'data/race_3003.json';
    this.loadReplay(replayFile);

    // Animation Loop
    this.lastTime = performance.now();
    requestAnimationFrame((t) => this.loop(t));
  }

  initDOMElements() {
    this.dom = {
      timingTower: document.getElementById('driver-tower'),
      telemetrySpeed: document.getElementById('telemetry-speed'),
      barSpeed: document.getElementById('bar-speed'),
      telemetryGas: document.getElementById('telemetry-gas'),
      barGas: document.getElementById('bar-gas'),
      telemetryBrake: document.getElementById('telemetry-brake'),
      barBrake: document.getElementById('bar-brake'),
      telemetrySteer: document.getElementById('telemetry-steer'),
      barSteer: document.getElementById('bar-steer'),
      telemetryProgress: document.getElementById('telemetry-progress'),
      barProgress: document.getElementById('bar-progress'),
      telemetryLap: document.getElementById('telemetry-lap'),
      wheelDots: [
        document.getElementById('wheel-fl'),
        document.getElementById('wheel-fr'),
        document.getElementById('wheel-rl'),
        document.getElementById('wheel-rr'),
      ],
      timeScrubber: document.getElementById('time-scrubber'),
      currentTimeText: document.getElementById('current-time'),
      btnPlayPause: document.getElementById('btn-play-pause'),
      startGantryModal: document.getElementById('start-gantry-overlay'),
      btnStartRace: document.getElementById('btn-start-race'),
      gantryLamps: [
        document.getElementById('lamp-1'),
        document.getElementById('lamp-2'),
        document.getElementById('lamp-3'),
        document.getElementById('lamp-4'),
        document.getElementById('lamp-5'),
      ],
      gantryStatusText: document.getElementById('gantry-status-text'),
      raceControlBanner: document.getElementById('race-control-banner'),
      rcText: document.getElementById('rc-text'),
      minimapCanvas: document.getElementById('minimap-canvas'),
      replaySelect: document.getElementById('replay-select'),
      gridConfigBtn: document.getElementById('btn-grid-config'),
      gridModal: document.getElementById('grid-modal'),
      btnCloseGrid: document.getElementById('btn-close-grid'),
      gridSlotsContainer: document.getElementById('grid-slots-container'),
    };
  }

  setupEventListeners() {
    window.addEventListener('resize', () => {
      this.scene.onResize();
      this.cameraDirector.onResize();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      const key = e.key.toUpperCase();
      if (key >= '1' && key <= '6') {
        const mode = parseInt(key);
        this.cameraDirector.setMode(mode);
        this.updateActiveCameraButtons(mode);
      } else if (e.code === 'Space') {
        e.preventDefault();
        this.togglePlayPause();
      } else if (key === 'F') {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      } else if (key === 'H') {
        this.toggleAboutModal();
      }
    });

    // Scrubber
    this.dom.timeScrubber.addEventListener('input', (e) => {
      if (this.maxSteps > 0) {
        const step = (parseFloat(e.target.value) / 100.0) * this.maxSteps;
        this.seekToStep(step);
      }
    });

    // Play / Pause Button
    this.dom.btnPlayPause.addEventListener('click', () => this.togglePlayPause());

    // Speed Toggles
    document.querySelectorAll('.speed-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.speed-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        this.playbackSpeed = parseFloat(btn.dataset.speed);
      });
    });

    // Camera Mode Buttons
    document.querySelectorAll('.cam-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const mode = parseInt(btn.dataset.mode);
        this.cameraDirector.setMode(mode);
        this.updateActiveCameraButtons(mode);
      });
    });

    // Manual Start Race Button
    this.dom.btnStartRace.addEventListener('click', () => {
      this.triggerManualStartSequence();
    });

    // Replay Dropdown
    if (this.dom.replaySelect) {
      this.dom.replaySelect.addEventListener('change', (e) => {
        this.loadReplay(e.target.value);
      });
    }

    // Grid Config Modal
    if (this.dom.gridConfigBtn) {
      this.dom.gridConfigBtn.addEventListener('click', () => {
        this.openGridModal();
      });
    }
    if (this.dom.btnCloseGrid) {
      this.dom.btnCloseGrid.addEventListener('click', () => {
        this.dom.gridModal.classList.add('hidden');
      });
    }

    // Drag and Drop JSON Replay
    window.addEventListener('dragover', (e) => e.preventDefault());
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        const reader = new FileReader();
        reader.onload = (evt) => {
          try {
            const data = JSON.parse(evt.target.result);
            this.parseAndInitReplay(data);
          } catch (err) {
            alert('Failed to parse replay JSON: ' + err.message);
          }
        };
        reader.readAsText(file);
      }
    });
  }

  updateActiveCameraButtons(mode) {
    document.querySelectorAll('.cam-btn').forEach((b) => {
      b.classList.toggle('active', parseInt(b.dataset.mode) === mode);
    });
  }

  loadReplay(url) {
    fetch(url)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
        return res.json();
      })
      .then((data) => {
        this.parseAndInitReplay(data);
      })
      .catch((err) => {
        console.error('Replay load failed:', err);
        // Fallback default message
        this.showRaceControlNotice('NOTICE', `Could not load ${url}. Drag & drop a JSON replay file to view.`);
      });
  }

  parseAndInitReplay(data) {
    this.replayData = data;
    this.maxSteps = (data.meta && data.meta.total_steps) ? data.meta.total_steps - 1 : 1000;
    this.currentStep = 0.0;
    this.isPlaying = false;
    this.isManualStartPending = true;

    // 1. Build Track Geometry in 3D
    this.scene.buildTrack(data.track);
    this.cameraDirector.setTracksidePosts(data.track.centerline);

    // 2. Build Cars in 3D
    // Clear old cars
    while (this.scene.carsGroup.children.length > 0) {
      this.scene.carsGroup.remove(this.scene.carsGroup.children[0]);
    }
    this.cars = {};

    (data.drivers || []).forEach((d) => {
      const car = new F1Car(d);
      this.cars[d.id] = car;
      this.scene.carsGroup.add(car.root);
    });

    // Default focus: Leader / Pole sitter (P1)
    this.focusedCarId = data.drivers && data.drivers.length > 0 ? data.drivers[0].id : null;

    // 3. Render Initial Grid State (Cars lined up on grid)
    this.updateCars(0.0);

    // 4. Populate Timing Tower
    this.renderTimingTower();

    // 5. Draw Minimap Track Outline
    this.drawMinimapTrack(data.track);

    // 6. Reset Start Gantry
    this.resetStartGantryUI();

    // Check for deterministic capture parameters (?capture=1&frame=N&camera=3)
    if (this.urlParams.get('capture') === '1') {
      if (this.dom.startGantryModal) this.dom.startGantryModal.classList.add('hidden');
      this.isManualStartPending = false;
      const targetFrame = parseInt(this.urlParams.get('frame') || '0');
      const camMode = parseInt(this.urlParams.get('camera') || '1');
      this.cameraDirector.setMode(camMode);
      this.seekToStep(targetFrame);
      this.scene.renderer.render(this.scene.scene, this.cameraDirector.activeCamera);
      window.__FRAME_RENDERED = true;
    }

    // Expose frame seeking API for automation / video3d
    window.seekFrame = (step, camMode) => {
      if (this.dom.startGantryModal) this.dom.startGantryModal.classList.add('hidden');
      this.isManualStartPending = false;
      if (camMode !== undefined && camMode !== null) {
        const cm = parseInt(camMode);
        this.cameraDirector.setMode(cm);
        this.updateActiveCameraButtons(cm);
      }
      this.seekToStep(step);
      this.scene.renderer.render(this.scene.scene, this.cameraDirector.activeCamera);
      window.__FRAME_RENDERED = true;
      return true;
    };
  }

  seekToStep(step) {
    this.currentStep = Math.max(0, Math.min(this.maxSteps, step));
    this.updateCars(this.currentStep);
    this.updateHUD();
    if (this.focusedCarId && this.cars[this.focusedCarId]) {
      this.cameraDirector.update(this.cars[this.focusedCarId], 0.02);
    }
  }

  triggerManualStartSequence() {
    if (this.lightsSequenceActive) return;
    this.lightsSequenceActive = true;
    this.dom.btnStartRace.disabled = true;
    this.dom.gantryStatusText.innerText = 'STARTING SEQUENCE INITIATED...';

    let currentLamp = 0;
    const interval = setInterval(() => {
      currentLamp++;
      this.setGantryLampUI(currentLamp);
      this.scene.setStartLights(currentLamp, false);

      if (currentLamp === 5) {
        clearInterval(interval);
        // Random pause between 0.8s and 1.6s
        const randomHoldMs = 800 + Math.random() * 800;
        this.dom.gantryStatusText.innerText = 'HOLDING FOR LIGHTS OUT...';

        setTimeout(() => {
          // LIGHTS OUT!
          this.setGantryLampUI(0, true);
          this.scene.setStartLights(0, true);
          this.dom.gantryStatusText.innerText = 'LIGHTS OUT AND AWAY WE GO!';
          this.showRaceControlNotice('START', 'LIGHTS OUT AND AWAY WE GO!');

          setTimeout(() => {
            this.dom.startGantryModal.classList.add('hidden');
            this.isManualStartPending = false;
            this.lightsSequenceActive = false;
            this.isPlaying = true;
            this.dom.btnPlayPause.innerText = '⏸️ Pause';
          }, 600);
        }, randomHoldMs);
      }
    }, 800);
  }

  setGantryLampUI(count, allOut = false) {
    this.dom.gantryLamps.forEach((lamp, idx) => {
      if (allOut) {
        lamp.classList.remove('active');
        lamp.classList.add('out');
      } else if (idx < count) {
        lamp.classList.add('active');
        lamp.classList.remove('out');
      } else {
        lamp.classList.remove('active', 'out');
      }
    });
  }

  resetStartGantryUI() {
    this.dom.startGantryModal.classList.remove('hidden');
    this.dom.btnStartRace.disabled = false;
    this.dom.gantryStatusText.innerText = 'CARS STATIONARY ON GRID · CLICK TO START';
    this.setGantryLampUI(0);
    this.scene.setStartLights(0, false);
  }

  togglePlayPause() {
    if (this.isManualStartPending) {
      this.triggerManualStartSequence();
      return;
    }
    this.isPlaying = !this.isPlaying;
    this.dom.btnPlayPause.innerText = this.isPlaying ? '⏸️ Pause' : '▶️ Play';
  }

  renderTimingTower() {
    if (!this.replayData || !this.replayData.drivers) return;
    this.dom.timingTower.innerHTML = '';

    this.replayData.drivers.forEach((d, idx) => {
      const row = document.createElement('div');
      row.className = `driver-row ${d.id === this.focusedCarId ? 'active' : ''}`;
      row.dataset.carId = d.id;

      row.innerHTML = `
        <span class="driver-pos">${idx + 1}</span>
        <div class="driver-stripe" style="background: ${d.color};"></div>
        <div class="driver-info">
          <span class="driver-code">${d.code}</span>
          <span class="driver-name-sub">${d.name} (${d.team})</span>
        </div>
        <div class="driver-gap" id="gap-${d.id}">${idx === 0 ? 'LEADER' : '+0.00s'}</div>
      `;

      // CRITICAL FIX: Clicking driver switches focus without resetting playback time
      row.addEventListener('click', () => {
        this.focusedCarId = d.id;
        document.querySelectorAll('.driver-row').forEach((r) => r.classList.remove('active'));
        row.classList.add('active');
        this.updateHUD();
      });

      this.dom.timingTower.appendChild(row);
    });
  }

  openGridModal() {
    if (!this.replayData || !this.replayData.drivers) return;
    this.dom.gridModal.classList.remove('hidden');
    this.dom.gridSlotsContainer.innerHTML = '';

    this.replayData.drivers.forEach((d, i) => {
      const item = document.createElement('div');
      item.className = 'grid-slot-item';
      item.innerHTML = `
        <span class="slot-badge">P${i + 1}</span>
        <div class="slot-color" style="background: ${d.color};"></div>
        <span class="slot-code">${d.code}</span>
        <span class="slot-name">${d.name}</span>
      `;
      this.dom.gridSlotsContainer.appendChild(item);
    });
  }

  showRaceControlNotice(type, text) {
    if (!this.dom.raceControlBanner) return;
    this.dom.rcText.innerText = text;
    this.dom.raceControlBanner.classList.remove('hidden');
    clearTimeout(this.rcTimeout);
    this.rcTimeout = setTimeout(() => {
      this.dom.raceControlBanner.classList.add('hidden');
    }, 4500);
  }

  updateCars(stepFloat) {
    if (!this.replayData || !this.replayData.cars) return;

    const baseStep = Math.floor(stepFloat);
    const nextStep = Math.min(this.maxSteps, baseStep + 1);
    const frac = stepFloat - baseStep;

    for (const [cid, car] of Object.entries(this.cars)) {
      const traj = this.replayData.cars[cid];
      if (!traj || !traj.x || baseStep >= traj.x.length) continue;

      // Linear interpolation between frame states
      const x = traj.x[baseStep] * (1.0 - frac) + (traj.x[nextStep] || traj.x[baseStep]) * frac;
      const y = traj.y[baseStep] * (1.0 - frac) + (traj.y[nextStep] || traj.y[baseStep]) * frac;
      const speed = traj.speed[baseStep] * (1.0 - frac) + (traj.speed[nextStep] || traj.speed[baseStep]) * frac;
      const steer = traj.steer[baseStep] * (1.0 - frac) + (traj.steer[nextStep] || traj.steer[baseStep]) * frac;
      const gas = traj.gas[baseStep] * (1.0 - frac) + (traj.gas[nextStep] || traj.gas[baseStep]) * frac;
      const brake = traj.brake[baseStep] * (1.0 - frac) + (traj.brake[nextStep] || traj.brake[baseStep]) * frac;
      const lateralG = traj.lateral_g[baseStep] * (1.0 - frac) + (traj.lateral_g[nextStep] || traj.lateral_g[baseStep]) * frac;

      // Heading angle unwrapped slerp
      const h1 = traj.heading[baseStep];
      const h2 = traj.heading[nextStep] !== undefined ? traj.heading[nextStep] : h1;
      let dh = h2 - h1;
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      const heading = h1 + dh * frac;

      const ha1 = traj.hull_angle[baseStep];
      const ha2 = traj.hull_angle[nextStep] !== undefined ? traj.hull_angle[nextStep] : ha1;
      let dha = ha2 - ha1;
      while (dha > Math.PI) dha -= 2 * Math.PI;
      while (dha < -Math.PI) dha += 2 * Math.PI;
      const hullAngle = ha1 + dha * frac;

      const state = {
        x,
        y,
        heading,
        hull_angle: hullAngle,
        speed,
        steer,
        gas,
        brake,
        lateral_g: lateralG,
        wheels: traj.wheels[baseStep] || [true, true, true, true],
        progress: traj.progress[baseStep] || 0.0,
        lap: traj.lap[baseStep] || 1,
      };

      car.update(state, 0.02);
    }
  }

  updateHUD() {
    if (!this.replayData || !this.focusedCarId) return;

    const baseStep = Math.min(this.maxSteps, Math.floor(this.currentStep));
    const traj = this.replayData.cars[this.focusedCarId];
    if (!traj || baseStep >= traj.x.length) return;

    // Scrubber update
    if (this.maxSteps > 0) {
      this.dom.timeScrubber.value = (this.currentStep / this.maxSteps) * 100.0;
    }
    const curSec = Math.floor(this.currentStep / 50);
    const maxSec = Math.floor(this.maxSteps / 50);
    this.dom.currentTimeText.innerText = `${this.fmtTime(curSec)} / ${this.fmtTime(maxSec)}`;

    // Telemetry Display
    const spdWorld = traj.speed[baseStep] || 0.0;
    const spdKmh = Math.round(spdWorld * 3.6);
    this.dom.telemetrySpeed.innerText = spdKmh;
    this.dom.barSpeed.style.width = `${Math.min(100, (spdKmh / 160) * 100)}%`;

    const gasPct = Math.round((traj.gas[baseStep] || 0.0) * 100);
    this.dom.telemetryGas.innerText = `${gasPct}%`;
    this.dom.barGas.style.width = `${gasPct}%`;

    const brakePct = Math.round((traj.brake[baseStep] || 0.0) * 100);
    this.dom.telemetryBrake.innerText = `${brakePct}%`;
    this.dom.barBrake.style.width = `${brakePct}%`;

    const steerVal = traj.steer[baseStep] || 0.0;
    this.dom.telemetrySteer.innerText = (steerVal > 0 ? '+' : '') + steerVal.toFixed(2);
    const steerPct = (steerVal + 1.0) * 50.0; // 0..100%
    this.dom.barSteer.style.width = `${steerPct}%`;

    const prog = traj.progress[baseStep] || 0.0;
    this.dom.telemetryProgress.innerText = `${prog.toFixed(1)}%`;
    this.dom.barProgress.style.width = `${prog}%`;

    const lapNum = traj.lap[baseStep] || 1;
    this.dom.telemetryLap.innerText = `LAP ${lapNum}/${this.replayData.meta.total_laps || 10}`;

    // 4-Wheel Track Limit Status Dots
    const wheels = traj.wheels[baseStep] || [true, true, true, true];
    this.dom.wheelDots.forEach((dot, idx) => {
      if (dot) {
        dot.classList.toggle('on-track', wheels[idx]);
        dot.classList.toggle('off-track', !wheels[idx]);
      }
    });

    // Check race control events at current step
    (this.replayData.events || []).forEach((ev) => {
      if (Math.abs(ev.step - baseStep) <= 2) {
        this.showRaceControlNotice(ev.type, ev.text);
      }
    });

    // Update Minimap Dots
    this.updateMinimapDots(baseStep);
  }

  drawMinimapTrack(trackData) {
    const canvas = this.dom.minimapCanvas;
    if (!canvas || !trackData || !trackData.centerline) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Find bounding box
    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    trackData.centerline.forEach((p) => {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minZ = Math.min(minZ, -p[1]);
      maxZ = Math.max(maxZ, -p[1]);
    });

    const pad = 18;
    const scaleX = (w - pad * 2) / (maxX - minX);
    const scaleZ = (h - pad * 2) / (maxZ - minZ);
    this.mmScale = Math.min(scaleX, scaleZ);
    this.mmOffsetX = (w - (maxX - minX) * this.mmScale) / 2 - minX * this.mmScale;
    this.mmOffsetZ = (h - (maxZ - minZ) * this.mmScale) / 2 - minZ * this.mmScale;

    // Cache static track path
    this.mmTrackCanvas = document.createElement('canvas');
    this.mmTrackCanvas.width = w;
    this.mmTrackCanvas.height = h;
    const mctx = this.mmTrackCanvas.getContext('2d');

    mctx.fillStyle = '#080c14';
    mctx.fillRect(0, 0, w, h);

    mctx.beginPath();
    trackData.centerline.forEach((p, idx) => {
      const cx = p[0] * this.mmScale + this.mmOffsetX;
      const cz = (-p[1]) * this.mmScale + this.mmOffsetZ;
      if (idx === 0) mctx.moveTo(cx, cz);
      else mctx.lineTo(cx, cz);
    });
    mctx.closePath();
    mctx.strokeStyle = '#242e42';
    mctx.lineWidth = 8;
    mctx.stroke();

    mctx.strokeStyle = '#3d4d6b';
    mctx.lineWidth = 4;
    mctx.stroke();

    // Checkered Start / Finish Marker
    if (trackData.start_finish && trackData.start_finish.p1) {
      const sfX = trackData.start_finish.p1[0] * this.mmScale + this.mmOffsetX;
      const sfZ = (-trackData.start_finish.p1[1]) * this.mmScale + this.mmOffsetZ;
      mctx.fillStyle = '#ffc700';
      mctx.beginPath();
      mctx.arc(sfX, sfZ, 4, 0, Math.PI * 2);
      mctx.fill();
    }
  }

  updateMinimapDots(step) {
    const canvas = this.dom.minimapCanvas;
    if (!canvas || !this.mmTrackCanvas) return;
    const ctx = canvas.getContext('2d');

    ctx.drawImage(this.mmTrackCanvas, 0, 0);

    // Draw each car dot
    for (const [cid, car] of Object.entries(this.cars)) {
      const traj = this.replayData.cars[cid];
      if (!traj || step >= traj.x.length) continue;

      const cx = traj.x[step] * this.mmScale + this.mmOffsetX;
      const cz = (-traj.y[step]) * this.mmScale + this.mmOffsetZ;

      ctx.fillStyle = car.colorHex;
      ctx.beginPath();
      ctx.arc(cx, cz, cid === this.focusedCarId ? 5.5 : 4.0, 0, Math.PI * 2);
      ctx.fill();

      if (cid === this.focusedCarId) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  fmtTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  }

  toggleAboutModal() {
    const modal = document.getElementById('about-modal');
    if (modal) modal.classList.toggle('hidden');
  }

  loop(timestamp) {
    const dt = Math.min(0.1, (timestamp - this.lastTime) / 1000);
    this.lastTime = timestamp;

    if (this.isPlaying && !this.isManualStartPending) {
      this.currentStep += 50 * dt * this.playbackSpeed;
      if (this.currentStep >= this.maxSteps) {
        this.currentStep = this.maxSteps;
        this.isPlaying = false;
        this.dom.btnPlayPause.innerText = '▶️ Replay';
      }
      this.updateCars(this.currentStep);
      this.updateHUD();
    }

    // Camera follow & render
    if (this.focusedCarId && this.cars[this.focusedCarId]) {
      this.cameraDirector.update(this.cars[this.focusedCarId], dt);
    }
    this.scene.renderer.render(this.scene.scene, this.cameraDirector.activeCamera);

    requestAnimationFrame((t) => this.loop(t));
  }
}

window.addEventListener('DOMContentLoaded', () => {
  window.app = new ReplayApp();
});
