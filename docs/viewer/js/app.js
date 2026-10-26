/**
 * docs/viewer/js/app.js - Main Application Orchestrator for 3D Replay Viewer & Live Stream.
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

    // Live mode state
    this.isLiveMode = false;
    this.ws = null;
    this._lastSparkStep = -1;

    // HUD Elements
    this.initDOMElements();
    this.setupEventListeners();

    // Check URL parameters (?capture=1&frame=N&camera=3, ?replay=..., ?live=1)
    this.urlParams = new URLSearchParams(window.location.search);
    if (this.urlParams.get('live') === '1') {
      this.connectLiveWebSocket();
    } else {
      this.initReplayMode();
    }

    // Animation Loop
    this.lastTime = performance.now();
    requestAnimationFrame((t) => this.loop(t));
  }

  initDOMElements() {
    this.dom = {
      liveBadge: document.getElementById('live-badge'),
      sessionCountdownBanner: document.getElementById('session-countdown-banner'),
      timelineMarkersContainer: document.getElementById('timeline-markers-container'),
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
      } else if (key === 'C') {
        const calibModal = document.getElementById('calibration-modal');
        if (calibModal) calibModal.classList.toggle('hidden');
      }
    });

    // Scrubber
    this.dom.timeScrubber.addEventListener('input', (e) => {
      if (this.maxSteps > 0 && !this.isLiveMode) {
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

    this.setupCalibrationControls();
  }

  setupCalibrationControls() {
    const modal = document.getElementById('calibration-modal');
    const btnOpen = document.getElementById('btn-calibrate');
    const btnClose = document.getElementById('btn-close-calib');
    const modelSelect = document.getElementById('calib-model-select');
    const showWireframe = document.getElementById('calib-show-wireframe');
    const scaleSlider = document.getElementById('calib-scale');
    const oxSlider = document.getElementById('calib-ox');
    const oySlider = document.getElementById('calib-oy');
    const ozSlider = document.getElementById('calib-oz');
    const rySlider = document.getElementById('calib-ry');
    const rxSlider = document.getElementById('calib-rx');
    const preview = document.getElementById('calib-json-preview');
    const btnCopy = document.getElementById('btn-copy-calib');

    if (btnOpen) {
      btnOpen.addEventListener('click', () => {
        modal.classList.remove('hidden');
        onValChange();
      });
    }
    if (btnClose) {
      btnClose.addEventListener('click', () => {
        modal.classList.add('hidden');
      });
    }

    const onValChange = () => {
      if (!scaleSlider) return;
      document.getElementById('calib-scale-val').innerText = scaleSlider.value;
      document.getElementById('calib-ox-val').innerText = oxSlider.value + 'm';
      document.getElementById('calib-oy-val').innerText = oySlider.value + 'm';
      document.getElementById('calib-oz-val').innerText = ozSlider.value + 'm';
      document.getElementById('calib-ry-val').innerText = rySlider.value + '°';
      document.getElementById('calib-rx-val').innerText = rxSlider.value + '°';

      const calib = {
        scale: parseFloat(scaleSlider.value),
        offsetX: parseFloat(oxSlider.value),
        offsetY: parseFloat(oySlider.value),
        offsetZ: parseFloat(ozSlider.value),
        rotY: parseFloat(rySlider.value),
        rotX: parseFloat(rxSlider.value),
      };

      const targetCar = this.focusedCarId ? this.cars[this.focusedCarId] : Object.values(this.cars)[0];
      if (targetCar && typeof targetCar.applyCalibration === 'function') {
        targetCar.applyCalibration(calib);
      }

      if (preview) {
        preview.value = JSON.stringify(
          {
            model: modelSelect ? modelSelect.value : 'procedural',
            scale: parseFloat(scaleSlider.value),
            offset: [parseFloat(oxSlider.value), parseFloat(oySlider.value), parseFloat(ozSlider.value)],
            rot_y_deg: parseFloat(rySlider.value),
            rot_x_deg: parseFloat(rxSlider.value),
          },
          null,
          2
        );
      }
    };

    [scaleSlider, oxSlider, oySlider, ozSlider, rySlider, rxSlider].forEach((s) => {
      if (s) s.addEventListener('input', onValChange);
    });

    if (showWireframe) {
      showWireframe.addEventListener('change', (e) => {
        Object.values(this.cars).forEach((car) => {
          if (typeof car.togglePhysicsWireframe === 'function') {
            car.togglePhysicsWireframe(e.target.checked);
          }
        });
      });
    }

    if (modelSelect) {
      modelSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        const targetCar = this.focusedCarId ? this.cars[this.focusedCarId] : Object.values(this.cars)[0];
        if (targetCar && typeof targetCar.loadCustomModel === 'function') {
          targetCar.loadCustomModel(val);
          if (val === 'aston_martin_f1') scaleSlider.value = 0.001;
          else if (val === 'f1_senna_1991') scaleSlider.value = 0.117;
          else if (val === 'racecar_rc09') scaleSlider.value = 0.55;
          else scaleSlider.value = 1.0;
          onValChange();
        }
      });
    }

    if (btnCopy) {
      btnCopy.addEventListener('click', () => {
        if (preview) {
          navigator.clipboard.writeText(preview.value);
          btnCopy.innerText = '✅ Copied!';
          setTimeout(() => (btnCopy.innerText = '📋 Copy Config'), 2000);
        }
      });
    }
  }

  initReplayMode() {
    if (this.dom.liveBadge) {
      this.dom.liveBadge.innerText = 'REPLAY';
      this.dom.liveBadge.classList.remove('live');
    }
    const replayFile = this.urlParams.get('replay');
    if (replayFile) {
      this.loadReplay(replayFile);
    } else {
      // Fetch catalog index.json and pick random replay
      fetch('data/index.json')
        .then((res) => {
          if (!res.ok) throw new Error('No index.json');
          return res.json();
        })
        .then((catalog) => {
          const replays = Array.isArray(catalog) ? catalog : (catalog.replays || []);
          if (replays && replays.length > 0) {
            if (this.dom.replaySelect) {
              this.dom.replaySelect.innerHTML = '';
              replays.forEach((item) => {
                const opt = document.createElement('option');
                opt.value = item.file;
                opt.innerText = item.name;
                this.dom.replaySelect.appendChild(opt);
              });
            }
            const randomIndex = Math.floor(Math.random() * replays.length);
            const chosen = replays[randomIndex];
            if (this.dom.replaySelect) {
              this.dom.replaySelect.value = chosen.file;
            }
            this.loadReplay(chosen.file);
          } else {
            this.loadReplay('data/race_3003.json');
          }
        })
        .catch(() => {
          this.loadReplay('data/race_3003.json');
        });
    }
  }

  connectLiveWebSocket() {
    this.isLiveMode = true;
    if (this.dom.liveBadge) {
      this.dom.liveBadge.innerText = '● LIVE';
      this.dom.liveBadge.classList.add('live');
    }
    if (this.dom.startGantryModal) {
      this.dom.startGantryModal.classList.add('hidden');
    }
    this.isManualStartPending = false;
    this.isPlaying = true;
    this.dom.btnPlayPause.disabled = true;

    const wsPort = this.urlParams.get('ws_port') || '8765';
    const wsHost = window.location.hostname || '127.0.0.1';
    const wsUrl = `ws://${wsHost}:${wsPort}`;
    console.log('[LIVE] Connecting to WebSocket:', wsUrl);

    try {
      this.ws = new WebSocket(wsUrl);
      this.ws.onopen = () => {
        console.log('[LIVE] WebSocket connected.');
        this.showRaceControlNotice('INFO', 'CONNECTED TO LIVE RACE STREAM');
      };
      this.ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          this.handleLiveMessage(msg);
        } catch (e) {
          console.error('[LIVE] Message parse error:', e);
        }
      };
      this.ws.onclose = () => {
        console.log('[LIVE] WebSocket connection closed.');
        if (this.dom.liveBadge) {
          this.dom.liveBadge.innerText = 'DISCONNECTED';
          this.dom.liveBadge.classList.remove('live');
        }
      };
      this.ws.onerror = (err) => {
        console.error('[LIVE] WebSocket error:', err);
      };
    } catch (err) {
      console.error('[LIVE] Failed to connect WebSocket:', err);
    }
  }

  handleLiveMessage(msg) {
    if (msg.type === 'init') {
      this.parseAndInitLive(msg);
    } else if (msg.type === 'step') {
      this.applyLiveStep(msg);
    } else if (msg.type === 'finish') {
      this.showRaceControlNotice('CHEQUERED', 'GRAND PRIX FINISHED — REPLAY SAVED');
      if (this.dom.liveBadge) {
        this.dom.liveBadge.innerText = 'REPLAY';
        this.dom.liveBadge.classList.remove('live');
      }
      this.dom.btnPlayPause.disabled = false;
    }
  }

  parseAndInitLive(msg) {
    this.replayData = {
      meta: msg.meta,
      track: msg.track,
      drivers: msg.drivers,
      cars: {},
      events: [],
      contacts: [],
    };
    (msg.drivers || []).forEach((d) => {
      this.replayData.cars[d.id] = {
        x: [], y: [], heading: [], hull_angle: [], speed: [],
        steer: [], gas: [], brake: [], wheels: [], progress: [],
        lap: [], offroad: [], lateral_g: []
      };
    });

    this.scene.buildTrack(msg.track);
    this.cameraDirector.setTracksidePosts(msg.track.centerline);

    while (this.scene.carsGroup.children.length > 0) {
      this.scene.carsGroup.remove(this.scene.carsGroup.children[0]);
    }
    this.cars = {};

    (msg.drivers || []).forEach((d) => {
      const car = new F1Car(d);
      this.cars[d.id] = car;
      this.scene.carsGroup.add(car.root);
    });

    this.focusedCarId = msg.drivers && msg.drivers.length > 0 ? msg.drivers[0].id : null;
    this.renderTimingTower();
    this.drawMinimapTrack(msg.track);
  }

  applyLiveStep(msg) {
    this.currentStep = msg.step;
    this.maxSteps = Math.max(this.maxSteps, msg.step);

    // Update each car position directly
    for (const [cid, cData] of Object.entries(msg.cars)) {
      const car = this.cars[cid];
      if (car) {
        car.update(
          cData.x,
          cData.y,
          cData.hull_angle !== undefined ? cData.hull_angle : (cData.heading - Math.PI / 2),
          cData.steer || 0.0,
          cData.gas || 0.0,
          cData.brake || 0.0,
          cData.speed || 0.0,
          cData.lateral_g || 0.0
        );
      }
      if (this.replayData && this.replayData.cars[cid]) {
        const traj = this.replayData.cars[cid];
        traj.x.push(cData.x);
        traj.y.push(cData.y);
        traj.speed.push(cData.speed);
        traj.steer.push(cData.steer);
        traj.gas.push(cData.gas);
        traj.brake.push(cData.brake);
        traj.wheels.push(cData.wheels);
        traj.progress.push(cData.progress);
        traj.lap.push(cData.lap);
      }
    }

    // Update HUD
    const curSec = Math.floor(msg.step / 50);
    this.dom.currentTimeText.innerText = `LIVE: ${this.fmtTime(curSec)}`;
    if (this.focusedCarId && msg.cars[this.focusedCarId]) {
      const f = msg.cars[this.focusedCarId];
      const spdKmh = Math.round((f.speed || 0.0) * 3.6);
      this.dom.telemetrySpeed.innerText = spdKmh;
      this.dom.barSpeed.style.width = `${Math.min(100, (spdKmh / 160) * 100)}%`;
      this.dom.telemetryGas.innerText = `${Math.round((f.gas || 0.0) * 100)}%`;
      this.dom.barGas.style.width = `${Math.round((f.gas || 0.0) * 100)}%`;
      this.dom.telemetryBrake.innerText = `${Math.round((f.brake || 0.0) * 100)}%`;
      this.dom.barBrake.style.width = `${Math.round((f.brake || 0.0) * 100)}%`;
      this.dom.telemetrySteer.innerText = (f.steer > 0 ? '+' : '') + (f.steer || 0.0).toFixed(2);
      this.dom.barSteer.style.width = `${((f.steer || 0.0) + 1.0) * 50.0}%`;
      this.dom.telemetryProgress.innerText = `${(f.progress || 0.0).toFixed(1)}%`;
      this.dom.barProgress.style.width = `${f.progress || 0.0}%`;
      this.dom.telemetryLap.innerText = `LAP ${f.lap || 1}/${this.replayData.meta.total_laps || 10}`;

      const wheels = f.wheels || [true, true, true, true];
      this.dom.wheelDots.forEach((dot, idx) => {
        if (dot) {
          dot.classList.toggle('on-track', wheels[idx]);
          dot.classList.toggle('off-track', !wheels[idx]);
        }
      });
    }

    // Update timing leaderboard
    if (msg.leaderboard) {
      this.updateLiveTimingTower(msg.leaderboard);
    }

    // Events
    if (msg.events && msg.events.length > 0) {
      const latest = msg.events[msg.events.length - 1];
      this.showRaceControlNotice(latest.type, latest.text);
    }

    // Minimap
    this.updateMinimapDots(Math.floor(this.currentStep));
  }

  updateLiveTimingTower(leaderboard) {
    const list = this.dom.timingTower;
    if (!list) return;
    list.innerHTML = '';
    leaderboard.forEach((d) => {
      const row = document.createElement('div');
      row.className = `driver-row ${d.id === this.focusedCarId ? 'focused' : ''}`;
      row.addEventListener('click', (e) => {
        e.stopPropagation();
        this.focusedCarId = d.id;
        document.querySelectorAll('.driver-row').forEach((r) => r.classList.remove('focused'));
        row.classList.add('focused');
      });

      const fastLapHtml = d.is_fastest_lap
        ? `<span style="color: var(--f1-purple); font-size: 0.7rem; font-weight: 700; margin-left: 4px;">🟣 FL</span>`
        : '';
      const penaltyHtml = (d.penalty_seconds && d.penalty_seconds > 0)
        ? `<span style="color: var(--f1-red); font-size: 0.7rem; font-weight: 700; margin-left: 4px;">+${d.penalty_seconds}s</span>`
        : '';

      const paceVarianceHtml = d.pace_variance
        ? `<span style="color: var(--text-muted); font-size: 0.68rem; margin-left: 4px;">(${d.pace_variance})</span>`
        : '';

      row.innerHTML = `
        <span class="driver-pos">${d.pos}</span>
        <span class="driver-bar" style="background-color: ${d.color || '#fff'};"></span>
        <div class="driver-info">
          <div class="driver-code">${d.code} ${fastLapHtml} ${penaltyHtml} ${paceVarianceHtml}</div>
          <div class="driver-team">${d.team || ''}</div>
        </div>
        <div class="driver-gap">${d.gap !== undefined ? d.gap : '-'}</div>
      `;
      list.appendChild(row);
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
    while (this.scene.carsGroup.children.length > 0) {
      this.scene.carsGroup.remove(this.scene.carsGroup.children[0]);
    }
    this.cars = {};

    (data.drivers || []).forEach((d) => {
      const car = new F1Car(d);
      this.cars[d.id] = car;
      this.scene.carsGroup.add(car.root);
    });

    this.focusedCarId = data.drivers && data.drivers.length > 0 ? data.drivers[0].id : null;

    // 3. Render Initial Grid State
    this.updateCars(0.0);

    // 4. Populate Timing Tower
    this.renderTimingTower();

    // 5. Draw Minimap Track Outline
    this.drawMinimapTrack(data.track);

    // 6. Reset Start Gantry
    this.resetStartGantryUI();

    // 7. Populate Contact Markers on Timeline
    if (this.dom.timelineMarkersContainer) {
      this.dom.timelineMarkersContainer.innerHTML = '';
      (data.contacts || []).forEach((c) => {
        const pct = (c.step / Math.max(1, this.maxSteps)) * 100;
        const marker = document.createElement('div');
        marker.className = 'timeline-marker' + (c.at_fault ? ' penalty' : '');
        marker.style.left = `${pct}%`;
        marker.title = `Contact at Step ${c.step} (${c.time_s}s): ${c.car_a} vs ${c.car_b}`;
        marker.addEventListener('click', (e) => {
          e.stopPropagation();
          this.seekToStep(c.step);
        });
        this.dom.timelineMarkersContainer.appendChild(marker);
      });
    }

    // Capture parameters (?capture=1&frame=N&camera=3)
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

    // Frame seeking API for automation / video3d
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
        const randomHoldMs = 800 + Math.random() * 800;
        this.dom.gantryStatusText.innerText = 'HOLDING FOR LIGHTS OUT...';

        setTimeout(() => {
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
    if (this.isLiveMode) return;
    if (this.isManualStartPending) {
      this.triggerManualStartSequence();
      return;
    }
    this.isPlaying = !this.isPlaying;
    this.dom.btnPlayPause.innerText = this.isPlaying ? '⏸️ Pause' : '▶️ Play';
  }

  updateCars(stepFloat) {
    if (!this.replayData || !this.replayData.cars) return;
    const baseStep = Math.floor(stepFloat);
    const alpha = stepFloat - baseStep;
    const nextStep = Math.min(this.maxSteps, baseStep + 1);

    for (const [cid, car] of Object.entries(this.cars)) {
      const traj = this.replayData.cars[cid];
      if (!traj || baseStep >= traj.x.length) continue;

      const x = traj.x[baseStep] * (1.0 - alpha) + (traj.x[nextStep] || traj.x[baseStep]) * alpha;
      const y = traj.y[baseStep] * (1.0 - alpha) + (traj.y[nextStep] || traj.y[baseStep]) * alpha;

      let hullAngle = 0.0;
      if (traj.hull_angle && traj.hull_angle[baseStep] !== undefined) {
        hullAngle = traj.hull_angle[baseStep] * (1.0 - alpha) + (traj.hull_angle[nextStep] || traj.hull_angle[baseStep]) * alpha;
      } else {
        const heading = traj.heading[baseStep] * (1.0 - alpha) + (traj.heading[nextStep] || traj.heading[baseStep]) * alpha;
        hullAngle = heading - Math.PI / 2;
      }

      const steer = traj.steer[baseStep] || 0.0;
      const gas = traj.gas[baseStep] || 0.0;
      const brake = traj.brake[baseStep] || 0.0;
      const speed = traj.speed[baseStep] || 0.0;
      const latG = (traj.lateral_g && traj.lateral_g[baseStep] !== undefined) ? traj.lateral_g[baseStep] : 0.0;

      car.update(x, y, hullAngle, steer, gas, brake, speed, latG);
    }
  }

  renderTimingTower() {
    const list = this.dom.timingTower;
    if (!list || !this.replayData || !this.replayData.drivers) return;
    list.innerHTML = '';

    this.replayData.drivers.forEach((d, idx) => {
      const row = document.createElement('div');
      row.className = `driver-row ${d.id === this.focusedCarId ? 'focused' : ''}`;
      row.dataset.carId = d.id;

      row.addEventListener('click', (e) => {
        e.stopPropagation();
        this.focusedCarId = d.id;
        document.querySelectorAll('.driver-row').forEach((r) => r.classList.remove('focused'));
        row.classList.add('focused');
      });

      const paceVarianceHtml = d.pace_variance
        ? `<span style="color: var(--text-muted); font-size: 0.68rem; margin-left: 4px;">(${d.pace_variance})</span>`
        : '';

      row.innerHTML = `
        <span class="driver-pos">${idx + 1}</span>
        <span class="driver-bar" style="background-color: ${d.color};"></span>
        <div class="driver-info">
          <div class="driver-code">${d.code} ${paceVarianceHtml}</div>
          <div class="driver-team">${d.team}</div>
        </div>
        <div class="driver-gap">-</div>
      `;
      list.appendChild(row);
    });
  }

  updateHUD() {
    if (!this.replayData || !this.focusedCarId) return;
    const traj = this.replayData.cars[this.focusedCarId];
    if (!traj) return;

    const baseStep = Math.floor(this.currentStep);

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
    const steerPct = (steerVal + 1.0) * 50.0;
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

    // Check race control events
    (this.replayData.events || []).forEach((ev) => {
      if (Math.abs(ev.step - baseStep) <= 2) {
        this.showRaceControlNotice(ev.type, ev.text);
      }
    });

    // Cosmetic collision sparks and camera shake on contact
    if (this.replayData && this.replayData.contacts) {
      this.replayData.contacts.forEach((c) => {
        if (Math.abs(c.step - baseStep) <= 1 && c.step !== this._lastSparkStep) {
          this._lastSparkStep = c.step;
          const car = this.cars[c.car_a] || this.cars[c.car_b];
          if (car) {
            this.scene.emitSparks(car.root.position, 35);
          }
          this.cameraDirector.addTrauma(0.55);
        }
      });
    }

    // Session-end countdown banner check
    if (this.dom.sessionCountdownBanner && this.maxSteps > 750) {
      const cooldownStart = this.maxSteps - 750;
      if (baseStep >= cooldownStart && baseStep < this.maxSteps) {
        const remSteps = this.maxSteps - baseStep;
        const remSec = Math.ceil(remSteps / 50);
        this.dom.sessionCountdownBanner.innerText = `SESSION ENDS IN 00:${String(remSec).padStart(2, '0')}`;
        this.dom.sessionCountdownBanner.classList.remove('hidden');
      } else {
        this.dom.sessionCountdownBanner.classList.add('hidden');
      }
    }

    this.updateMinimapDots(baseStep);
  }

  showRaceControlNotice(level, text) {
    if (!this.dom.raceControlBanner || !this.dom.rcText) return;
    this.dom.rcText.innerText = text;
    this.dom.raceControlBanner.classList.remove('hidden');

    clearTimeout(this.rcTimeout);
    this.rcTimeout = setTimeout(() => {
      this.dom.raceControlBanner.classList.add('hidden');
    }, 4500);
  }

  drawMinimapTrack(trackData) {
    const canvas = this.dom.minimapCanvas;
    if (!canvas || !trackData || !trackData.centerline) return;

    let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
    trackData.centerline.forEach((p) => {
      minX = Math.min(minX, p[0]);
      maxX = Math.max(maxX, p[0]);
      minZ = Math.min(minZ, -p[1]);
      maxZ = Math.max(maxZ, -p[1]);
    });

    const w = canvas.width;
    const h = canvas.height;
    const pad = 16;
    const scaleX = (w - pad * 2) / (maxX - minX || 1);
    const scaleZ = (h - pad * 2) / (maxZ - minZ || 1);
    this.mmScale = Math.min(scaleX, scaleZ);
    this.mmOffsetX = (w - (maxX - minX) * this.mmScale) / 2 - minX * this.mmScale;
    this.mmOffsetZ = (h - (maxZ - minZ) * this.mmScale) / 2 - minZ * this.mmScale;

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

    if (this.isPlaying && !this.isManualStartPending && !this.isLiveMode) {
      this.currentStep += 50 * dt * this.playbackSpeed;
      if (this.currentStep >= this.maxSteps) {
        this.currentStep = this.maxSteps;
        this.isPlaying = false;
        this.dom.btnPlayPause.innerText = '▶️ Replay';
      }
      this.updateCars(this.currentStep);
      this.updateHUD();
    }

    // Update cosmetic collision sparks
    this.scene.updateSparks(dt);

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
