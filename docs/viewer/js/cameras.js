/**
 * docs/viewer/js/cameras.js - 6 Multi-View Camera System
 *
 * Implements:
 * 1. Chase Cam (Key 1): Follows behind and above car, velocity-smoothed
 * 2. Onboard Cam (Key 2): Low cockpit/nose-level POV
 * 3. TV Helicopter Cam (Key 3): High broadcast cinematic angle
 * 4. Top-Down Cam (Key 4): Orthographic overhead view matching 2D dashboard
 * 5. Trackside Cam (Key 5): Automatic handover between static grandstand/corner cameras
 * 6. Free Orbit Cam (Key 6): Interactive mouse/touch orbit & zoom
 */

export class CameraDirector {
  constructor(container, scene) {
    this.container = container;
    this.scene = scene;

    this.aspect = container.clientWidth / container.clientHeight;

    // Perspective Camera (Modes 1, 2, 3, 5, 6)
    this.perspCam = new THREE.PerspectiveCamera(54, this.aspect, 0.5, 1200);

    // Orthographic Camera (Mode 4 Top-Down)
    const frustumSize = 120;
    this.orthoCam = new THREE.OrthographicCamera(
      (-frustumSize * this.aspect) / 2,
      (frustumSize * this.aspect) / 2,
      frustumSize / 2,
      -frustumSize / 2,
      1,
      1000
    );
    this.orthoCam.position.set(0, 300, 0);
    this.orthoCam.up.set(0, 0, -1); // So -Z (North) is Screen Up
    this.orthoCam.lookAt(0, 0, 0);

    this.activeCamera = this.perspCam;
    this.currentMode = 1; // Default: Chase Cam

    // Smoothing state
    this.camPos = new THREE.Vector3(0, 5, 10);
    this.targetPos = new THREE.Vector3(0, 0, 0);
    this.modeChanged = true;

    // Orbit state
    this.isDragging = false;
    this.orbitYaw = 0.0;
    this.orbitPitch = 0.35;
    this.orbitDist = 28.0;
    this.lastMouseX = 0;
    this.lastMouseY = 0;

    // Trackside cameras array
    this.tracksidePosts = [];

    // Camera shake (cosmetic impact feedback)
    this.trauma = 0.0;

    this.setupOrbitListeners();
  }

  addTrauma(amount = 0.5) {
    this.trauma = Math.min(1.0, this.trauma + amount);
  }

  setupOrbitListeners() {
    this.container.addEventListener('mousedown', (e) => {
      if (this.currentMode === 6) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
      }
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
    });

    this.container.addEventListener('mousemove', (e) => {
      if (this.isDragging && this.currentMode === 6) {
        const dx = e.clientX - this.lastMouseX;
        const dy = e.clientY - this.lastMouseY;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;

        this.orbitYaw -= dx * 0.006;
        this.orbitPitch = THREE.MathUtils.clamp(this.orbitPitch + dy * 0.006, 0.05, Math.PI / 2 - 0.05);
      }
    });

    this.container.addEventListener('wheel', (e) => {
      if (this.currentMode === 6) {
        e.preventDefault();
        this.orbitDist = THREE.MathUtils.clamp(this.orbitDist + e.deltaY * 0.03, 6.0, 150.0);
      }
    }, { passive: false });
  }

  setTracksidePosts(centerline) {
    if (!centerline || centerline.length === 0) return;
    this.tracksidePosts = [];
    const count = 8;
    const step = Math.floor(centerline.length / count);

    for (let i = 0; i < centerline.length; i += step) {
      const pt = centerline[i];
      // Stand 22m outside track, 8m elevated
      this.tracksidePosts.push({
        position: new THREE.Vector3(pt[0] + 16, 9.0, -pt[1] + 16),
        waypointIndex: i,
      });
    }
  }

  setMode(mode) {
    if (this.currentMode !== mode) {
      this.modeChanged = true;
    }
    this.currentMode = mode;
    if (mode === 4) {
      this.activeCamera = this.orthoCam;
    } else {
      this.activeCamera = this.perspCam;
    }
  }

  snapNext() {
    this.modeChanged = true;
  }

  update(focusedCar, dt = 0.02) {
    if (!focusedCar) return;

    // Smart billboard badge visibility management
    if (this.scene && this.scene.carsGroup) {
      const activeCamPos = (this.currentMode === 4) ? this.orthoCam.position : this.camPos;
      this.scene.carsGroup.children.forEach((carRoot) => {
        const badge = carRoot.children.find((c) => c.isSprite);
        if (badge) {
          if (carRoot === focusedCar.root) {
            // Never show badge on the car being driven / tracked in close views
            badge.visible = (this.currentMode === 4);
          } else if (this.currentMode === 2) {
            // In cockpit view, hide other cars' badges within 30m so windshield stays completely clear
            const d = carRoot.position.distanceTo(activeCamPos);
            badge.visible = d > 30.0;
          } else if (this.currentMode === 1) {
            // In chase cam, hide other cars' badges within 6m to prevent blocking lens in tight racing
            const d = carRoot.position.distanceTo(activeCamPos);
            badge.visible = d > 6.0;
          } else {
            badge.visible = true;
          }
        }
      });
    }

    const carPos = focusedCar.root.position;
    const carHeading = focusedCar.root.rotation.y;

    // Direction vector of car (car points along -Z when rotY=0)
    const forward = new THREE.Vector3(
      -Math.sin(carHeading),
      0,
      -Math.cos(carHeading)
    );
    const right = new THREE.Vector3(
      Math.cos(carHeading),
      0,
      -Math.sin(carHeading)
    );

    let desiredPos = new THREE.Vector3();
    let desiredTarget = carPos.clone().add(new THREE.Vector3(0, 0.8, 0));

    switch (this.currentMode) {
      case 1: // Chase Cam (Dynamic 3rd-person racing view)
        desiredPos = carPos.clone()
          .sub(forward.clone().multiplyScalar(6.2))
          .add(new THREE.Vector3(0, 2.4, 0));
        desiredTarget = carPos.clone()
          .add(forward.clone().multiplyScalar(10.0))
          .add(new THREE.Vector3(0, 0.8, 0));
        break;

      case 2: // Onboard TV Cockpit / T-Cam (Mounted on airbox / roof looking over nose)
        desiredPos = carPos.clone()
          .sub(forward.clone().multiplyScalar(0.25))
          .add(new THREE.Vector3(0, 1.25, 0));
        desiredTarget = desiredPos.clone()
          .add(forward.clone().multiplyScalar(30.0))
          .add(new THREE.Vector3(0, -0.1, 0));
        break;

      case 3: // TV Helicopter Cam (Elevated Orbit)
        const time = Date.now() * 0.0006;
        const heliDist = 24.0;
        desiredPos = carPos.clone().add(new THREE.Vector3(
          Math.sin(time) * heliDist,
          16.0,
          Math.cos(time) * heliDist
        ));
        desiredTarget = carPos.clone().add(new THREE.Vector3(0, 0.5, 0));
        break;

      case 4: // Top-Down Orthographic
        this.orthoCam.position.set(carPos.x, 250, carPos.z);
        this.orthoCam.lookAt(carPos.x, 0, carPos.z);
        return;

      case 5: // Trackside Handover
        if (this.tracksidePosts.length > 0) {
          // Find closest trackside camera
          let bestPost = this.tracksidePosts[0];
          let bestDist = Infinity;
          for (const post of this.tracksidePosts) {
            const d = post.position.distanceTo(carPos);
            if (d < bestDist) {
              bestDist = d;
              bestPost = post;
            }
          }
          desiredPos = bestPost.position.clone();
          desiredTarget = carPos.clone().add(new THREE.Vector3(0, 0.6, 0));

          // Cut instantly on post handover
          if (this.lastTracksidePost !== bestPost) {
            this.camPos.copy(desiredPos);
            this.lastTracksidePost = bestPost;
          }
        } else {
          desiredPos = carPos.clone().add(new THREE.Vector3(15, 10, 15));
        }
        break;

      case 6: // Free Orbit
        const cy = carPos.y + 0.6;
        desiredPos.x = carPos.x + this.orbitDist * Math.cos(this.orbitPitch) * Math.sin(this.orbitYaw);
        desiredPos.y = cy + this.orbitDist * Math.sin(this.orbitPitch);
        desiredPos.z = carPos.z + this.orbitDist * Math.cos(this.orbitPitch) * Math.cos(this.orbitYaw);
        desiredTarget = carPos.clone().add(new THREE.Vector3(0, 0.6, 0));
        break;
    }

    // Smooth camera movement & orientation tracking
    if (this.modeChanged || this.camPos.distanceTo(desiredPos) > 40.0) {
      this.camPos.copy(desiredPos);
      this.targetPos.copy(desiredTarget);
      this.modeChanged = false;
    } else {
      const smoothFactor = (this.currentMode === 2) ? 0.75 : 0.35;
      this.camPos.lerp(desiredPos, smoothFactor);
      this.targetPos.lerp(desiredTarget, smoothFactor);
    }
    this.perspCam.position.copy(this.camPos);

    // Apply cosmetic camera shake
    if (this.trauma > 0.0) {
      const shakeMag = this.trauma * this.trauma * 0.6;
      this.perspCam.position.x += (Math.random() * 2 - 1) * shakeMag;
      this.perspCam.position.y += (Math.random() * 2 - 1) * shakeMag * 0.5;
      this.perspCam.position.z += (Math.random() * 2 - 1) * shakeMag;
      this.trauma = Math.max(0.0, this.trauma - dt * 2.5);
    }

    this.perspCam.lookAt(this.targetPos);
  }

  onResize() {
    this.aspect = this.container.clientWidth / this.container.clientHeight;
    this.perspCam.aspect = this.aspect;
    this.perspCam.updateProjectionMatrix();

    const frustumSize = 120;
    this.orthoCam.left = (-frustumSize * this.aspect) / 2;
    this.orthoCam.right = (frustumSize * this.aspect) / 2;
    this.orthoCam.top = frustumSize / 2;
    this.orthoCam.bottom = -frustumSize / 2;
    this.orthoCam.updateProjectionMatrix();
  }
}
