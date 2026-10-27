/**
 * docs/viewer/js/car_model.js - Procedural F1 Car Model, Custom 3D Model Loader & Calibration
 *
 * Implements:
 * - Low-poly procedural F1-style chassis, nose, wings, halo, cockpit & driver helmet
 * - Real dimensions from Gym CarRacing fixtures (Length: 5.0m, Width: 2.4m, Wheelbase: 3.24m)
 * - Custom glTF 3D model loader with car.json configuration and automatic procedural fallback
 * - Physics wireframe overlay (5.0m x 2.4m hull box & 4 wheel footprints) for calibration mode
 * - Front wheel steering & rotational wheel spin (procedural + glTF wheel nodes)
 * - Brake light emissive glow
 * - 3-Letter Driver Code billboard badge
 * - Subtle cosmetic body pitch (braking/acceleration) and roll (cornering)
 */

export class F1Car {
  constructor(driverMeta, initialCustomModel = null) {
    this.meta = driverMeta;
    this.id = driverMeta.id;
    this.code = driverMeta.code;
    this.colorHex = driverMeta.color || '#00E5FF';
    this.teamColor = new THREE.Color(this.colorHex);

    this.root = new THREE.Group();
    this.carBody = new THREE.Group();
    this.root.add(this.carBody);

    // Groups for procedural vs custom glTF models
    this.proceduralGroup = new THREE.Group();
    this.customModelGroup = new THREE.Group();
    this.carBody.add(this.proceduralGroup);
    this.carBody.add(this.customModelGroup);

    this.isCustomModel = false;
    this.customConfig = null;
    this.customWheelNodes = {}; // { fl, fr, rl, rr }
    this.customBrakeMaterials = [];

    // Physics wireframe overlay for calibration mode
    this.wireframeOverlay = new THREE.Group();
    this.wireframeOverlay.visible = false;
    this.root.add(this.wireframeOverlay);
    this.buildPhysicsWireframeOverlay();

    // Procedural wheels & steering
    this.frontLeftWheel = null;
    this.frontRightWheel = null;
    this.rearLeftWheel = null;
    this.rearRightWheel = null;
    this.flSteer = null;
    this.frSteer = null;
    this.brakeLightMesh = null;

    this.buildProceduralModel();
    this.buildBillboardBadge();

    // Wheel spin angle accumulator
    this.wheelSpinAngle = 0.0;

    // Load custom model if requested or configured
    const targetModel = initialCustomModel || driverMeta.custom_model || driverMeta.model_name;
    if (targetModel && targetModel !== 'procedural') {
      this.loadCustomModel(targetModel);
    }
  }

  buildPhysicsWireframeOverlay() {
    // 1. Box2D Physics Hull Wireframe: Length 5.0m, Width 2.4m, Height 1.0m
    const hullGeo = new THREE.BoxGeometry(2.4, 1.0, 5.0);
    const hullWire = new THREE.WireframeGeometry(hullGeo);
    const hullLine = new THREE.LineSegments(
      hullWire,
      new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 2, transparent: true, opacity: 0.85 })
    );
    hullLine.position.set(0, 0.5, 0);
    this.wireframeOverlay.add(hullLine);

    // 2. Wheel Contact Footprints: 4 patches at (+-1.15m, +-1.62m)
    const fpGeo = new THREE.BoxGeometry(0.38, 0.08, 0.82);
    const fpWire = new THREE.WireframeGeometry(fpGeo);
    const fpMat = new THREE.LineBasicMaterial({ color: 0x00e5ff, linewidth: 2 });

    const wheelPositions = [
      [-1.15, 0.04, -1.60], // FL
      [1.15, 0.04, -1.60],  // FR
      [-1.15, 0.04, 1.64],  // RL
      [1.15, 0.04, 1.64],   // RR
    ];

    for (const [wx, wy, wz] of wheelPositions) {
      const fpLine = new THREE.LineSegments(fpWire, fpMat);
      fpLine.position.set(wx, wy, wz);
      this.wireframeOverlay.add(fpLine);
    }
  }

  togglePhysicsWireframe(show) {
    if (this.wireframeOverlay) {
      this.wireframeOverlay.visible = !!show;
    }
  }

  buildProceduralModel() {
    const liveryMat = new THREE.MeshStandardMaterial({
      color: this.teamColor,
      roughness: 0.25,
      metalness: 0.65,
    });

    const carbonMat = new THREE.MeshStandardMaterial({
      color: 0x141820,
      roughness: 0.5,
      metalness: 0.8,
    });

    const tireMat = new THREE.MeshStandardMaterial({
      color: 0x181a1f,
      roughness: 0.85,
      metalness: 0.1,
    });

    const rimMat = new THREE.MeshStandardMaterial({
      color: 0x8a92a5,
      roughness: 0.3,
      metalness: 0.9,
    });

    // 1. Main Monocoque Tub (Z is along car length; front is -Z)
    const tubGeo = new THREE.BoxGeometry(1.3, 0.55, 2.6);
    const tub = new THREE.Mesh(tubGeo, liveryMat);
    tub.position.set(0, 0.45, 0.1);
    tub.castShadow = true;
    tub.receiveShadow = true;
    this.proceduralGroup.add(tub);

    // 2. Tapered Nose Cone
    const noseGeo = new THREE.ConeGeometry(0.55, 1.8, 6);
    noseGeo.rotateX(-Math.PI / 2);
    const nose = new THREE.Mesh(noseGeo, liveryMat);
    nose.position.set(0, 0.36, -1.9);
    nose.scale.set(1.1, 0.65, 1.0);
    nose.castShadow = true;
    this.proceduralGroup.add(nose);

    // 3. Front Wing & Endplates
    const fwGeo = new THREE.BoxGeometry(2.3, 0.08, 0.6);
    const frontWing = new THREE.Mesh(fwGeo, carbonMat);
    frontWing.position.set(0, 0.2, -2.4);
    frontWing.castShadow = true;
    this.proceduralGroup.add(frontWing);

    const epGeo = new THREE.BoxGeometry(0.06, 0.35, 0.65);
    const leftEp = new THREE.Mesh(epGeo, liveryMat);
    leftEp.position.set(-1.15, 0.32, -2.4);
    const rightEp = new THREE.Mesh(epGeo, liveryMat);
    rightEp.position.set(1.15, 0.32, -2.4);
    this.proceduralGroup.add(leftEp);
    this.proceduralGroup.add(rightEp);

    // 4. Rear Wing & Endplates
    const rwGeo = new THREE.BoxGeometry(1.9, 0.1, 0.5);
    const rearWing = new THREE.Mesh(rwGeo, carbonMat);
    rearWing.position.set(0, 1.1, 2.2);
    rearWing.castShadow = true;
    this.proceduralGroup.add(rearWing);

    const repGeo = new THREE.BoxGeometry(0.06, 0.6, 0.6);
    const leftRep = new THREE.Mesh(repGeo, liveryMat);
    leftRep.position.set(-0.95, 0.95, 2.2);
    const rightRep = new THREE.Mesh(repGeo, liveryMat);
    rightRep.position.set(0.95, 0.95, 2.2);
    this.proceduralGroup.add(leftRep);
    this.proceduralGroup.add(rightRep);

    // 5. Halo Structure
    const haloCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.72, -0.6),
      new THREE.Vector3(-0.35, 0.85, 0.0),
      new THREE.Vector3(0, 0.88, 0.4),
      new THREE.Vector3(0.35, 0.85, 0.0),
      new THREE.Vector3(0, 0.72, -0.6),
    ]);
    const haloGeo = new THREE.TubeGeometry(haloCurve, 20, 0.06, 8, false);
    const halo = new THREE.Mesh(haloGeo, carbonMat);
    this.proceduralGroup.add(halo);

    // 6. Driver Helmet & Visor
    const helmetGeo = new THREE.SphereGeometry(0.24, 16, 16);
    const helmetMat = new THREE.MeshStandardMaterial({ color: this.teamColor, roughness: 0.2 });
    const helmet = new THREE.Mesh(helmetGeo, helmetMat);
    helmet.position.set(0, 0.72, 0.1);
    this.proceduralGroup.add(helmet);

    const visorGeo = new THREE.BoxGeometry(0.22, 0.08, 0.15);
    const visorMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    const visor = new THREE.Mesh(visorGeo, visorMat);
    visor.position.set(0, 0.72, -0.12);
    this.proceduralGroup.add(visor);

    // 7. Brake Light
    const blGeo = new THREE.BoxGeometry(0.3, 0.18, 0.1);
    this.blMat = new THREE.MeshStandardMaterial({
      color: 0x330000,
      emissive: 0x000000,
      roughness: 0.4,
    });
    this.brakeLightMesh = new THREE.Mesh(blGeo, this.blMat);
    this.brakeLightMesh.position.set(0, 0.45, 2.45);
    this.proceduralGroup.add(this.brakeLightMesh);

    // 8. Wheels
    const wheelR = 0.54;
    const wheelW = 0.52;

    const makeWheel = () => {
      const wheelGroup = new THREE.Group();
      const tGeo = new THREE.CylinderGeometry(wheelR, wheelR, wheelW, 20);
      tGeo.rotateZ(Math.PI / 2);
      const tire = new THREE.Mesh(tGeo, tireMat);
      tire.castShadow = true;
      wheelGroup.add(tire);

      const rGeo = new THREE.CylinderGeometry(wheelR * 0.55, wheelR * 0.55, wheelW + 0.02, 12);
      rGeo.rotateZ(Math.PI / 2);
      const rim = new THREE.Mesh(rGeo, rimMat);
      wheelGroup.add(rim);
      return wheelGroup;
    };

    // Front Steerable Wheels
    this.flSteer = new THREE.Group();
    this.flSteer.position.set(-1.15, wheelR, -1.60);
    this.frontLeftWheel = makeWheel();
    this.flSteer.add(this.frontLeftWheel);
    this.proceduralGroup.add(this.flSteer);

    this.frSteer = new THREE.Group();
    this.frSteer.position.set(1.15, wheelR, -1.60);
    this.frontRightWheel = makeWheel();
    this.frSteer.add(this.frontRightWheel);
    this.proceduralGroup.add(this.frSteer);

    // Rear Wheels
    this.rearLeftWheel = makeWheel();
    this.rearLeftWheel.position.set(-1.15, wheelR, 1.64);
    this.proceduralGroup.add(this.rearLeftWheel);

    this.rearRightWheel = makeWheel();
    this.rearRightWheel.position.set(1.15, wheelR, 1.64);
    this.proceduralGroup.add(this.rearRightWheel);
  }

  buildBillboardBadge() {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = 'rgba(12, 16, 26, 0.85)';
    ctx.strokeStyle = this.colorHex;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(8, 8, 112, 48, 12);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 30px "BigShoulders", sans-serif, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this.code, 64, 32);

    const texture = new THREE.CanvasTexture(canvas);
    const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false });
    this.badgeSprite = new THREE.Sprite(spriteMat);
    this.badgeSprite.scale.set(3.2, 1.6, 1.0);
    this.badgeSprite.position.set(0, 2.7, 0);
    this.root.add(this.badgeSprite);
  }

  /**
   * Load custom 3D car model from assets/private/cars/<name>/car.json
   * Falls back to procedural car with a console warning on any failure.
   */
  async loadCustomModel(carKey, customJsonPath = null) {
    if (carKey === 'procedural') {
      this.proceduralGroup.visible = true;
      this.customModelGroup.visible = false;
      this.isCustomModel = false;
      return;
    }

    const basePath = customJsonPath ? customJsonPath.substring(0, customJsonPath.lastIndexOf('/')) : `../../assets/private/cars/${carKey}`;
    const jsonUrl = customJsonPath || `${basePath}/car.json`;

    try {
      const res = await fetch(jsonUrl);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status} fetching ${jsonUrl}`);
      }
      const cfg = await res.json();

      // Validate configuration schema
      if (!cfg.file) {
        throw new Error("Invalid car.json: missing 'file' property");
      }
      if (cfg.scale === undefined || cfg.scale === null) {
        throw new Error("Invalid car.json: missing 'scale' property");
      }

      if (typeof THREE.GLTFLoader === 'undefined') {
        throw new Error("THREE.GLTFLoader is not loaded");
      }

      const loader = new THREE.GLTFLoader();
      const modelUrl = `${basePath}/${cfg.file}`;

      loader.load(
        modelUrl,
        (gltf) => {
          // Clear any previous custom model
          while (this.customModelGroup.children.length > 0) {
            this.customModelGroup.remove(this.customModelGroup.children[0]);
          }

          const model = gltf.scene || gltf.scenes[0];
          this.customConfig = cfg;
          this.customModel = model;

          // Apply initial transformations
          this.applyModelTransform(cfg);

          // Find and tint materials
          this.customBrakeMaterials = [];
          if (cfg.tint_materials || cfg.brake_light_material) {
            const tintList = (cfg.tint_materials || []).map((s) => s.toLowerCase());
            const brakeMatName = (cfg.brake_light_material || '').toLowerCase();

            model.traverse((node) => {
              if (node.isMesh && node.material) {
                const mats = Array.isArray(node.material) ? node.material : [node.material];
                mats.forEach((mat) => {
                  const mName = (mat.name || '').toLowerCase();
                  if (tintList.some((tm) => mName.includes(tm))) {
                    mat.color = this.teamColor;
                    mat.needsUpdate = true;
                  }
                  if (brakeMatName && mName.includes(brakeMatName)) {
                    this.customBrakeMaterials.push(mat);
                  }
                });
              }
            });
          }

          // Resolve custom wheel nodes
          this.customWheelNodes = {};
          if (cfg.wheel_nodes) {
            for (const [posKey, nodeName] of Object.entries(cfg.wheel_nodes)) {
              let foundNode = null;
              model.traverse((child) => {
                if (child.name === nodeName || child.name.toLowerCase() === nodeName.toLowerCase()) {
                  foundNode = child;
                }
              });
              if (foundNode) {
                this.customWheelNodes[posKey] = foundNode;
              }
            }
          }

          this.customModelGroup.add(model);
          this.proceduralGroup.visible = false;
          this.customModelGroup.visible = true;
          this.isCustomModel = true;
          console.log(`[F1Car] Successfully loaded custom model '${carKey}' for ${this.code}`);
        },
        undefined,
        (err) => {
          console.warn(`[F1Car] Error loading glTF model from ${modelUrl}:`, err, `- Falling back to procedural car.`);
          this.proceduralGroup.visible = true;
          this.customModelGroup.visible = false;
          this.isCustomModel = false;
        }
      );
    } catch (err) {
      console.warn(`[F1Car] Could not load car.json for '${carKey}':`, err.message, `- Falling back to procedural car.`);
      this.proceduralGroup.visible = true;
      this.customModelGroup.visible = false;
      this.isCustomModel = false;
    }
  }

  applyModelTransform(cfg) {
    if (!this.customModel) return;

    // Scale
    let sX = 1.0, sY = 1.0, sZ = 1.0;
    if (typeof cfg.scale === 'number') {
      sX = sY = sZ = cfg.scale;
    } else if (Array.isArray(cfg.scale)) {
      sX = cfg.scale[0] || 1.0;
      sY = cfg.scale[1] || sX;
      sZ = cfg.scale[2] || sX;
    }
    this.customModel.scale.set(sX, sY, sZ);

    // Offset
    const ox = (cfg.offset && cfg.offset[0]) || 0.0;
    const oy = (cfg.offset && cfg.offset[1]) || 0.0;
    const oz = (cfg.offset && cfg.offset[2]) || 0.0;
    this.customModel.position.set(ox, oy, oz);

    // Rotation / Axis Alignment
    this.customModel.rotation.set(0, 0, 0);

    // Up axis correction
    if (cfg.up === '+Z') {
      this.customModel.rotation.x = -Math.PI / 2;
    } else if (cfg.up === '-Z') {
      this.customModel.rotation.x = Math.PI / 2;
    }

    // Forward axis correction
    if (cfg.forward === '+X') {
      this.customModel.rotation.y += Math.PI / 2;
    } else if (cfg.forward === '-X') {
      this.customModel.rotation.y -= Math.PI / 2;
    } else if (cfg.forward === '+Z') {
      // In Three.js car points along -Z, so +Z forward rotates 180 deg
      this.customModel.rotation.y += Math.PI;
    }

    // Explicit Euler rotations if provided in calibration
    if (cfg.rot_x) this.customModel.rotation.x += cfg.rot_x;
    if (cfg.rot_y) this.customModel.rotation.y += cfg.rot_y;
    if (cfg.rot_z) this.customModel.rotation.z += cfg.rot_z;
  }

  applyCalibration(calibData) {
    if (!this.customConfig) {
      this.customConfig = {
        name: this.meta.code,
        scale: 1.0,
        offset: [0, 0, 0],
      };
    }

    if (calibData.scale !== undefined) this.customConfig.scale = parseFloat(calibData.scale);
    if (calibData.offsetX !== undefined || calibData.offsetY !== undefined || calibData.offsetZ !== undefined) {
      this.customConfig.offset = [
        parseFloat(calibData.offsetX || 0),
        parseFloat(calibData.offsetY || 0),
        parseFloat(calibData.offsetZ || 0),
      ];
    }
    if (calibData.rotX !== undefined) this.customConfig.rot_x = parseFloat(calibData.rotX) * (Math.PI / 180);
    if (calibData.rotY !== undefined) this.customConfig.rot_y = parseFloat(calibData.rotY) * (Math.PI / 180);
    if (calibData.rotZ !== undefined) this.customConfig.rot_z = parseFloat(calibData.rotZ) * (Math.PI / 180);

    this.applyModelTransform(this.customConfig);
    return this.customConfig;
  }

  update(arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, dt = 0.02) {
    if (arg1 === undefined || arg1 === null) return;

    let x = 0, y = 0, hullAngle = 0, steer = 0, gas = 0, brake = 0, speed = 0, latG = 0;

    if (typeof arg1 === 'object') {
      const state = arg1;
      x = Number(state.x) || 0.0;
      y = Number(state.y) || 0.0;
      hullAngle = state.hull_angle !== undefined 
        ? Number(state.hull_angle) 
        : ((state.heading !== undefined ? Number(state.heading) - Math.PI / 2 : 0.0));
      steer = Number(state.steer) || 0.0;
      gas = Number(state.gas) || 0.0;
      brake = Number(state.brake) || 0.0;
      speed = Number(state.speed) || 0.0;
      latG = state.lateral_g !== undefined ? Number(state.lateral_g) : (steer * speed * 0.015);
      if (typeof arg2 === 'number') dt = arg2;
    } else {
      x = Number(arg1) || 0.0;
      y = Number(arg2) || 0.0;
      hullAngle = arg3 !== undefined ? Number(arg3) : 0.0;
      steer = Number(arg4) || 0.0;
      gas = Number(arg5) || 0.0;
      brake = Number(arg6) || 0.0;
      speed = Number(arg7) || 0.0;
      latG = arg8 !== undefined ? Number(arg8) : (steer * speed * 0.015);
      if (typeof dt !== 'number') dt = 0.02;
    }

    // 1. Position mapping: Sim (x, y) -> 3D (x, 0, -y)
    this.root.position.x = x;
    this.root.position.y = 0;
    this.root.position.z = -y;

    // 2. Heading rotation: rotation.y = hull_angle
    this.root.rotation.y = hullAngle;

    // 3. Front Wheel Steering Angle
    const steerRad = steer * 0.42;

    // Procedural front wheels
    if (this.flSteer && this.frSteer) {
      this.flSteer.rotation.y = steerRad;
      this.frSteer.rotation.y = steerRad;
    }

    // Custom model front wheels
    if (this.customWheelNodes.fl) {
      this.customWheelNodes.fl.rotation.y = steerRad;
    }
    if (this.customWheelNodes.fr) {
      this.customWheelNodes.fr.rotation.y = steerRad;
    }

    // 4. Wheel Spin Rotation from Speed
    const wheelR = 0.54;
    const angularDist = (speed * dt) / wheelR;
    this.wheelSpinAngle -= angularDist;

    // Procedural wheels spin
    if (this.frontLeftWheel) this.frontLeftWheel.rotation.x = this.wheelSpinAngle;
    if (this.frontRightWheel) this.frontRightWheel.rotation.x = this.wheelSpinAngle;
    if (this.rearLeftWheel) this.rearLeftWheel.rotation.x = this.wheelSpinAngle;
    if (this.rearRightWheel) this.rearRightWheel.rotation.x = this.wheelSpinAngle;

    // Custom wheels spin
    for (const [wKey, node] of Object.entries(this.customWheelNodes)) {
      if (node && node.rotation) {
        node.rotation.x = this.wheelSpinAngle;
      }
    }

    // 5. Brake Light Glow
    const isBraking = brake > 0.05;
    if (this.blMat) {
      if (isBraking) {
        this.blMat.color.setHex(0xff1744);
        this.blMat.emissive.setHex(0xff0033);
        this.blMat.emissiveIntensity = 1.8;
      } else {
        this.blMat.color.setHex(0x330000);
        this.blMat.emissive.setHex(0x000000);
        this.blMat.emissiveIntensity = 0.0;
      }
    }

    // Custom brake light materials
    for (const mat of this.customBrakeMaterials) {
      if (isBraking) {
        mat.emissive = new THREE.Color(0xff1744);
        mat.emissiveIntensity = 2.0;
      } else {
        mat.emissive = new THREE.Color(0x000000);
        mat.emissiveIntensity = 0.0;
      }
    }

    // 6. Cosmetic Pitch & Roll Dynamics
    const pitchAmount = brake * -0.045 + gas * 0.025;
    const lateralG = latG;
    const rollAmount = THREE.MathUtils.clamp(lateralG * -0.06, -0.08, 0.08);

    this.carBody.rotation.x = THREE.MathUtils.lerp(this.carBody.rotation.x, pitchAmount, 0.2);
    this.carBody.rotation.z = THREE.MathUtils.lerp(this.carBody.rotation.z, rollAmount, 0.2);
  }
}
