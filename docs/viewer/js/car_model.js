/**
 * docs/viewer/js/car_model.js - Procedural F1 Car Model & Dynamic State
 *
 * Implements:
 * - Low-poly F1-style chassis, nose, wings, halo, cockpit & driver helmet
 * - Real dimensions from Gym CarRacing fixtures (Length: 5.0m, Width: 2.4m, Wheelbase: 3.24m)
 * - Front wheel steering & rotational wheel spin
 * - Brake light emissive glow
 * - 3-Letter Driver Code billboard badge
 * - Subtle cosmetic body pitch (braking/acceleration) and roll (cornering)
 */

export class F1Car {
  constructor(driverMeta) {
    this.meta = driverMeta;
    this.id = driverMeta.id;
    this.code = driverMeta.code;
    this.colorHex = driverMeta.color || '#00E5FF';
    this.teamColor = new THREE.Color(this.colorHex);

    this.root = new THREE.Group();
    this.carBody = new THREE.Group();
    this.root.add(this.carBody);

    this.frontLeftWheel = null;
    this.frontRightWheel = null;
    this.rearLeftWheel = null;
    this.rearRightWheel = null;
    this.brakeLightMesh = null;

    this.buildModel();
    this.buildBillboardBadge();

    // Wheel spin angle accumulator
    this.wheelSpinAngle = 0.0;
  }

  buildModel() {
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

    // 1. Main Monocoque / Cockpit Tub (Z is along car length; front is -Z)
    const tubGeo = new THREE.BoxGeometry(1.3, 0.55, 2.6);
    const tub = new THREE.Mesh(tubGeo, liveryMat);
    tub.position.set(0, 0.45, 0.1);
    tub.castShadow = true;
    tub.receiveShadow = true;
    this.carBody.add(tub);

    // 2. Tapered Nose Cone (front along -Z)
    const noseGeo = new THREE.ConeGeometry(0.55, 1.8, 6);
    noseGeo.rotateX(-Math.PI / 2);
    const nose = new THREE.Mesh(noseGeo, liveryMat);
    nose.position.set(0, 0.36, -1.9);
    nose.scale.set(1.1, 0.65, 1.0);
    nose.castShadow = true;
    this.carBody.add(nose);

    // 3. Front Wing & Endplates
    const fwGeo = new THREE.BoxGeometry(2.3, 0.08, 0.6);
    const frontWing = new THREE.Mesh(fwGeo, carbonMat);
    frontWing.position.set(0, 0.2, -2.4);
    frontWing.castShadow = true;
    this.carBody.add(frontWing);

    // Endplates
    const epGeo = new THREE.BoxGeometry(0.06, 0.35, 0.65);
    const leftEp = new THREE.Mesh(epGeo, liveryMat);
    leftEp.position.set(-1.15, 0.32, -2.4);
    const rightEp = new THREE.Mesh(epGeo, liveryMat);
    rightEp.position.set(1.15, 0.32, -2.4);
    this.carBody.add(leftEp);
    this.carBody.add(rightEp);

    // 4. Rear Wing
    const rwGeo = new THREE.BoxGeometry(1.9, 0.1, 0.5);
    const rearWing = new THREE.Mesh(rwGeo, carbonMat);
    rearWing.position.set(0, 1.1, 2.2);
    rearWing.castShadow = true;
    this.carBody.add(rearWing);

    // Rear Endplates
    const repGeo = new THREE.BoxGeometry(0.06, 0.6, 0.6);
    const leftRep = new THREE.Mesh(repGeo, liveryMat);
    leftRep.position.set(-0.95, 0.95, 2.2);
    const rightRep = new THREE.Mesh(repGeo, liveryMat);
    rightRep.position.set(0.95, 0.95, 2.2);
    this.carBody.add(leftRep);
    this.carBody.add(rightRep);

    // 5. Halo Safety Structure
    const haloCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.72, -0.6),
      new THREE.Vector3(-0.35, 0.85, 0.0),
      new THREE.Vector3(0, 0.88, 0.4),
      new THREE.Vector3(0.35, 0.85, 0.0),
      new THREE.Vector3(0, 0.72, -0.6),
    ]);
    const haloGeo = new THREE.TubeGeometry(haloCurve, 20, 0.06, 8, false);
    const halo = new THREE.Mesh(haloGeo, carbonMat);
    this.carBody.add(halo);

    // 6. Driver's Helmet
    const helmetGeo = new THREE.SphereGeometry(0.24, 16, 16);
    const helmetMat = new THREE.MeshStandardMaterial({ color: this.teamColor, roughness: 0.2 });
    const helmet = new THREE.Mesh(helmetGeo, helmetMat);
    helmet.position.set(0, 0.72, 0.1);
    this.carBody.add(helmet);

    // Visor
    const visorGeo = new THREE.BoxGeometry(0.22, 0.08, 0.15);
    const visorMat = new THREE.MeshBasicMaterial({ color: 0x111111 });
    const visor = new THREE.Mesh(visorGeo, visorMat);
    visor.position.set(0, 0.72, -0.12);
    this.carBody.add(visor);

    // 7. Red Rain / Brake Light at Rear
    const blGeo = new THREE.BoxGeometry(0.3, 0.18, 0.1);
    this.blMat = new THREE.MeshStandardMaterial({
      color: 0x330000,
      emissive: 0x000000,
      roughness: 0.4,
    });
    this.brakeLightMesh = new THREE.Mesh(blGeo, this.blMat);
    this.brakeLightMesh.position.set(0, 0.45, 2.45);
    this.carBody.add(this.brakeLightMesh);

    // 8. Wheels (Wheel radius = 0.54m, Wheel width = 0.56m, Wheelbase = 3.24m)
    // Box2D WHEELPOS: [(-1.10, +1.60), (+1.10, +1.60), (-1.10, -1.64), (+1.10, -1.64)]
    const wheelR = 0.54;
    const wheelW = 0.52;

    const makeWheel = (isFront) => {
      const wheelGroup = new THREE.Group();
      // Outer tire
      const tGeo = new THREE.CylinderGeometry(wheelR, wheelR, wheelW, 20);
      tGeo.rotateZ(Math.PI / 2);
      const tire = new THREE.Mesh(tGeo, tireMat);
      tire.castShadow = true;
      wheelGroup.add(tire);

      // Rim
      const rGeo = new THREE.CylinderGeometry(wheelR * 0.55, wheelR * 0.55, wheelW + 0.02, 12);
      rGeo.rotateZ(Math.PI / 2);
      const rim = new THREE.Mesh(rGeo, rimMat);
      wheelGroup.add(rim);

      return wheelGroup;
    };

    // Front Wheels (Steerable groups)
    this.flSteer = new THREE.Group();
    this.flSteer.position.set(-1.15, wheelR, -1.60);
    this.frontLeftWheel = makeWheel(true);
    this.flSteer.add(this.frontLeftWheel);
    this.carBody.add(this.flSteer);

    this.frSteer = new THREE.Group();
    this.frSteer.position.set(1.15, wheelR, -1.60);
    this.frontRightWheel = makeWheel(true);
    this.frSteer.add(this.frontRightWheel);
    this.carBody.add(this.frSteer);

    // Rear Wheels
    this.rearLeftWheel = makeWheel(false);
    this.rearLeftWheel.position.set(-1.15, wheelR, 1.64);
    this.carBody.add(this.rearLeftWheel);

    this.rearRightWheel = makeWheel(false);
    this.rearRightWheel.position.set(1.15, wheelR, 1.64);
    this.carBody.add(this.rearRightWheel);
  }

  buildBillboardBadge() {
    // 3-Letter Driver Code Sprite above car
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    // Rounded background pill
    ctx.fillStyle = 'rgba(12, 16, 26, 0.85)';
    ctx.strokeStyle = this.colorHex;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.roundRect(8, 8, 112, 48, 12);
    ctx.fill();
    ctx.stroke();

    // Text
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

  update(state, dt = 0.02) {
    if (!state) return;

    // 1. Position mapping: Sim (x, y) -> 3D (x, 0, -y)
    this.root.position.x = state.x;
    this.root.position.y = 0;
    this.root.position.z = -state.y;

    // 2. Heading rotation: rotation.y = hull.angle
    // Note: Since car mesh points along -Z, rotation.y = hull_angle
    const hullAngle = state.hull_angle !== undefined ? state.hull_angle : (state.heading - Math.PI / 2);
    this.root.rotation.y = hullAngle;

    // 3. Front Wheel Steering Angle
    const steerRad = (state.steer || 0.0) * 0.42;
    this.flSteer.rotation.y = steerRad;
    this.frSteer.rotation.y = steerRad;

    // 4. Wheel Spin Rotation from Speed
    const speed = state.speed || 0.0;
    const wheelR = 0.54;
    const angularDist = (speed * dt) / wheelR;
    this.wheelSpinAngle -= angularDist;

    this.frontLeftWheel.rotation.x = this.wheelSpinAngle;
    this.frontRightWheel.rotation.x = this.wheelSpinAngle;
    this.rearLeftWheel.rotation.x = this.wheelSpinAngle;
    this.rearRightWheel.rotation.x = this.wheelSpinAngle;

    // 5. Brake Light Glow
    const isBraking = (state.brake || 0.0) > 0.05;
    if (isBraking) {
      this.blMat.color.setHex(0xff1744);
      this.blMat.emissive.setHex(0xff0033);
      this.blMat.emissiveIntensity = 1.8;
    } else {
      this.blMat.color.setHex(0x330000);
      this.blMat.emissive.setHex(0x000000);
      this.blMat.emissiveIntensity = 0.0;
    }

    // 6. Cosmetic Pitch & Roll Dynamics
    // Pitch from longitudinal acceleration: braking pitches nose DOWN (negative X rotation)
    const pitchAmount = (state.brake || 0.0) * -0.045 + (state.gas || 0.0) * 0.025;
    // Roll from lateral acceleration / steering * speed: rolling into corner
    const lateralG = state.lateral_g || (state.steer * speed * 0.015);
    const rollAmount = THREE.MathUtils.clamp(lateralG * -0.06, -0.08, 0.08);

    this.carBody.rotation.x = THREE.MathUtils.lerp(this.carBody.rotation.x, pitchAmount, 0.2);
    this.carBody.rotation.z = THREE.MathUtils.lerp(this.carBody.rotation.z, rollAmount, 0.2);
  }
}
