/**
 * docs/viewer/js/scene.js - Three.js Scene, Track Geometry, Floodlighting & Start Gantry
 */

export class TrackScene {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x060810);
    this.scene.fog = new THREE.FogExp2(0x060810, 0.0035);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: true });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    container.appendChild(this.renderer.domElement);

    // Groups
    this.trackGroup = new THREE.Group();
    this.environmentGroup = new THREE.Group();
    this.gantryGroup = new THREE.Group();
    this.carsGroup = new THREE.Group();

    this.scene.add(this.trackGroup);
    this.scene.add(this.environmentGroup);
    this.scene.add(this.gantryGroup);
    this.scene.add(this.carsGroup);

    this.startLights = [];
    this.sparksList = [];
    this.setupLighting();
    this.setupEnvironment();
  }

  setupLighting() {
    // 1. Hemisphere Light - smooth sky/ground ambient fill for rich PBR car paints
    const hemiLight = new THREE.HemisphereLight(0xddeeff, 0x1a2233, 1.6);
    this.scene.add(hemiLight);

    // 2. Ambient Night Light
    const ambient = new THREE.AmbientLight(0x283850, 0.8);
    this.scene.add(ambient);

    // 3. Moon / Primary Directional Key Light
    const moonLight = new THREE.DirectionalLight(0xbad7ff, 1.8);
    moonLight.position.set(120, 250, 150);
    moonLight.castShadow = true;
    moonLight.shadow.mapSize.width = 2048;
    moonLight.shadow.mapSize.height = 2048;
    moonLight.shadow.camera.near = 10;
    moonLight.shadow.camera.far = 600;
    const d = 220;
    moonLight.shadow.camera.left = -d;
    moonLight.shadow.camera.right = d;
    moonLight.shadow.camera.top = d;
    moonLight.shadow.camera.bottom = -d;
    moonLight.shadow.bias = -0.0005;
    this.scene.add(moonLight);
    this.mainLight = moonLight;

    // 4. Secondary Fill Directional Light (illuminates shadow side of cars)
    const fillLight = new THREE.DirectionalLight(0x6688aa, 0.9);
    fillLight.position.set(-150, 180, -150);
    this.scene.add(fillLight);
  }

  setupEnvironment() {
    // Ground plane (dark grass with perimeter fade)
    const groundGeo = new THREE.PlaneGeometry(1200, 1200, 64, 64);
    groundGeo.rotateX(-Math.PI / 2);

    // Procedural grass texture
    const grassCanvas = document.createElement('canvas');
    grassCanvas.width = 512;
    grassCanvas.height = 512;
    const ctx = grassCanvas.getContext('2d');
    ctx.fillStyle = '#0a160d';
    ctx.fillRect(0, 0, 512, 512);
    for (let i = 0; i < 4000; i++) {
      const x = Math.random() * 512;
      const y = Math.random() * 512;
      const r = Math.random() * 2 + 1;
      ctx.fillStyle = Math.random() > 0.5 ? '#102416' : '#07120a';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
    const grassTex = new THREE.CanvasTexture(grassCanvas);
    grassTex.wrapS = THREE.RepeatWrapping;
    grassTex.wrapT = THREE.RepeatWrapping;
    grassTex.repeat.set(60, 60);

    const groundMat = new THREE.MeshStandardMaterial({
      map: grassTex,
      roughness: 0.95,
      metalness: 0.05,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.position.y = -0.05;
    ground.receiveShadow = true;
    this.environmentGroup.add(ground);

    // Starfield Sky Dome
    const starsGeo = new THREE.BufferGeometry();
    const starCount = 1800;
    const starPositions = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const radius = 550 + Math.random() * 150;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 0.85); // Upper hemisphere
      starPositions[i * 3 + 0] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.cos(phi);
      starPositions[i * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
    }
    starsGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starsMat = new THREE.PointsMaterial({ color: 0xddeeff, size: 2.0, transparent: true, opacity: 0.75 });
    const starField = new THREE.Points(starsGeo, starsMat);
    this.environmentGroup.add(starField);
  }

  createProceduralAsphaltTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d');

    // Asphalt dark charcoal base
    ctx.fillStyle = '#282c35';
    ctx.fillRect(0, 0, 512, 512);

    // Aggregate noise
    const imgData = ctx.getImageData(0, 0, 512, 512);
    const data = imgData.data;
    for (let i = 0; i < data.length; i += 4) {
      const noise = (Math.random() - 0.5) * 22;
      data[i] = Math.max(0, Math.min(255, data[i] + noise));
      data[i + 1] = Math.max(0, Math.min(255, data[i + 1] + noise));
      data[i + 2] = Math.max(0, Math.min(255, data[i + 2] + noise));
    }
    ctx.putImageData(imgData, 0, 0);

    // Faint dark racing grooves (rubbered in)
    ctx.fillStyle = 'rgba(15, 18, 22, 0.45)';
    ctx.fillRect(130, 0, 100, 512);
    ctx.fillRect(282, 0, 100, 512);

    // Edge lines (crisp white borders)
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.fillRect(0, 0, 18, 512);
    ctx.fillRect(494, 0, 18, 512);

    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(1, 1);
    return tex;
  }

  buildTrack(trackData) {
    // Clear existing track meshes
    while (this.trackGroup.children.length > 0) {
      this.trackGroup.remove(this.trackGroup.children[0]);
    }

    const polygons = trackData.polygons || [];
    const asphaltTex = this.createProceduralAsphaltTexture();

    // Road ribbon mesh from quads
    const vertices = [];
    const uvs = [];
    const indices = [];

    polygons.forEach((poly, polyIdx) => {
      // poly is 4 points: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
      // Sim to 3D mapping: (x, y=0.04, z=-y)
      const p1 = [poly[0][0], 0.04, -poly[0][1]];
      const p2 = [poly[1][0], 0.04, -poly[1][1]];
      const p3 = [poly[2][0], 0.04, -poly[2][1]];
      const p4 = [poly[3][0], 0.04, -poly[3][1]];

      const baseIdx = vertices.length / 3;

      vertices.push(...p1, ...p2, ...p3, ...p4);
      uvs.push(0, 0, 1, 0, 1, 1, 0, 1);

      // Two triangles per quad with Counter-Clockwise (upward normal) winding
      indices.push(baseIdx, baseIdx + 2, baseIdx + 1);
      indices.push(baseIdx, baseIdx + 3, baseIdx + 2);
    });

    const roadGeo = new THREE.BufferGeometry();
    roadGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    roadGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    roadGeo.setIndex(indices);
    roadGeo.computeVertexNormals();

    const roadMat = new THREE.MeshStandardMaterial({
      map: asphaltTex,
      roughness: 0.82,
      metalness: 0.12,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const roadMesh = new THREE.Mesh(roadGeo, roadMat);
    roadMesh.receiveShadow = true;
    this.trackGroup.add(roadMesh);

    // Continuous Left and Right Track Boundary Lines
    this.buildEdgeLines(trackData.left_edge || [], trackData.right_edge || []);

    // Build Single Start / Finish Checkered Line
    this.buildStartFinishLine(trackData.left_edge, trackData.right_edge, trackData.start_finish);

    // Build Staggered Grid Boxes
    this.buildGridBoxes(trackData.grid_boxes || []);

    // Build Red / White Corner Kerbs
    this.buildKerbs(trackData.kerbs || [], trackData.centerline || []);

    // Build Trackside Floodlights
    this.buildFloodlights(trackData.centerline || []);

    // Build Start Gantry with 5 LED Lights
    this.buildStartGantry(trackData.left_edge, trackData.right_edge, trackData.start_finish);
  }

  buildEdgeLines(leftEdge, rightEdge) {
    if (leftEdge.length > 1) {
      const leftPts = leftEdge.map(pt => new THREE.Vector3(pt[0], 0.045, -pt[1]));
      leftPts.push(leftPts[0]); // Loop back
      const leftGeo = new THREE.BufferGeometry().setFromPoints(leftPts);
      const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
      const leftLine = new THREE.Line(leftGeo, edgeMat);
      this.trackGroup.add(leftLine);
    }

    if (rightEdge.length > 1) {
      const rightPts = rightEdge.map(pt => new THREE.Vector3(pt[0], 0.045, -pt[1]));
      rightPts.push(rightPts[0]); // Loop back
      const rightGeo = new THREE.BufferGeometry().setFromPoints(rightPts);
      const edgeMat = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });
      const rightLine = new THREE.Line(rightGeo, edgeMat);
      this.trackGroup.add(rightLine);
    }
  }

  buildStartFinishLine(leftEdge, rightEdge, sfData = null) {
    let x1, z1, x2, z2;
    if (leftEdge && leftEdge.length > 0 && rightEdge && rightEdge.length > 0) {
      x1 = leftEdge[0][0];
      z1 = -leftEdge[0][1];
      x2 = rightEdge[0][0];
      z2 = -rightEdge[0][1];
    } else if (sfData && sfData.p1 && sfData.p2) {
      x1 = sfData.p1[0];
      z1 = -sfData.p1[1];
      x2 = sfData.p2[0];
      z2 = -sfData.p2[1];
    } else {
      return;
    }

    // Checkered line canvas texture
    const checkCanvas = document.createElement('canvas');
    checkCanvas.width = 256;
    checkCanvas.height = 32;
    const ctx = checkCanvas.getContext('2d');

    const numCols = 16;
    const numRows = 2;
    const colW = 256 / numCols;
    const rowH = 32 / numRows;

    for (let r = 0; r < numRows; r++) {
      for (let c = 0; c < numCols; c++) {
        ctx.fillStyle = (r + c) % 2 === 0 ? '#ffffff' : '#111111';
        ctx.fillRect(c * colW, r * rowH, colW, rowH);
      }
    }

    const checkTex = new THREE.CanvasTexture(checkCanvas);
    const midX = (x1 + x2) / 2;
    const midZ = (z1 + z2) / 2;
    const length = Math.hypot(x2 - x1, z2 - z1);
    const angle = Math.atan2(z2 - z1, x2 - x1);

    const sfGeo = new THREE.PlaneGeometry(length, 1.8);
    sfGeo.rotateX(-Math.PI / 2);
    sfGeo.rotateY(-angle);

    const sfMat = new THREE.MeshStandardMaterial({
      map: checkTex,
      roughness: 0.6,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const sfMesh = new THREE.Mesh(sfGeo, sfMat);
    sfMesh.position.set(midX, 0.046, midZ);
    sfMesh.receiveShadow = true;
    this.trackGroup.add(sfMesh);
  }

  buildGridBoxes(gridBoxes) {
    const boxMat = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: false });

    gridBoxes.forEach(g => {
      const corners = g.corners; // 4 corners
      if (!corners || corners.length < 4) return;

      const p1 = new THREE.Vector3(corners[0][0], 0.03, -corners[0][1]);
      const p2 = new THREE.Vector3(corners[1][0], 0.03, -corners[1][1]);
      const p3 = new THREE.Vector3(corners[2][0], 0.03, -corners[2][1]);
      const p4 = new THREE.Vector3(corners[3][0], 0.03, -corners[3][1]);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([p1, p2, p3, p4, p1]);
      const lineMat = new THREE.LineBasicMaterial({ color: 0xf5f7fb, linewidth: 2 });
      const line = new THREE.Line(lineGeo, lineMat);
      this.trackGroup.add(line);

      // Staggered Position Number on Asphalt (e.g. 1..6)
      const numCanvas = document.createElement('canvas');
      numCanvas.width = 64;
      numCanvas.height = 64;
      const ctx = numCanvas.getContext('2d');
      ctx.fillStyle = '#f5f7fb';
      ctx.font = 'bold 44px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(g.slot), 32, 32);

      const numTex = new THREE.CanvasTexture(numCanvas);
      const numMat = new THREE.MeshBasicMaterial({ map: numTex, transparent: true, opacity: 0.9 });
      const numMesh = new THREE.Mesh(new THREE.PlaneGeometry(1.4, 1.4), numMat);
      numMesh.rotation.x = -Math.PI / 2;
      numMesh.rotation.z = -g.heading;
      numMesh.position.set(g.center[0], 0.032, -g.center[1]);
      this.trackGroup.add(numMesh);
    });
  }

  buildKerbs(kerbs, centerline) {
    if (!kerbs || kerbs.length === 0) return;

    // Red & white alternating kerb segments
    kerbs.forEach((k, idx) => {
      const pt = k.center;
      const isRed = idx % 2 === 0;
      const kerbGeo = new THREE.BoxGeometry(2.2, 0.12, 1.4);
      const kerbMat = new THREE.MeshStandardMaterial({
        color: isRed ? 0xe10600 : 0xf0f2f8,
        roughness: 0.7,
      });
      const kerbMesh = new THREE.Mesh(kerbGeo, kerbMat);
      kerbMesh.position.set(pt[0], 0.05, -pt[1]);
      kerbMesh.receiveShadow = true;
      this.trackGroup.add(kerbMesh);
    });
  }

  buildFloodlights(centerline) {
    if (!centerline || centerline.length === 0) return;

    // Place light towers every ~35 waypoints around track
    const step = Math.max(20, Math.floor(centerline.length / 10));

    for (let i = 0; i < centerline.length; i += step) {
      const pt = centerline[i];
      const nextPt = centerline[(i + 1) % centerline.length];
      const dx = nextPt[0] - pt[0];
      const dz = -nextPt[1] - (-pt[1]);
      const len = Math.hypot(dx, dz) || 1;
      const normX = -dz / len;
      const normZ = dx / len;

      // Place pole 14m outside the road edge
      const poleDist = 18.0;
      const px = pt[0] + normX * poleDist;
      const pz = -pt[1] + normZ * poleDist;

      // Steel Pole
      const poleGeo = new THREE.CylinderGeometry(0.35, 0.45, 24, 8);
      const poleMat = new THREE.MeshStandardMaterial({ color: 0x4a5568, metalness: 0.8, roughness: 0.3 });
      const pole = new THREE.Mesh(poleGeo, poleMat);
      pole.position.set(px, 12, pz);
      pole.castShadow = true;
      this.environmentGroup.add(pole);

      // Light Bar Arm
      const armGeo = new THREE.BoxGeometry(3.5, 0.4, 0.6);
      const arm = new THREE.Mesh(armGeo, poleMat);
      arm.position.set(px, 23.8, pz);
      this.environmentGroup.add(arm);

      // Emissive Floodlight Head
      const lampGeo = new THREE.BoxGeometry(3.2, 0.5, 0.8);
      const lampMat = new THREE.MeshBasicMaterial({ color: 0xffffff });
      const lamp = new THREE.Mesh(lampGeo, lampMat);
      lamp.position.set(px, 23.5, pz);
      this.environmentGroup.add(lamp);

      // Spotlight illuminating track surface
      const spot = new THREE.SpotLight(0xf0f5ff, 1.8, 120, Math.PI / 4, 0.45, 1.2);
      spot.position.set(px, 23, pz);
      spot.target.position.set(pt[0], 0, -pt[1]);
      this.scene.add(spot.target);
      this.environmentGroup.add(spot);
    }
  }

  buildStartGantry(leftEdge, rightEdge, sfData = null) {
    let x1, z1, x2, z2;
    if (leftEdge && leftEdge.length > 0 && rightEdge && rightEdge.length > 0) {
      x1 = leftEdge[0][0];
      z1 = -leftEdge[0][1];
      x2 = rightEdge[0][0];
      z2 = -rightEdge[0][1];
    } else if (sfData && sfData.p1 && sfData.p2) {
      x1 = sfData.p1[0];
      z1 = -sfData.p1[1];
      x2 = sfData.p2[0];
      z2 = -sfData.p2[1];
    } else {
      return;
    }

    const dx = x2 - x1;
    const dz = z2 - z1;
    const roadWidth = Math.hypot(dx, dz);
    const ux = dx / roadWidth;
    const uz = dz / roadWidth;

    const midX = (x1 + x2) / 2;
    const midZ = (z1 + z2) / 2;
    const width = roadWidth + 8.0; // Place pillars 4m outside track edges on grass
    const angle = Math.atan2(dz, dx);

    // Gantry Structure (Truss bridge over track)
    const trussMat = new THREE.MeshStandardMaterial({ color: 0x1f2430, metalness: 0.85, roughness: 0.3 });

    // Left & Right Pillars (safely 4m outside on grass)
    const pillarGeo = new THREE.BoxGeometry(0.8, 11, 0.8);
    const leftPillar = new THREE.Mesh(pillarGeo, trussMat);
    const rightPillar = new THREE.Mesh(pillarGeo, trussMat);

    leftPillar.position.set(x1 - ux * 4.0, 5.5, z1 - uz * 4.0);
    rightPillar.position.set(x2 + ux * 4.0, 5.5, z2 + uz * 4.0);

    this.gantryGroup.add(leftPillar);
    this.gantryGroup.add(rightPillar);

    // Overhead Crossbeam
    const beamGeo = new THREE.BoxGeometry(width, 1.0, 1.2);
    const beam = new THREE.Mesh(beamGeo, trussMat);
    beam.position.set(midX, 10.5, midZ);
    beam.rotation.y = -angle;
    this.gantryGroup.add(beam);

    // 5-Light Start Gantry Box
    const boxGeo = new THREE.BoxGeometry(5.0, 1.4, 0.5);
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x0a0c10, roughness: 0.9 });
    const gantryBox = new THREE.Mesh(boxGeo, boxMat);
    gantryBox.position.set(midX, 9.2, midZ);
    gantryBox.rotation.y = -angle;
    this.gantryGroup.add(gantryBox);

    // 5 Start Light Bulbs
    this.startLights = [];
    const lightOffsets = [-1.6, -0.8, 0.0, 0.8, 1.6];

    lightOffsets.forEach((off, i) => {
      const bulbGeo = new THREE.SphereGeometry(0.3, 16, 16);
      const bulbMat = new THREE.MeshBasicMaterial({ color: 0x220000 }); // Off state (dark red)
      const bulb = new THREE.Mesh(bulbGeo, bulbMat);

      // Offset along the gantry beam
      const bx = midX + ux * off;
      const bz = midZ + uz * off;
      bulb.position.set(bx, 9.2, bz);
      this.gantryGroup.add(bulb);
      this.startLights.push(bulb);
    });
  }

  setStartLights(count, allOut = false) {
    // 0 lights, 1..5 lights on, or all lights out
    this.startLights.forEach((bulb, i) => {
      if (allOut) {
        bulb.material.color.setHex(0x111111); // Off
      } else if (i < count) {
        bulb.material.color.setHex(0xff1744); // Illuminated Red
      } else {
        bulb.material.color.setHex(0x220000); // Unlit Dark
      }
    });
  }

  onResize() {
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
  }

  emitSparks(position, count = 25) {
    if (!position || isNaN(position.x) || isNaN(position.y) || isNaN(position.z)) return;
    // Cosmetic collision sparks
    const geom = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const velocities = [];
    for (let i = 0; i < count; i++) {
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y + 0.3;
      positions[i * 3 + 2] = position.z;
      velocities.push(new THREE.Vector3(
        (Math.random() - 0.5) * 8.0,
        Math.random() * 6.0 + 2.0,
        (Math.random() - 0.5) * 8.0
      ));
    }
    geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0xffaa00,
      size: 0.45,
      transparent: true,
      opacity: 1.0,
      blending: THREE.AdditiveBlending,
    });
    const pSystem = new THREE.Points(geom, mat);
    this.scene.add(pSystem);
    this.sparksList.push({ points: pSystem, velocities: velocities, age: 0, maxAge: 0.45 });
  }

  updateSparks(dt = 0.02) {
    for (let i = this.sparksList.length - 1; i >= 0; i--) {
      const s = this.sparksList[i];
      s.age += dt;
      if (s.age >= s.maxAge) {
        this.scene.remove(s.points);
        s.points.geometry.dispose();
        s.points.material.dispose();
        this.sparksList.splice(i, 1);
        continue;
      }
      const posAttr = s.points.geometry.attributes.position;
      const arr = posAttr.array;
      for (let j = 0; j < s.velocities.length; j++) {
        arr[j * 3] += s.velocities[j].x * dt;
        arr[j * 3 + 1] += s.velocities[j].y * dt;
        arr[j * 3 + 2] += s.velocities[j].z * dt;
        s.velocities[j].y -= 9.8 * dt; // Gravity
      }
      posAttr.needsUpdate = true;
      s.points.material.opacity = 1.0 - (s.age / s.maxAge);
    }
  }
}
