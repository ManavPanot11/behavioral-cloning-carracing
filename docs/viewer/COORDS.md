# Coordinate System & Transformation Specification: Sim to 3D

## 1. Physical Simulation Space (Box2D / Gymnasium CarRacing-v3)

- **Dimensionality**: 2D Cartesian plane ($X, Y$), flat ground. No vertical elevation.
- **Units**: Meters (world units). CarRacing uses `SCALE = 6.0` pixels/meter in rendering.
- **Axes**:
  - $+X$: World Right / East
  - $+Y$: World Up / North
- **Car Orientation**:
  - In `gymnasium.envs.box2d.car_dynamics.Car`, the car's forward direction is the **local $+Y$ body axis** (NOT $+X$).
  - When the Box2D body angle `hull.angle == 0`, the car faces directly along $+Y$ (North).
  - World heading angle $\psi_{\text{sim}} = \text{hull.angle} + \frac{\pi}{2}$.
  - Counter-clockwise rotation is positive (steering $> 0$ turns left, counter-clockwise).

---

## 2. Three.js 3D Visualization Space

- **Dimensionality**: 3D Right-Handed Euclidean space ($X, Y, Z$).
- **Ground Plane**: Horizontal plane defined by $Y_{\text{3D}} = 0$. (Altitude is strictly 0; sim physics is purely 2D).
- **Axes**:
  - $+X_{\text{3D}}$: Screen Right (East)
  - $+Y_{\text{3D}}$: Vertical Elevation (Up into the sky)
  - $+Z_{\text{3D}}$: South / Towards Viewer (so $-Z_{\text{3D}}$ is North / Screen Up when viewed top-down)

---

## 3. Position Mapping

To preserve orientation, geometry, and handedness without mirroring:

$$\begin{aligned}
X_{\text{3D}} &= x_{\text{sim}} \\
Y_{\text{3D}} &= 0 \\
Z_{\text{3D}} &= -y_{\text{sim}}
\end{aligned}$$

### Proof of Non-Mirroring:
1. **Right remains Right**: $+X_{\text{sim}} \implies +X_{\text{3D}}$.
2. **Forward remains Forward**: In sim, moving forward (toward $+Y_{\text{sim}}$) translates to moving toward $-Z_{\text{3D}}$. In a top-down orthographic camera (where camera $+Y$ is up into the sky and camera up vector is $(0, 0, -1)$), $-Z_{\text{3D}}$ projects to the top of the screen, exactly matching $+Y_{\text{sim}}$ in Gym's rendering.
3. **Left Turn Test**:
   - A car facing $+Y_{\text{sim}}$ turns Left $\implies$ velocity vector acquires a negative $X_{\text{sim}}$ component ($v_x < 0$).
   - In 3D, $X_{\text{3D}} = x_{\text{sim}} < 0$, which is to the left of $-Z_{\text{3D}}$.
   - Therefore, a left turn in simulation visibly turns left in the 3D viewer. No mirroring occurs.

---

## 4. Heading & Yaw Rotation

In Three.js, with the car 3D mesh modeled facing along $-Z_{\text{3D}}$ (default front direction):

$$\text{rotation.y} = \text{hull.angle}$$

- When `hull.angle == 0`: car faces along $-Z_{\text{3D}}$ (North).
- When `hull.angle > 0` (turning Left in Box2D): `rotation.y` rotates the front of the car from $-Z$ towards $-X$ (Left).
- Steering angle $\delta$: The front left and right wheels rotate about their local vertical axes by $\delta$ radians (clamped to $[-0.4, +0.4]$ rad per Box2D `lowerAngle`/`upperAngle`).
