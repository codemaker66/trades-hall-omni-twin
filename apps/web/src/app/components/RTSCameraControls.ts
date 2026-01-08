import * as THREE from "three";

type Bounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

type RTSCameraOptions = {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  bounds: Bounds;
  target?: THREE.Vector3;
  yaw?: number;
  pitch?: number;
  distance?: number;
  minDistance?: number;
  maxDistance?: number;
  edgeMarginPx?: number;
  panSpeed?: number;
  zoomSpeed?: number;
  enableRotate?: boolean;
  padding?: number;
};

type RTSCameraHandle = {
  update: (delta: number) => void;
  setPanSpeed: (speed: number) => void;
  setPanInputsEnabled: (enabled: boolean) => void;
  setEnableRotate: (enabled: boolean) => void;
  focusOn: (pos: THREE.Vector3) => void;
  dispose: () => void;
};

type KeyState = {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  rotateLeft: boolean;
  rotateRight: boolean;
};

const PAN_ACCEL = 20;
const PAN_FRICTION = 16;
const ZOOM_FRICTION = 14;
const ROTATE_DAMPING = 12;
const ROTATE_KEY_SPEED = 1.6;
const ROTATE_DRAG_SPEED = 0.005;
const FOCUS_DAMPING = 6;

export function createRTSCameraControls(options: RTSCameraOptions): RTSCameraHandle {
  const { camera, domElement, bounds } = options;
  const target = options.target ? options.target.clone() : new THREE.Vector3(0, 0, 0);

  let yaw = options.yaw ?? THREE.MathUtils.degToRad(40);
  let targetYaw = yaw;
  let pitch = options.pitch ?? THREE.MathUtils.degToRad(55);
  let distance = options.distance ?? 18;
  const minDistance = options.minDistance ?? 6;
  const maxDistance = options.maxDistance ?? 32;
  const edgeMargin = options.edgeMarginPx ?? 28;
  const padding = options.padding ?? 0.6;

  let panSpeed = options.panSpeed ?? 6;
  const zoomSpeed = options.zoomSpeed ?? 10;
  let enableRotate = options.enableRotate ?? false;
  let panInputsEnabled = true;

  const panVelocity = new THREE.Vector3();
  let zoomVelocity = 0;
  let focusTarget: THREE.Vector3 | null = null;

  const keyState: KeyState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    rotateLeft: false,
    rotateRight: false
  };

  let pointerInside = false;
  let pointerPos: { x: number; y: number } | null = null;

  const raycaster = new THREE.Raycaster();
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  let isMiddleDown = false;
  let isRightDown = false;
  let lastPointerX = 0;
  let dragOrigin: THREE.Vector3 | null = null;
  let dragStartTarget: THREE.Vector3 | null = null;

  const clampTarget = () => {
    target.x = THREE.MathUtils.clamp(target.x, bounds.minX + padding, bounds.maxX - padding);
    target.z = THREE.MathUtils.clamp(target.z, bounds.minZ + padding, bounds.maxZ - padding);
    target.y = 0;
  };

  const shouldIgnoreKey = (event: KeyboardEvent) => {
    const targetEl = event.target as HTMLElement | null;
    if (!targetEl) return false;
    if (targetEl.isContentEditable) return true;
    const tagName = targetEl.tagName;
    return tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
  };

  const edgeStrength = (value: number, max: number) => {
    if (value < edgeMargin) {
      const t = (edgeMargin - value) / edgeMargin;
      return -t * t;
    }
    if (value > max - edgeMargin) {
      const t = (value - (max - edgeMargin)) / edgeMargin;
      return t * t;
    }
    return 0;
  };

  const getPlaneIntersection = (event: PointerEvent) => {
    const rect = domElement.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    raycaster.setFromCamera({ x, y }, camera);
    const hit = new THREE.Vector3();
    const ok = raycaster.ray.intersectPlane(floorPlane, hit);
    return ok ? hit : null;
  };

  const onPointerEnter = () => {
    pointerInside = true;
  };

  const onPointerLeave = () => {
    pointerInside = false;
    pointerPos = null;
  };

  const onPointerMove = (event: PointerEvent) => {
    if (pointerInside) {
      const rect = domElement.getBoundingClientRect();
      pointerPos = { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    if (isRightDown && enableRotate && panInputsEnabled) {
      const movementX = event.movementX || event.clientX - lastPointerX;
      lastPointerX = event.clientX;
      targetYaw -= movementX * ROTATE_DRAG_SPEED;
      focusTarget = null;
    }

    if (!isMiddleDown) return;

    const hit = getPlaneIntersection(event);
    if (!hit || !dragOrigin || !dragStartTarget) return;

    const delta = hit.clone().sub(dragOrigin);
    target.copy(dragStartTarget).sub(delta);
    clampTarget();
    focusTarget = null;
    panVelocity.set(0, 0, 0);
  };

  const onPointerDown = (event: PointerEvent) => {
    if (event.button === 1) {
      const hit = getPlaneIntersection(event);
      if (!hit) return;

      isMiddleDown = true;
      dragOrigin = hit;
      dragStartTarget = target.clone();
      domElement.setPointerCapture?.(event.pointerId);
      return;
    }

    if (event.button === 2 && enableRotate && panInputsEnabled) {
      isRightDown = true;
      lastPointerX = event.clientX;
      domElement.setPointerCapture?.(event.pointerId);
      event.preventDefault();
    }
  };

  const onPointerUp = (event: PointerEvent) => {
    if (event.button === 1) {
      isMiddleDown = false;
      dragOrigin = null;
      dragStartTarget = null;
      domElement.releasePointerCapture?.(event.pointerId);
      return;
    }

    if (event.button === 2) {
      isRightDown = false;
      domElement.releasePointerCapture?.(event.pointerId);
    }
  };

  const onContextMenu = (event: MouseEvent) => {
    if (enableRotate) {
      event.preventDefault();
    }
  };

  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    zoomVelocity += event.deltaY * 0.002 * zoomSpeed;
    zoomVelocity = THREE.MathUtils.clamp(zoomVelocity, -20, 20);
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (shouldIgnoreKey(event)) return;

    switch (event.code) {
      case "KeyW":
        keyState.forward = true;
        break;
      case "KeyS":
        keyState.backward = true;
        break;
      case "KeyA":
        keyState.left = true;
        break;
      case "KeyD":
        keyState.right = true;
        break;
      case "KeyQ":
        if (!enableRotate) return;
        keyState.rotateLeft = true;
        break;
      case "KeyE":
        if (!enableRotate) return;
        keyState.rotateRight = true;
        break;
      default:
        return;
    }

    event.preventDefault();
  };

  const onKeyUp = (event: KeyboardEvent) => {
    if (shouldIgnoreKey(event)) return;

    switch (event.code) {
      case "KeyW":
        keyState.forward = false;
        break;
      case "KeyS":
        keyState.backward = false;
        break;
      case "KeyA":
        keyState.left = false;
        break;
      case "KeyD":
        keyState.right = false;
        break;
      case "KeyQ":
        if (!enableRotate) return;
        keyState.rotateLeft = false;
        break;
      case "KeyE":
        if (!enableRotate) return;
        keyState.rotateRight = false;
        break;
      default:
        return;
    }

    event.preventDefault();
  };

  domElement.addEventListener("pointerenter", onPointerEnter);
  domElement.addEventListener("pointerleave", onPointerLeave);
  domElement.addEventListener("pointermove", onPointerMove);
  domElement.addEventListener("pointerdown", onPointerDown);
  domElement.addEventListener("pointerup", onPointerUp);
  domElement.addEventListener("contextmenu", onContextMenu);
  domElement.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);

  const updateCamera = () => {
    const cosPitch = Math.cos(pitch);
    const sinPitch = Math.sin(pitch);

    camera.position.set(
      target.x + Math.sin(yaw) * cosPitch * distance,
      target.y + sinPitch * distance,
      target.z + Math.cos(yaw) * cosPitch * distance
    );
    camera.lookAt(target);
  };

  const update = (delta: number) => {
    const dt = Math.min(Math.max(delta, 0), 0.1);

    if (focusTarget) {
      const t = 1 - Math.exp(-FOCUS_DAMPING * dt);
      target.lerp(focusTarget, t);
      if (target.distanceToSquared(focusTarget) < 0.0004) {
        focusTarget = null;
      }
      clampTarget();
    }

    if (enableRotate && panInputsEnabled) {
      const rotationInput = (keyState.rotateRight ? 1 : 0) - (keyState.rotateLeft ? 1 : 0);
      if (rotationInput !== 0) {
        targetYaw += rotationInput * ROTATE_KEY_SPEED * dt;
      }
    }

    yaw = THREE.MathUtils.damp(yaw, targetYaw, ROTATE_DAMPING, dt);

    const canPan = panInputsEnabled && !isMiddleDown && !isRightDown;
    let inputX = 0;
    let inputZ = 0;

    if (canPan) {
      if (pointerInside && pointerPos) {
        const rect = domElement.getBoundingClientRect();
        inputX += edgeStrength(pointerPos.x, rect.width);
        inputZ += edgeStrength(pointerPos.y, rect.height);
      }

      inputX += (keyState.right ? 1 : 0) - (keyState.left ? 1 : 0);
      inputZ += (keyState.backward ? 1 : 0) - (keyState.forward ? 1 : 0);
    }

    const inputVec = new THREE.Vector3(inputX, 0, inputZ);
    if (inputVec.lengthSq() > 1) {
      inputVec.normalize();
    }

    if (inputVec.lengthSq() > 0) {
      const forward = new THREE.Vector3(-Math.sin(yaw), 0, -Math.cos(yaw));
      const right = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
      const desired = new THREE.Vector3();
      desired.addScaledVector(right, inputVec.x);
      desired.addScaledVector(forward, -inputVec.z);
      desired.multiplyScalar(panSpeed);

      panVelocity.x = THREE.MathUtils.damp(panVelocity.x, desired.x, PAN_ACCEL, dt);
      panVelocity.z = THREE.MathUtils.damp(panVelocity.z, desired.z, PAN_ACCEL, dt);
      focusTarget = null;
    } else {
      panVelocity.x = THREE.MathUtils.damp(panVelocity.x, 0, PAN_FRICTION, dt);
      panVelocity.z = THREE.MathUtils.damp(panVelocity.z, 0, PAN_FRICTION, dt);
    }

    target.addScaledVector(panVelocity, dt);
    clampTarget();

    if (Math.abs(zoomVelocity) > 0.0001) {
      distance += zoomVelocity * dt;
      zoomVelocity = THREE.MathUtils.damp(zoomVelocity, 0, ZOOM_FRICTION, dt);
    }

    distance = THREE.MathUtils.clamp(distance, minDistance, maxDistance);

    updateCamera();
  };

  const setPanSpeed = (speed: number) => {
    panSpeed = speed;
  };

  const setPanInputsEnabled = (enabled: boolean) => {
    panInputsEnabled = enabled;
    if (!enabled) {
      panVelocity.set(0, 0, 0);
      isRightDown = false;
      isMiddleDown = false;
    }
  };

  const setEnableRotate = (enabled: boolean) => {
    enableRotate = enabled;
    if (!enabled) {
      keyState.rotateLeft = false;
      keyState.rotateRight = false;
      targetYaw = yaw;
      isRightDown = false;
    }
  };

  const focusOn = (pos: THREE.Vector3) => {
    focusTarget = new THREE.Vector3(pos.x, 0, pos.z);
    panVelocity.set(0, 0, 0);
  };

  const dispose = () => {
    domElement.removeEventListener("pointerenter", onPointerEnter);
    domElement.removeEventListener("pointerleave", onPointerLeave);
    domElement.removeEventListener("pointermove", onPointerMove);
    domElement.removeEventListener("pointerdown", onPointerDown);
    domElement.removeEventListener("pointerup", onPointerUp);
    domElement.removeEventListener("contextmenu", onContextMenu);
    domElement.removeEventListener("wheel", onWheel);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
  };

  updateCamera();

  return {
    update,
    setPanSpeed,
    setPanInputsEnabled,
    setEnableRotate,
    focusOn,
    dispose
  };
}
