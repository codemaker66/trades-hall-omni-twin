"use client";

import * as THREE from "three";
import { PointerLockControls } from "three/examples/jsm/controls/PointerLockControls.js";

type Bounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

type FpsControllerOptions = {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  scene: THREE.Scene;
  bounds: Bounds;
  eyeHeight: number;
  sensitivity?: number;
  invertY?: boolean;
  moveSpeed?: number;
  sprintMultiplier?: number;
};

type KeyState = {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  sprint: boolean;
};

type FpsControllerHandle = {
  update: (delta: number) => void;
  setEnabled: (enabled: boolean) => void;
  setSensitivity: (value: number) => void;
  setInvertY: (value: boolean) => void;
  setMoveSpeed: (value: number) => void;
  dispose: () => void;
};

const UP = new THREE.Vector3(0, 1, 0);
const TEMP_FORWARD = new THREE.Vector3();
const TEMP_RIGHT = new THREE.Vector3();
const WISH_DIR = new THREE.Vector3();
const DESIRED_VEL = new THREE.Vector3();

const MAX_PITCH = Math.PI / 2 - 0.05;
const ROTATION_DAMPING = 12;
const BASE_SPEED = 4.5;
const SPRINT_MULTIPLIER = 1.6;
const ACCELERATION = 18;
const FRICTION = 14;
const BOUND_MARGIN = 0.2;

export function createFpsController(options: FpsControllerOptions): FpsControllerHandle {
  const { camera, domElement, scene, bounds, eyeHeight } = options;

  let sensitivity = options.sensitivity ?? 1;
  let invertY = options.invertY ?? false;
  let moveSpeed = options.moveSpeed ?? BASE_SPEED;
  let sprintMultiplier = options.sprintMultiplier ?? SPRINT_MULTIPLIER;
  let enabled = false;

  const controls = new PointerLockControls(camera, domElement) as PointerLockControls & {
    connect: () => void;
    disconnect: () => void;
    _onMouseMove: (event: MouseEvent) => void;
  };

  controls.disconnect();

  const yawObject = new THREE.Object3D();
  const velocity = new THREE.Vector3();
  const keyState: KeyState = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    sprint: false
  };

  let yaw = 0;
  let pitch = 0;
  let targetYaw = 0;
  let targetPitch = 0;

  const minX = bounds.minX + BOUND_MARGIN;
  const maxX = bounds.maxX - BOUND_MARGIN;
  const minZ = bounds.minZ + BOUND_MARGIN;
  const maxZ = bounds.maxZ - BOUND_MARGIN;

  const setFromCamera = () => {
    const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
    yaw = euler.y;
    pitch = THREE.MathUtils.clamp(euler.x, -MAX_PITCH, MAX_PITCH);
    targetYaw = yaw;
    targetPitch = pitch;
  };

  const clearInput = () => {
    keyState.forward = false;
    keyState.backward = false;
    keyState.left = false;
    keyState.right = false;
    keyState.sprint = false;
  };

  const onMouseMove = (event: MouseEvent) => {
    if (!controls.isLocked || !enabled) return;

    const movementX = event.movementX || (event as any).mozMovementX || (event as any).webkitMovementX || 0;
    const movementY = event.movementY || (event as any).mozMovementY || (event as any).webkitMovementY || 0;

    const scale = 0.002 * sensitivity;
    const invert = invertY ? -1 : 1;

    targetYaw -= movementX * scale;
    targetPitch -= movementY * scale * invert;
    targetPitch = THREE.MathUtils.clamp(targetPitch, -MAX_PITCH, MAX_PITCH);
  };

  controls._onMouseMove = onMouseMove;
  controls.connect();

  const onCanvasClick = () => {
    if (!enabled) return;
    controls.lock();
  };

  const shouldIgnoreKey = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    if (!target) return false;
    if (target.isContentEditable) return true;
    const tagName = target.tagName;
    return tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
  };

  const onKeyDown = (event: KeyboardEvent) => {
    if (!enabled) return;
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
      case "ShiftLeft":
      case "ShiftRight":
        keyState.sprint = true;
        break;
      default:
        return;
    }

    event.preventDefault();
  };

  const onKeyUp = (event: KeyboardEvent) => {
    if (!enabled) return;
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
      case "ShiftLeft":
      case "ShiftRight":
        keyState.sprint = false;
        break;
      default:
        return;
    }

    event.preventDefault();
  };

  const onUnlock = () => {
    clearInput();
    velocity.set(0, 0, 0);
  };

  const attachCamera = () => {
    setFromCamera();

    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    camera.getWorldPosition(worldPos);
    camera.getWorldQuaternion(worldQuat);

    yawObject.position.copy(worldPos);
    yawObject.position.y = eyeHeight;
    yawObject.rotation.y = yaw;

    yawObject.add(camera);
    camera.position.set(0, 0, 0);
    camera.rotation.set(pitch, 0, 0);

    scene.add(yawObject);
  };

  const detachCamera = () => {
    const worldPos = new THREE.Vector3();
    const worldQuat = new THREE.Quaternion();
    camera.getWorldPosition(worldPos);
    camera.getWorldQuaternion(worldQuat);

    yawObject.remove(camera);
    scene.add(camera);
    camera.position.copy(worldPos);
    camera.quaternion.copy(worldQuat);
    scene.remove(yawObject);
  };

  const setEnabled = (value: boolean) => {
    if (enabled === value) return;
    enabled = value;

    if (enabled) {
      attachCamera();
    } else {
      controls.unlock();
      clearInput();
      velocity.set(0, 0, 0);
      detachCamera();
    }
  };

  const update = (delta: number) => {
    if (!enabled) return;

    const dt = Math.min(Math.max(delta, 0), 0.1);

    yaw = THREE.MathUtils.damp(yaw, targetYaw, ROTATION_DAMPING, dt);
    pitch = THREE.MathUtils.damp(pitch, targetPitch, ROTATION_DAMPING, dt);

    yawObject.rotation.y = yaw;
    camera.rotation.x = pitch;
    camera.rotation.y = 0;
    camera.rotation.z = 0;

    const inputForward = (keyState.forward ? 1 : 0) - (keyState.backward ? 1 : 0);
    const inputRight = (keyState.right ? 1 : 0) - (keyState.left ? 1 : 0);
    const canMove = controls.isLocked;
    const speed = moveSpeed * (keyState.sprint ? sprintMultiplier : 1);

    if (canMove && (inputForward !== 0 || inputRight !== 0)) {
      TEMP_FORWARD.set(0, 0, -1).applyAxisAngle(UP, yaw);
      TEMP_RIGHT.set(1, 0, 0).applyAxisAngle(UP, yaw);
      WISH_DIR.copy(TEMP_FORWARD).multiplyScalar(inputForward).addScaledVector(TEMP_RIGHT, inputRight);

      if (WISH_DIR.lengthSq() > 0) {
        WISH_DIR.normalize();
      }

      DESIRED_VEL.copy(WISH_DIR).multiplyScalar(speed);
      velocity.x = THREE.MathUtils.damp(velocity.x, DESIRED_VEL.x, ACCELERATION, dt);
      velocity.z = THREE.MathUtils.damp(velocity.z, DESIRED_VEL.z, ACCELERATION, dt);
    } else {
      velocity.x = THREE.MathUtils.damp(velocity.x, 0, FRICTION, dt);
      velocity.z = THREE.MathUtils.damp(velocity.z, 0, FRICTION, dt);
    }

    velocity.y = 0;
    yawObject.position.addScaledVector(velocity, dt);
    yawObject.position.x = THREE.MathUtils.clamp(yawObject.position.x, minX, maxX);
    yawObject.position.z = THREE.MathUtils.clamp(yawObject.position.z, minZ, maxZ);
    yawObject.position.y = eyeHeight;
  };

  domElement.addEventListener("click", onCanvasClick);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  controls.addEventListener("unlock", onUnlock);

  const setSensitivity = (value: number) => {
    sensitivity = value;
  };

  const setInvertY = (value: boolean) => {
    invertY = value;
  };

  const setMoveSpeed = (value: number) => {
    moveSpeed = value;
  };

  const dispose = () => {
    setEnabled(false);
    controls.removeEventListener("unlock", onUnlock);
    domElement.removeEventListener("click", onCanvasClick);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    controls.disconnect();
    scene.remove(yawObject);
  };

  return {
    update,
    setEnabled,
    setSensitivity,
    setInvertY,
    setMoveSpeed,
    dispose
  };
}
