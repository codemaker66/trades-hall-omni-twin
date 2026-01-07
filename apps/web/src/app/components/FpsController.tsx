"use client";

import { useEffect, useMemo, useRef, type MutableRefObject } from "react";
import * as THREE from "three";
import { PointerLockControls } from "@react-three/drei";
import type { PointerLockControls as PointerLockControlsImpl } from "three-stdlib";

type Bounds = {
  minX: number;
  maxX: number;
  minZ: number;
  maxZ: number;
};

type FpsControllerProps = {
  enabled: boolean;
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  scene: THREE.Scene;
  bounds: Bounds;
  eyeHeight: number;
  sensitivity: number;
  invertY: boolean;
  updateRef: MutableRefObject<(delta: number) => void>;
};

const LOOK_DAMPING = 14;
const MOVE_ACCEL = 16;
const MOVE_FRICTION = 14;
const BASE_SPEED = 2.6;
const SPRINT_MULT = 1.7;
const WALK_MARGIN = 0.35;
const MOUSE_SCALE = 0.002;
const PITCH_LIMIT = Math.PI / 2 - 0.02;

export default function FpsController({
  enabled,
  camera,
  domElement,
  scene,
  bounds,
  eyeHeight,
  sensitivity,
  invertY,
  updateRef
}: FpsControllerProps) {
  const controlsRef = useRef<PointerLockControlsImpl | null>(null);
  const yawObjectRef = useMemo(() => new THREE.Object3D(), []);
  const velocityRef = useRef(new THREE.Vector3());
  const movementRef = useRef({ forward: false, back: false, left: false, right: false, sprint: false });
  const enabledRef = useRef(enabled);
  const sensitivityRef = useRef(sensitivity);
  const invertYRef = useRef(invertY);
  const lockedRef = useRef(false);

  const yawRef = useRef(0);
  const pitchRef = useRef(0);
  const targetYawRef = useRef(0);
  const targetPitchRef = useRef(0);

  const tempForward = useMemo(() => new THREE.Vector3(), []);
  const tempRight = useMemo(() => new THREE.Vector3(), []);
  const tempWish = useMemo(() => new THREE.Vector3(), []);
  const tempDesired = useMemo(() => new THREE.Vector3(), []);
  const tempPos = useMemo(() => new THREE.Vector3(), []);
  const tempQuat = useMemo(() => new THREE.Quaternion(), []);
  const tempEuler = useMemo(() => new THREE.Euler(0, 0, 0, "YXZ"), []);

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  useEffect(() => {
    sensitivityRef.current = sensitivity;
  }, [sensitivity]);

  useEffect(() => {
    invertYRef.current = invertY;
  }, [invertY]);

  useEffect(() => {
    if (!domElement) return;
    const handleClick = () => {
      if (!enabledRef.current) return;
      controlsRef.current?.lock();
    };
    domElement.addEventListener("click", handleClick);
    return () => {
      domElement.removeEventListener("click", handleClick);
    };
  }, [domElement]);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    controls.onMouseMove = (event: MouseEvent) => {
      if (!enabledRef.current || !controls.isLocked) return;

      const invert = invertYRef.current ? -1 : 1;
      const lookScale = MOUSE_SCALE * sensitivityRef.current;

      targetYawRef.current -= event.movementX * lookScale;
      targetPitchRef.current -= event.movementY * lookScale * invert;
      targetPitchRef.current = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, targetPitchRef.current));
    };
  }, [enabled]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!enabledRef.current) return;
      const key = event.key.toLowerCase();
      if (key === "w") movementRef.current.forward = true;
      if (key === "s") movementRef.current.back = true;
      if (key === "a") movementRef.current.left = true;
      if (key === "d") movementRef.current.right = true;
      if (key === "shift") movementRef.current.sprint = true;
      if (key === "w" || key === "a" || key === "s" || key === "d" || key === "shift") {
        event.preventDefault();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if (key === "w") movementRef.current.forward = false;
      if (key === "s") movementRef.current.back = false;
      if (key === "a") movementRef.current.left = false;
      if (key === "d") movementRef.current.right = false;
      if (key === "shift") movementRef.current.sprint = false;
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    const onLock = () => {
      lockedRef.current = true;
    };
    const onUnlock = () => {
      lockedRef.current = false;
      velocityRef.current.set(0, 0, 0);
      movementRef.current = { forward: false, back: false, left: false, right: false, sprint: false };
    };

    updateRef.current = (delta: number) => {
      if (!enabledRef.current || !lockedRef.current) return;

      const smoothing = 1 - Math.exp(-LOOK_DAMPING * delta);
      yawRef.current += (targetYawRef.current - yawRef.current) * smoothing;
      pitchRef.current += (targetPitchRef.current - pitchRef.current) * smoothing;

      yawObjectRef.rotation.y = yawRef.current;
      camera.rotation.x = pitchRef.current;
      camera.rotation.y = 0;
      camera.rotation.z = 0;

      const forwardInput = (movementRef.current.forward ? 1 : 0) + (movementRef.current.back ? -1 : 0);
      const rightInput = (movementRef.current.right ? 1 : 0) + (movementRef.current.left ? -1 : 0);

      tempWish.set(0, 0, 0);
      if (forwardInput !== 0 || rightInput !== 0) {
        tempForward.set(0, 0, -1).applyQuaternion(yawObjectRef.quaternion);
        tempForward.y = 0;
        tempForward.normalize();

        tempRight.crossVectors(tempForward, camera.up).normalize();

        tempWish.addScaledVector(tempForward, forwardInput);
        tempWish.addScaledVector(tempRight, rightInput);
        if (tempWish.lengthSq() > 0) tempWish.normalize();
      }

      const velocity = velocityRef.current;
      const maxSpeed = BASE_SPEED * (movementRef.current.sprint ? SPRINT_MULT : 1);

      if (tempWish.lengthSq() > 0) {
        tempDesired.copy(tempWish).multiplyScalar(maxSpeed);
        const diff = tempDesired.sub(velocity);
        const accelStep = MOVE_ACCEL * delta;
        if (diff.length() > accelStep) diff.setLength(accelStep);
        velocity.add(diff);
      } else {
        const speed = velocity.length();
        if (speed > 0) {
          const drop = MOVE_FRICTION * delta;
          const newSpeed = Math.max(0, speed - drop);
          velocity.multiplyScalar(newSpeed / speed);
        }
      }

      if (velocity.length() > maxSpeed) {
        velocity.setLength(maxSpeed);
      }

      yawObjectRef.position.addScaledVector(velocity, delta);
      yawObjectRef.position.x = clamp(yawObjectRef.position.x, bounds.minX + WALK_MARGIN, bounds.maxX - WALK_MARGIN);
      yawObjectRef.position.z = clamp(yawObjectRef.position.z, bounds.minZ + WALK_MARGIN, bounds.maxZ - WALK_MARGIN);
      yawObjectRef.position.y = 0;

      camera.position.set(0, eyeHeight, 0);
      camera.updateMatrixWorld();
    };

    const controls = controlsRef.current;
    controls?.addEventListener("lock", onLock);
    controls?.addEventListener("unlock", onUnlock);

    return () => {
      updateRef.current = () => undefined;
      controls?.removeEventListener("lock", onLock);
      controls?.removeEventListener("unlock", onUnlock);
    };
  }, [bounds.maxX, bounds.maxZ, bounds.minX, bounds.minZ, camera, eyeHeight, updateRef, yawObjectRef]);

  useEffect(() => {
    if (!enabled) {
      if (controlsRef.current?.isLocked) {
        controlsRef.current.unlock();
      }
      lockedRef.current = false;
      velocityRef.current.set(0, 0, 0);
      movementRef.current = { forward: false, back: false, left: false, right: false, sprint: false };

      if (camera.parent === yawObjectRef) {
        camera.getWorldPosition(tempPos);
        camera.getWorldQuaternion(tempQuat);
        yawObjectRef.remove(camera);
        camera.position.copy(tempPos);
        camera.quaternion.copy(tempQuat);
      }

      if (scene.children.includes(yawObjectRef)) {
        scene.remove(yawObjectRef);
      }
      return () => undefined;
    }

    if (!scene.children.includes(yawObjectRef)) {
      scene.add(yawObjectRef);
    }

    camera.rotation.order = "YXZ";
    camera.getWorldPosition(tempPos);
    camera.getWorldQuaternion(tempQuat);
    tempEuler.setFromQuaternion(tempQuat, "YXZ");
    yawRef.current = tempEuler.y;
    pitchRef.current = tempEuler.x;
    targetYawRef.current = yawRef.current;
    targetPitchRef.current = pitchRef.current;

    yawObjectRef.position.set(tempPos.x, 0, tempPos.z);
    yawObjectRef.position.x = clamp(yawObjectRef.position.x, bounds.minX + WALK_MARGIN, bounds.maxX - WALK_MARGIN);
    yawObjectRef.position.z = clamp(yawObjectRef.position.z, bounds.minZ + WALK_MARGIN, bounds.maxZ - WALK_MARGIN);
    yawObjectRef.rotation.set(0, yawRef.current, 0);
    camera.rotation.set(pitchRef.current, 0, 0);
    camera.position.set(0, eyeHeight, 0);

    if (camera.parent !== yawObjectRef) {
      yawObjectRef.add(camera);
    }
    return () => {
      if (camera.parent === yawObjectRef) {
        camera.getWorldPosition(tempPos);
        camera.getWorldQuaternion(tempQuat);
        yawObjectRef.remove(camera);
        camera.position.copy(tempPos);
        camera.quaternion.copy(tempQuat);
      }
      if (scene.children.includes(yawObjectRef)) {
        scene.remove(yawObjectRef);
      }
    };
  }, [camera, eyeHeight, enabled, scene, tempEuler, tempPos, tempQuat, yawObjectRef]);

  return (
    <PointerLockControls
      ref={controlsRef}
      enabled={enabled}
      domElement={domElement}
      selector=".__fps_lock_unused__"
    />
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}
