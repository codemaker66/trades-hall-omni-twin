"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type PlaceType = "table" | "chair" | "stage";

type Footprint =
  | {
      kind: "circle";
      radius: number;
    }
  | {
      kind: "rect";
      width: number;
      depth: number;
    };

type ObjMeta = {
  id: string;
  type: PlaceType;
  footprint: Footprint;
  mesh: THREE.Object3D;
  baseColor: THREE.Color;
};

type CircleShape = {
  kind: "circle";
  x: number;
  z: number;
  radius: number;
};

type RectShape = {
  kind: "rect";
  x: number;
  z: number;
  width: number;
  depth: number;
  rotDeg: number;
};

type Shape2D = CircleShape | RectShape;

type Point2D = {
  x: number;
  z: number;
};

const ROOM_BOUNDS = {
  minX: -10,
  maxX: 10,
  minZ: -6,
  maxZ: 6
};

const GRID_SNAP = 0.25;
const ROTATE_SNAP_DEG = 15;
const ROTATE_FREE_DEG = 5;

function uid(prefix = "obj") {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

export default function VenueViewer() {
  const hostRef = useRef<HTMLDivElement | null>(null);

  const [zonesOn, setZonesOn] = useState(true);
  const [measureOn, setMeasureOn] = useState(false);
  const [measureText, setMeasureText] = useState<string>("(measurement off)");
  const [snapOn, setSnapOn] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedLabel, setSelectedLabel] = useState<string>("(none)");
  const [hint, setHint] = useState<string>(
    "Tip: Add an object, click it, drag it. Q/E rotates, Delete removes. Snap toggles grid."
  );

  // We keep these refs so Three state survives React re-renders.
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const selectedIdRef = useRef<string | null>(null);
  const snapOnRef = useRef(false);

  const raycaster = useMemo(() => new THREE.Raycaster(), []);
  const mouseNdc = useMemo(() => new THREE.Vector2(), []);
  const floorPlane = useMemo(() => new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), []);

  const objsRef = useRef<ObjMeta[]>([]);
  const zonesGroupRef = useRef<THREE.Group | null>(null);

  // dragging state
  const draggingRef = useRef<{
    active: boolean;
    target?: ObjMeta;
    lastGoodPos?: THREE.Vector3;
    dragOffset?: THREE.Vector3;
    pointerId?: number;
  }>({ active: false });

  // measurement state
  const measureRef = useRef<{
    a?: THREE.Vector3;
    b?: THREE.Vector3;
    line?: THREE.Line;
    dotA?: THREE.Mesh;
    dotB?: THREE.Mesh;
  }>({});

  useEffect(() => {
    snapOnRef.current = snapOn;
  }, [snapOn]);

  const rotationStep = snapOn ? ROTATE_SNAP_DEG : ROTATE_FREE_DEG;

  function getHostSize() {
    const el = hostRef.current;
    if (!el) return { w: 800, h: 600 };
    const r = el.getBoundingClientRect();
    return { w: Math.max(300, r.width), h: Math.max(300, r.height) };
  }

  function setMouseFromEvent(ev: PointerEvent, dom: HTMLElement) {
    const rect = dom.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    mouseNdc.set(x, y);
  }

  function intersectFloor(camera: THREE.Camera) {
    const hit = new THREE.Vector3();
    raycaster.setFromCamera(mouseNdc, camera);
    raycaster.ray.intersectPlane(floorPlane, hit);
    return hit;
  }

  function intersectObjects(camera: THREE.Camera, root: THREE.Object3D) {
    raycaster.setFromCamera(mouseNdc, camera);
    const hits = raycaster.intersectObjects(root.children, true);
    return hits;
  }

  function getRotationDeg(meta: ObjMeta) {
    return THREE.MathUtils.radToDeg(meta.mesh.rotation.y);
  }

  function getRotationStep() {
    return snapOnRef.current ? ROTATE_SNAP_DEG : ROTATE_FREE_DEG;
  }

  function snapValue(value: number, step: number) {
    return Math.round(value / step) * step;
  }

  function applySnapPosition(pos: THREE.Vector3) {
    if (!snapOnRef.current) return;
    pos.x = snapValue(pos.x, GRID_SNAP);
    pos.z = snapValue(pos.z, GRID_SNAP);
  }

  function toShape(meta: ObjMeta, pos: THREE.Vector3, rotDeg?: number): Shape2D {
    if (meta.footprint.kind === "circle") {
      return {
        kind: "circle",
        x: pos.x,
        z: pos.z,
        radius: meta.footprint.radius
      };
    }

    return {
      kind: "rect",
      x: pos.x,
      z: pos.z,
      width: meta.footprint.width,
      depth: meta.footprint.depth,
      rotDeg: rotDeg ?? getRotationDeg(meta)
    };
  }

  function isInsideRoom(x: number, z: number) {
    return x >= ROOM_BOUNDS.minX && x <= ROOM_BOUNDS.maxX && z >= ROOM_BOUNDS.minZ && z <= ROOM_BOUNDS.maxZ;
  }

  function shapeInsideRoom(shape: Shape2D) {
    if (shape.kind === "circle") {
      return (
        shape.x - shape.radius >= ROOM_BOUNDS.minX &&
        shape.x + shape.radius <= ROOM_BOUNDS.maxX &&
        shape.z - shape.radius >= ROOM_BOUNDS.minZ &&
        shape.z + shape.radius <= ROOM_BOUNDS.maxZ
      );
    }

    return getRectPoints(shape).every((point) => isInsideRoom(point.x, point.z));
  }

  function shapeOverlap(a: Shape2D, b: Shape2D): boolean {
    if (a.kind === "circle" && b.kind === "circle") {
      return circleCircleOverlap(a, b);
    }

    if (a.kind === "rect" && b.kind === "rect") {
      return rectRectOverlap(a, b);
    }

    const rect = a.kind === "rect" ? a : b;
    const circle = a.kind === "circle" ? a : b;
    return rectCircleOverlap(rect, circle);
  }

  function circleCircleOverlap(a: CircleShape, b: CircleShape): boolean {
    const dx = a.x - b.x;
    const dz = a.z - b.z;
    const radius = a.radius + b.radius;
    return dx * dx + dz * dz <= radius * radius;
  }

  function rectRectOverlap(a: RectShape, b: RectShape): boolean {
    const aPoints = getRectPoints(a);
    const bPoints = getRectPoints(b);
    const axes = [...getRectAxes(a), ...getRectAxes(b)];

    for (const axis of axes) {
      const aProjection = projectPoints(aPoints, axis);
      const bProjection = projectPoints(bPoints, axis);
      if (!rangesOverlap(aProjection, bProjection)) {
        return false;
      }
    }

    return true;
  }

  function rectCircleOverlap(rect: RectShape, circle: CircleShape): boolean {
    const rotation = degreesToRadians(-rect.rotDeg);
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);

    const dx = circle.x - rect.x;
    const dz = circle.z - rect.z;
    const localX = dx * cos - dz * sin;
    const localZ = dx * sin + dz * cos;

    const halfWidth = rect.width / 2;
    const halfDepth = rect.depth / 2;

    const closestX = clamp(localX, -halfWidth, halfWidth);
    const closestZ = clamp(localZ, -halfDepth, halfDepth);

    const deltaX = localX - closestX;
    const deltaZ = localZ - closestZ;

    return deltaX * deltaX + deltaZ * deltaZ <= circle.radius * circle.radius;
  }

  function getRectAxes(rect: RectShape): Point2D[] {
    const rotation = degreesToRadians(rect.rotDeg);
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);

    return [
      { x: cos, z: sin },
      { x: -sin, z: cos }
    ];
  }

  function getRectPoints(rect: RectShape): Point2D[] {
    const halfWidth = rect.width / 2;
    const halfDepth = rect.depth / 2;
    const rotation = degreesToRadians(rect.rotDeg);
    const cos = Math.cos(rotation);
    const sin = Math.sin(rotation);

    const localPoints = [
      { x: -halfWidth, z: -halfDepth },
      { x: halfWidth, z: -halfDepth },
      { x: halfWidth, z: halfDepth },
      { x: -halfWidth, z: halfDepth }
    ];

    return localPoints.map((point) => ({
      x: rect.x + point.x * cos - point.z * sin,
      z: rect.z + point.x * sin + point.z * cos
    }));
  }

  function projectPoints(points: Point2D[], axis: Point2D): { min: number; max: number } {
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    for (const point of points) {
      const projection = point.x * axis.x + point.z * axis.z;
      min = Math.min(min, projection);
      max = Math.max(max, projection);
    }

    return { min, max };
  }

  function rangesOverlap(a: { min: number; max: number }, b: { min: number; max: number }): boolean {
    return a.min <= b.max && b.min <= a.max;
  }

  function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
  }

  function degreesToRadians(value: number): number {
    return (value * Math.PI) / 180;
  }

  function isValidPlacement(candidate: ObjMeta, pos: THREE.Vector3, rotDeg?: number) {
    const candidateShape = toShape(candidate, pos, rotDeg);

    if (!shapeInsideRoom(candidateShape)) {
      return false;
    }

    for (const other of objsRef.current) {
      if (other.id === candidate.id) continue;
      if (candidate.type === "stage" && other.type === "stage") continue;

      const otherShape = toShape(other, other.mesh.position);
      if (shapeOverlap(candidateShape, otherShape)) return false;
    }

    return true;
  }

  function colorize(meta: ObjMeta, mode: "base" | "bad" | "selected") {
    meta.mesh.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
      if (!m) return;
      if (mode === "base") m.color.copy(meta.baseColor);
      if (mode === "bad") m.color.setRGB(1, 0.25, 0.25);
      if (mode === "selected") m.color.setRGB(0.35, 0.8, 1);
    });
  }

  function select(meta?: ObjMeta) {
    // clear previous selection highlight
    for (const o of objsRef.current) colorize(o, "base");
    if (meta) {
      colorize(meta, "selected");
      selectedIdRef.current = meta.id;
      setSelectedId(meta.id);
      setSelectedLabel(`${meta.type} (${meta.id})`);
    } else {
      selectedIdRef.current = null;
      setSelectedId(null);
      setSelectedLabel("(none)");
    }
  }

  function getSelectedMeta() {
    const id = selectedIdRef.current;
    if (!id) return undefined;
    return objsRef.current.find((obj) => obj.id === id);
  }

  function rotateSelected(deltaDeg: number) {
    const meta = getSelectedMeta();
    if (!meta) return;

    const currentDeg = getRotationDeg(meta);
    const baseDeg = snapOnRef.current ? snapValue(currentDeg, ROTATE_SNAP_DEG) : currentDeg;
    const nextDeg = baseDeg + deltaDeg;
    const finalDeg = snapOnRef.current ? snapValue(nextDeg, ROTATE_SNAP_DEG) : nextDeg;

    meta.mesh.rotation.y = THREE.MathUtils.degToRad(finalDeg);
    const ok = isValidPlacement(meta, meta.mesh.position, finalDeg);
    colorize(meta, ok ? "selected" : "bad");
  }

  function deleteSelected() {
    const scene = sceneRef.current;
    const meta = getSelectedMeta();
    if (!scene || !meta) return;

    scene.remove(meta.mesh);
    meta.mesh.traverse((o) => {
      const mesh = o as THREE.Mesh;
      mesh.geometry?.dispose?.();
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) {
        material.forEach((mat) => mat.dispose?.());
      } else {
        material?.dispose?.();
      }
    });

    objsRef.current = objsRef.current.filter((obj) => obj.id !== meta.id);
    if (rendererRef.current && draggingRef.current.pointerId !== undefined) {
      rendererRef.current.domElement.releasePointerCapture(draggingRef.current.pointerId);
    }
    draggingRef.current = { active: false };
    if (controlsRef.current) {
      controlsRef.current.enabled = true;
      controlsRef.current.update();
    }
    select(undefined);
  }

  function addObject(type: PlaceType) {
    const scene = sceneRef.current;
    if (!scene) return;

    const id = uid(type);

    let mesh: THREE.Object3D;
    let footprint: Footprint = { kind: "circle", radius: 0.6 };
    let baseColor = new THREE.Color(0.85, 0.85, 0.9);

    if (type === "table") {
      footprint = { kind: "circle", radius: 0.6 };
      baseColor = new THREE.Color(0.85, 0.78, 0.6);
      const geom = new THREE.CylinderGeometry(0.6, 0.6, 0.75, 24);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = 0.375;
      mesh = m;
    } else if (type === "chair") {
      footprint = { kind: "circle", radius: 0.35 };
      baseColor = new THREE.Color(0.6, 0.75, 0.9);
      const geom = new THREE.BoxGeometry(0.45, 0.9, 0.45);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = 0.45;
      mesh = m;
    } else {
      // stage
      footprint = { kind: "rect", width: 3.2, depth: 2.2 };
      baseColor = new THREE.Color(0.65, 0.6, 0.75);
      const geom = new THREE.BoxGeometry(3.2, 0.4, 2.2);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = 0.2;
      mesh = m;
    }

    mesh.position.set(0, mesh.position.y, 0);

    const meta: ObjMeta = { id, type, footprint, mesh, baseColor };

    // Place at a safe starting spot
    const start = new THREE.Vector3(-6, mesh.position.y, -4);
    if (isValidPlacement(meta, start)) mesh.position.set(start.x, start.y, start.z);

    objsRef.current.push(meta);
    scene.add(mesh);

    select(meta);
    setHint("Drag to move. Q/E rotates selected. Delete removes. Snap toggles grid. Red = invalid placement.");
  }

  function rebuildZones() {
    const scene = sceneRef.current;
    if (!scene) return;

    // Remove old zones
    if (zonesGroupRef.current) {
      scene.remove(zonesGroupRef.current);
      zonesGroupRef.current.traverse((o) => {
        const mesh = o as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose?.();
        const mat = mesh.material as THREE.Material;
        mat?.dispose?.();
      });
    }

    const g = new THREE.Group();

    // Example zones (rectangles) – placeholder for future polygon editor
    // "usable area" + "keep-out" as translucent overlays
    const zoneMatUsable = new THREE.MeshBasicMaterial({ color: 0x33ff99, transparent: true, opacity: 0.12, depthWrite: false });
    const zoneMatKeepout = new THREE.MeshBasicMaterial({ color: 0xff3366, transparent: true, opacity: 0.18, depthWrite: false });

    // Usable: big rectangle
    const usable = new THREE.PlaneGeometry(20, 12);
    const usableMesh = new THREE.Mesh(usable, zoneMatUsable);
    usableMesh.rotation.x = -Math.PI / 2;
    usableMesh.position.y = 0.011;
    g.add(usableMesh);

    // Keep-out: small rectangle near "top"
    const keepout = new THREE.PlaneGeometry(4, 2);
    const keepoutMesh = new THREE.Mesh(keepout, zoneMatKeepout);
    keepoutMesh.rotation.x = -Math.PI / 2;
    keepoutMesh.position.set(5, 0.012, 4);
    g.add(keepoutMesh);

    zonesGroupRef.current = g;
    scene.add(g);
  }

  function clearMeasurement() {
    const scene = sceneRef.current;
    if (!scene) return;

    const m = measureRef.current;
    if (m.line) scene.remove(m.line);
    if (m.dotA) scene.remove(m.dotA);
    if (m.dotB) scene.remove(m.dotB);
    measureRef.current = {};
    setMeasureText("(measurement off)");
  }

  function setMeasurementPoint(pt: THREE.Vector3) {
    const scene = sceneRef.current;
    if (!scene) return;

    const m = measureRef.current;

    const dotGeom = new THREE.SphereGeometry(0.08, 16, 16);
    const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

    if (!m.a) {
      m.a = pt.clone();
      m.dotA = new THREE.Mesh(dotGeom, dotMat);
      m.dotA.position.copy(m.a);
      scene.add(m.dotA);
      setMeasureText("Point A set. Click Point B.");
      return;
    }

    // set B
    m.b = pt.clone();
    m.dotB = new THREE.Mesh(dotGeom, dotMat);
    m.dotB.position.copy(m.b);
    scene.add(m.dotB);

    // line
    const lineGeom = new THREE.BufferGeometry().setFromPoints([m.a, m.b]);
    const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff });
    m.line = new THREE.Line(lineGeom, lineMat);
    scene.add(m.line);

    const dist = m.a.distanceTo(m.b);
    setMeasureText(`Distance: ${dist.toFixed(2)}m (click to start a new measure)`);

    // next click restarts measurement
    m.a = undefined;
    m.b = undefined;
  }

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    // --- init three ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0c0f14);

    const { w, h } = getHostSize();

    const camera = new THREE.PerspectiveCamera(55, w / h, 0.1, 200);
    camera.position.set(12, 10, 12);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0, 0);
    controls.update();

    // Lights
    const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 1.0);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(10, 15, 8);
    scene.add(dir);

    // Floor
    const floorGeom = new THREE.PlaneGeometry(20, 12);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x11161e, metalness: 0.0, roughness: 0.95 });
    const floor = new THREE.Mesh(floorGeom, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = 0;
    scene.add(floor);

    // Grid
    const grid = new THREE.GridHelper(20, 20, 0x2b3a4a, 0x1c2531);
    grid.position.y = 0.002;
    scene.add(grid);

    // simple pillars (visual-only placeholders)
    const pillarGeom = new THREE.CylinderGeometry(0.35, 0.35, 6, 18);
    const pillarMat = new THREE.MeshStandardMaterial({ color: 0x2a313c, roughness: 0.9 });
    const pillar1 = new THREE.Mesh(pillarGeom, pillarMat);
    pillar1.position.set(-4, 3, 2);
    scene.add(pillar1);

    const pillar2 = new THREE.Mesh(pillarGeom, pillarMat);
    pillar2.position.set(4, 3, -1);
    scene.add(pillar2);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;

    rebuildZones();

    // --- events ---
    const onResize = () => {
      const { w: nw, h: nh } = getHostSize();
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };

    const onPointerDown = (ev: PointerEvent) => {
      if (!sceneRef.current || !cameraRef.current || !rendererRef.current) return;

      setMouseFromEvent(ev, renderer.domElement);

      // measurement click
      if (measureOn) {
        const hit = intersectFloor(cameraRef.current);
        hit.y = 0.02;
        // if previous measurement exists, clear first
        const m = measureRef.current;
        if (m.line || m.dotA || m.dotB) clearMeasurement();
        setMeasurementPoint(hit);
        return;
      }

      // selection / drag
      const hits = intersectObjects(cameraRef.current, sceneRef.current);
      const picked = hits.find((h) => {
        // find top-level object meta by walking up parents
        let cur: THREE.Object3D | null = h.object;
        while (cur) {
          const meta = objsRef.current.find((o) => o.mesh === cur);
          if (meta) return true;
          cur = cur.parent;
        }
        return false;
      });

      if (!picked) {
        select(undefined);
        draggingRef.current = { active: false };
        return;
      }

      // resolve meta
      let cur: THREE.Object3D | null = picked.object;
      let meta: ObjMeta | undefined;
      while (cur) {
        meta = objsRef.current.find((o) => o.mesh === cur);
        if (meta) break;
        cur = cur.parent;
      }
      if (!meta) return;

      select(meta);

      draggingRef.current.active = true;
      draggingRef.current.target = meta;
      draggingRef.current.lastGoodPos = meta.mesh.position.clone();
      draggingRef.current.pointerId = ev.pointerId;

      const floorHit = intersectFloor(cameraRef.current);
      const offset = meta.mesh.position.clone().sub(floorHit);
      draggingRef.current.dragOffset = offset;

      if (controlsRef.current) {
        controlsRef.current.enabled = false;
      }
      renderer.domElement.setPointerCapture?.(ev.pointerId);
    };

    const onPointerMove = (ev: PointerEvent) => {
      if (!draggingRef.current.active) return;
      const meta = draggingRef.current.target;
      if (!meta) return;
      if (!cameraRef.current || !rendererRef.current) return;

      setMouseFromEvent(ev, renderer.domElement);

      const floorHit = intersectFloor(cameraRef.current);
      const offset = draggingRef.current.dragOffset ?? new THREE.Vector3();
      const nextPos = floorHit.clone().add(offset);

      // keep Y (height) constant
      nextPos.y = meta.mesh.position.y;
      applySnapPosition(nextPos);

      const ok = isValidPlacement(meta, nextPos);
      if (ok) {
        meta.mesh.position.copy(nextPos);
        draggingRef.current.lastGoodPos = nextPos.clone();
        colorize(meta, "selected");
      } else {
        // show invalid in red but still move visually
        meta.mesh.position.copy(nextPos);
        colorize(meta, "bad");
      }
    };

    const onPointerUp = () => {
      if (!draggingRef.current.active) return;
      const meta = draggingRef.current.target;
      if (!meta) return;

      // if current spot invalid, snap back to last good
      const pos = meta.mesh.position.clone();
      const ok = isValidPlacement(meta, pos, getRotationDeg(meta));
      if (!ok && draggingRef.current.lastGoodPos) {
        meta.mesh.position.copy(draggingRef.current.lastGoodPos);
        colorize(meta, "selected");
      }

      if (rendererRef.current && draggingRef.current.pointerId !== undefined) {
        rendererRef.current.domElement.releasePointerCapture(draggingRef.current.pointerId);
      }
      if (controlsRef.current) {
        controlsRef.current.enabled = true;
      }
      draggingRef.current = { active: false };
    };

    const onKeyDown = (ev: KeyboardEvent) => {
      if (ev.key === "Delete" || ev.key === "Backspace") {
        ev.preventDefault();
        deleteSelected();
        return;
      }

      const meta = getSelectedMeta();
      if (!meta) return;

      const step = getRotationStep();
      const key = ev.key.toLowerCase();
      if (key === "q") rotateSelected(step);
      if (key === "e") rotateSelected(-step);
    };

    window.addEventListener("resize", onResize);
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    window.addEventListener("keydown", onKeyDown);

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("keydown", onKeyDown);

      controls.dispose();
      renderer.dispose();

      if (renderer.domElement.parentElement) renderer.domElement.parentElement.removeChild(renderer.domElement);

      sceneRef.current = null;
      cameraRef.current = null;
      rendererRef.current = null;
      controlsRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measureOn]);

  // keep zones toggle in sync
  useEffect(() => {
    const g = zonesGroupRef.current;
    if (g) g.visible = zonesOn;
  }, [zonesOn]);

  return (
    <div className="w-full h-[calc(100vh-120px)] relative rounded-xl overflow-hidden border border-white/10">
      {/* 3D host */}
      <div ref={hostRef} className="absolute inset-0" />

      {/* UI overlay */}
      <div className="absolute top-3 left-3 flex flex-col gap-2 bg-black/40 border border-white/10 rounded-xl p-3 backdrop-blur">
        <div className="text-white font-semibold">Omni-Twin — Viewer + Editor Demo</div>

        <div className="flex flex-wrap gap-2">
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() => addObject("table")}
          >
            + Table
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() => addObject("chair")}
          >
            + Chair
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() => addObject("stage")}
          >
            + Stage
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() => setZonesOn((v) => !v)}
          >
            Zones: {zonesOn ? "ON" : "OFF"}
          </button>

          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() => {
              // turning on measure cancels drag selection visuals
              setMeasureOn((v) => !v);
              clearMeasurement();
              setHint("Measurement: click floor once (A), then again (B).");
            }}
          >
            Measure: {measureOn ? "ON" : "OFF"}
          </button>

          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm"
            onClick={() =>
              setSnapOn((v) => {
                const next = !v;
                snapOnRef.current = next;
                return next;
              })
            }
          >
            Snap: {snapOn ? "ON" : "OFF"}
          </button>
        </div>

        <div className="flex flex-wrap gap-2">
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={() => rotateSelected(rotationStep)}
            disabled={!selectedId}
          >
            Rotate ⟲
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-white/10 hover:bg-white/15 text-white text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={() => rotateSelected(-rotationStep)}
            disabled={!selectedId}
          >
            Rotate ⟳
          </button>
          <button
            className="px-3 py-2 rounded-lg bg-red-500/30 hover:bg-red-500/40 text-white text-sm disabled:opacity-40 disabled:cursor-not-allowed"
            onClick={deleteSelected}
            disabled={!selectedId}
          >
            Delete
          </button>
        </div>

        <div className="text-xs text-white/80">Selected: {selectedLabel}</div>
        <div className="text-xs text-white/80">Measure: {measureText}</div>
        <div className="text-xs text-white/70 max-w-[340px]">{hint}</div>
      </div>
    </div>
  );
}
