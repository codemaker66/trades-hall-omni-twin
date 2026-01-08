import * as THREE from "three";
import { createRTSCameraControls } from "./RTSCameraControls";

export type PlaceType = "table" | "chair" | "stage";
export type EngineMode = "nav" | "edit";

export type SelectedInfo = {
  id: string;
  type: PlaceType;
  position: { x: number; y: number; z: number };
  rotationDeg: number;
};

export type HistoryState = {
  canUndo: boolean;
  canRedo: boolean;
};

export type EngineOptions = {
  host: HTMLDivElement;
  mode: EngineMode;
  snap: boolean;
  grid: boolean;
  fov: number;
  panSpeed: number;
  onSelect?: (info: SelectedInfo | null) => void;
  onWarning?: (warning: string | null) => void;
  onHistoryChange?: (state: HistoryState) => void;
};

export type EngineApi = {
  setMode: (mode: EngineMode) => void;
  setSnap: (snap: boolean) => void;
  setGrid: (grid: boolean) => void;
  setFov: (fov: number) => void;
  setPanSpeed: (speed: number) => void;
  addObject: (type: PlaceType) => void;
  rotateSelected: (direction: "left" | "right") => void;
  deleteSelected: () => void;
  undo: () => void;
  redo: () => void;
  resize: () => void;
  dispose: () => void;
};

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

type DragState = {
  active: boolean;
  target?: ObjMeta;
  pointerId?: number;
  offset?: THREE.Vector3;
  lastGoodPos?: THREE.Vector3;
  startPos?: THREE.Vector3;
  preview?: THREE.Object3D;
  valid?: boolean;
};

type SnapshotItem = {
  id: string;
  type: PlaceType;
  position: { x: number; y: number; z: number };
  rotationY: number;
};

type SceneSnapshot = {
  items: SnapshotItem[];
  selectedId: string | null;
};

const ROOM_WIDTH_X = 10;
const ROOM_LENGTH_Z = 21;
const ROOM_HEIGHT_Y = 7;
const ROOM_BOUNDS = {
  minX: -ROOM_WIDTH_X / 2,
  maxX: ROOM_WIDTH_X / 2,
  minZ: -ROOM_LENGTH_Z / 2,
  maxZ: ROOM_LENGTH_Z / 2
};

const GRID_SNAP = 0.25;
const ROTATE_SNAP_DEG = 15;
const ROTATE_FREE_DEG = 5;
const STAGE_HEIGHT = 0.4;
const STAGE_WIDTH = 3.2;
const STAGE_DEPTH = 2.2;

const OUTLINE_SELECTED = 0x3aa2ff;
const OUTLINE_HOVER = 0xb0d9ff;

function uid(prefix = "obj") {
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function degreesToRadians(value: number): number {
  return (value * Math.PI) / 180;
}

function getRotationDeg(meta: ObjMeta) {
  return THREE.MathUtils.radToDeg(meta.mesh.rotation.y);
}

function getRotationStep(snapOn: boolean) {
  return snapOn ? ROTATE_SNAP_DEG : ROTATE_FREE_DEG;
}

function snapValue(value: number, step: number) {
  return Math.round(value / step) * step;
}

function applySnapPosition(pos: THREE.Vector3, snapOn: boolean) {
  if (!snapOn) return;
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
  return dx * dx + dz * dz < radius * radius;
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

  return deltaX * deltaX + deltaZ * deltaZ < circle.radius * circle.radius;
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
  return a.min < b.max && b.min < a.max;
}

export function createVenueEngine(options: EngineOptions): EngineApi {
  const { host, onSelect, onWarning, onHistoryChange } = options;

  let mode: EngineMode = options.mode;
  let snapOn = options.snap;
  let currentWarning: string | null = null;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xdfe6ef);

  const size = getHostSize(host);

  const camera = new THREE.PerspectiveCamera(options.fov, size.w / size.h, 0.1, 200);
  camera.position.set(0, 4.5, 9);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(size.w, size.h);
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);

  const rtsControls = createRTSCameraControls({
    camera,
    domElement: renderer.domElement,
    bounds: ROOM_BOUNDS,
    panSpeed: options.panSpeed,
    yaw: THREE.MathUtils.degToRad(35),
    pitch: THREE.MathUtils.degToRad(58),
    distance: 18,
    minDistance: 7,
    maxDistance: 30,
    edgeMarginPx: 26,
    padding: 0.8
  });
  rtsControls.update(0);

  const objectsGroup = new THREE.Group();
  scene.add(objectsGroup);

  const raycaster = new THREE.Raycaster();
  const mouseNdc = new THREE.Vector2();
  const floorPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

  const objs: ObjMeta[] = [];
  let selectedId: string | null = null;
  let hoveredId: string | null = null;
  const dragging: DragState = { active: false };

  const selectionOutline = new THREE.BoxHelper(new THREE.Object3D(), OUTLINE_SELECTED);
  const hoverOutline = new THREE.BoxHelper(new THREE.Object3D(), OUTLINE_HOVER);
  const selectionMat = selectionOutline.material as THREE.LineBasicMaterial;
  selectionMat.transparent = true;
  selectionMat.opacity = 0.9;
  selectionMat.depthTest = false;
  selectionOutline.renderOrder = 2;

  const hoverMat = hoverOutline.material as THREE.LineBasicMaterial;
  hoverMat.transparent = true;
  hoverMat.opacity = 0.6;
  hoverMat.depthTest = false;
  hoverOutline.renderOrder = 1;
  selectionOutline.visible = false;
  hoverOutline.visible = false;
  scene.add(selectionOutline);
  scene.add(hoverOutline);

  const grid = buildGrid(ROOM_WIDTH_X, ROOM_LENGTH_Z, 1);
  grid.visible = options.grid;
  scene.add(grid);

  const roomGroup = buildRoom();
  scene.add(roomGroup);

  let history: SceneSnapshot[] = [];
  let historyIndex = -1;
  let restoring = false;

  function getHostSize(el: HTMLElement) {
    const rect = el.getBoundingClientRect();
    return { w: Math.max(300, rect.width), h: Math.max(300, rect.height) };
  }

  function setWarning(message: string | null) {
    if (currentWarning === message) return;
    currentWarning = message;
    onWarning?.(message);
  }

  function notifyHistory() {
    onHistoryChange?.({
      canUndo: historyIndex > 0,
      canRedo: historyIndex >= 0 && historyIndex < history.length - 1
    });
  }

  function captureSnapshot(): SceneSnapshot {
    const items = objs
      .map((obj) => ({
        id: obj.id,
        type: obj.type,
        position: { x: obj.mesh.position.x, y: obj.mesh.position.y, z: obj.mesh.position.z },
        rotationY: obj.mesh.rotation.y
      }))
      .sort((a, b) => a.id.localeCompare(b.id));

    return { items, selectedId };
  }

  function snapshotsEqual(a: SceneSnapshot, b: SceneSnapshot) {
    if (a.items.length !== b.items.length) return false;
    for (let i = 0; i < a.items.length; i += 1) {
      const ai = a.items[i];
      const bi = b.items[i];
      if (ai.id !== bi.id || ai.type !== bi.type || ai.rotationY !== bi.rotationY) return false;
      if (ai.position.x !== bi.position.x || ai.position.y !== bi.position.y || ai.position.z !== bi.position.z) return false;
    }
    return a.selectedId === b.selectedId;
  }

  function pushHistory() {
    if (restoring) return;
    const snapshot = captureSnapshot();
    const last = history[historyIndex];
    if (last && snapshotsEqual(last, snapshot)) return;

    history = history.slice(0, historyIndex + 1);
    history.push(snapshot);
    historyIndex = history.length - 1;
    notifyHistory();
  }

  function clearObjects() {
    objs.forEach((meta) => disposeObject(meta.mesh));
    objs.length = 0;
  }

  function applySnapshot(snapshot: SceneSnapshot) {
    restoring = true;
    clearSelection();
    clearObjects();

    snapshot.items.forEach((item) => {
      const meta = createObjectMeta(item.type, item.id);
      meta.mesh.position.set(item.position.x, item.position.y, item.position.z);
      meta.mesh.rotation.y = item.rotationY;
      objs.push(meta);
      objectsGroup.add(meta.mesh);
    });

    if (snapshot.selectedId) {
      const found = objs.find((obj) => obj.id === snapshot.selectedId);
      if (found) {
        select(found);
      }
    }

    restoring = false;
    setWarning(null);
  }

  function undo() {
    if (dragging.active) endDrag(false);
    if (historyIndex <= 0) return;
    historyIndex -= 1;
    applySnapshot(history[historyIndex]);
    notifyHistory();
  }

  function redo() {
    if (dragging.active) endDrag(false);
    if (historyIndex >= history.length - 1) return;
    historyIndex += 1;
    applySnapshot(history[historyIndex]);
    notifyHistory();
  }

  function setMouseFromEvent(ev: PointerEvent) {
    const rect = renderer.domElement.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((ev.clientY - rect.top) / rect.height) * 2 - 1);
    mouseNdc.set(x, y);
  }

  function intersectFloor() {
    const hit = new THREE.Vector3();
    raycaster.setFromCamera(mouseNdc, camera);
    raycaster.ray.intersectPlane(floorPlane, hit);
    return hit;
  }

  function intersectObjects() {
    raycaster.setFromCamera(mouseNdc, camera);
    return raycaster.intersectObjects(objectsGroup.children, true);
  }

  function findMeta(object: THREE.Object3D) {
    let cur: THREE.Object3D | null = object;
    while (cur) {
      const meta = objs.find((o) => o.mesh === cur);
      if (meta) return meta;
      cur = cur.parent;
    }
    return undefined;
  }

  function getStageStackY(meta: ObjMeta, pos: THREE.Vector3, rotDeg: number) {
    if (meta.type !== "stage") {
      return meta.mesh.position.y;
    }

    const candidateShape = toShape(meta, pos, rotDeg);
    let highestTop = STAGE_HEIGHT / 2;

    for (const other of objs) {
      if (other.id === meta.id || other.type !== "stage") continue;
      const otherShape = toShape(other, other.mesh.position);
      if (!shapeOverlap(candidateShape, otherShape)) continue;

      const otherTop = other.mesh.position.y + STAGE_HEIGHT / 2;
      highestTop = Math.max(highestTop, otherTop);
    }

    return highestTop + STAGE_HEIGHT / 2;
  }

  function getPlacementStatus(candidate: ObjMeta, pos: THREE.Vector3, rotDeg?: number) {
    const candidateShape = toShape(candidate, pos, rotDeg);

    if (!shapeInsideRoom(candidateShape)) {
      return { ok: false, reason: "Outside room bounds" };
    }

    for (const other of objs) {
      if (other.id === candidate.id) continue;
      if (candidate.type === "stage" && other.type === "stage") continue;

      const otherShape = toShape(other, other.mesh.position);
      if (shapeOverlap(candidateShape, otherShape)) {
        return { ok: false, reason: "Overlaps another object" };
      }
    }

    return { ok: true, reason: null };
  }

  function colorize(meta: ObjMeta, mode: "base" | "bad" | "selected") {
    meta.mesh.traverse((o) => {
      const m = (o as THREE.Mesh).material as THREE.MeshStandardMaterial | undefined;
      if (!m) return;
      if (mode === "base") m.color.copy(meta.baseColor);
      if (mode === "bad") m.color.setRGB(1, 0.28, 0.28);
      if (mode === "selected") m.color.setRGB(0.35, 0.8, 1);
    });
  }

  function clearSelection() {
    selectedId = null;
    objs.forEach((o) => colorize(o, "base"));
    selectionOutline.visible = false;
    onSelect?.(null);
  }

  function select(meta?: ObjMeta) {
    objs.forEach((o) => colorize(o, "base"));
    if (!meta) {
      clearSelection();
      return;
    }

    selectedId = meta.id;
    hoveredId = null;
    updateHoverOutline(undefined);
    colorize(meta, "selected");
    selectionOutline.visible = true;
    selectionOutline.setFromObject(meta.mesh);
    notifySelection(meta);
  }

  function notifySelection(meta?: ObjMeta) {
    if (!meta) {
      onSelect?.(null);
      return;
    }

    onSelect?.({
      id: meta.id,
      type: meta.type,
      position: {
        x: meta.mesh.position.x,
        y: meta.mesh.position.y,
        z: meta.mesh.position.z
      },
      rotationDeg: getRotationDeg(meta)
    });
  }

  function updateSelectionOutline() {
    if (!selectionOutline.visible) return;

    if (dragging.active && dragging.preview) {
      selectionOutline.setFromObject(dragging.preview);
      return;
    }

    const meta = selectedId ? objs.find((obj) => obj.id === selectedId) : undefined;
    if (!meta) {
      selectionOutline.visible = false;
      return;
    }
    selectionOutline.setFromObject(meta.mesh);
  }

  function updateHoverOutline(meta?: ObjMeta) {
    if (!meta || meta.id === selectedId) {
      hoverOutline.visible = false;
      return;
    }
    hoverOutline.visible = true;
    hoverOutline.setFromObject(meta.mesh);
  }

  function createObjectMeta(type: PlaceType, id = uid(type)): ObjMeta {
    let mesh: THREE.Object3D;
    let footprint: Footprint = { kind: "circle", radius: 0.6 };
    let baseColor = new THREE.Color(0.85, 0.85, 0.9);

    if (type === "table") {
      footprint = { kind: "circle", radius: 0.6 };
      baseColor = new THREE.Color(0.86, 0.78, 0.6);
      const geom = new THREE.CylinderGeometry(0.6, 0.6, 0.75, 24);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = 0.375;
      mesh = m;
    } else if (type === "chair") {
      footprint = { kind: "circle", radius: 0.35 };
      baseColor = new THREE.Color(0.58, 0.75, 0.9);
      const geom = new THREE.BoxGeometry(0.45, 0.9, 0.45);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = 0.45;
      mesh = m;
    } else {
      footprint = { kind: "rect", width: STAGE_WIDTH, depth: STAGE_DEPTH };
      baseColor = new THREE.Color(0.64, 0.6, 0.75);
      const geom = new THREE.BoxGeometry(STAGE_WIDTH, STAGE_HEIGHT, STAGE_DEPTH);
      const mat = new THREE.MeshStandardMaterial({ color: baseColor });
      const m = new THREE.Mesh(geom, mat);
      m.position.y = STAGE_HEIGHT / 2;
      mesh = m;
    }

    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.position.set(0, mesh.position.y, 0);

    return { id, type, footprint, mesh, baseColor };
  }

  function createGhostMesh(meta: ObjMeta) {
    const clone = meta.mesh.clone();
    clone.traverse((o) => {
      const mesh = o as THREE.Mesh;
      mesh.castShadow = false;
      mesh.receiveShadow = false;
      if (!mesh.material) return;
      const mat = (mesh.material as THREE.MeshStandardMaterial).clone();
      mat.transparent = true;
      mat.opacity = 0.45;
      mat.depthWrite = false;
      mesh.material = mat;
    });
    return clone;
  }

  function updateGhostColor(ghost: THREE.Object3D, ok: boolean) {
    ghost.traverse((o) => {
      const mesh = o as THREE.Mesh;
      const mat = mesh.material as THREE.MeshStandardMaterial | undefined;
      if (!mat) return;
      if (ok) {
        mat.color.setRGB(0.5, 0.8, 0.95);
      } else {
        mat.color.setRGB(1, 0.35, 0.35);
      }
    });
  }

  function disposeObject(object: THREE.Object3D) {
    object.traverse((o) => {
      const mesh = o as THREE.Mesh;
      mesh.geometry?.dispose?.();
      const material = mesh.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(material)) {
        material.forEach((mat) => mat.dispose?.());
      } else {
        material?.dispose?.();
      }
    });
    if (object.parent) object.parent.remove(object);
  }

  function addObject(type: PlaceType) {
    if (dragging.active) endDrag(false);

    const meta = createObjectMeta(type);
    const start = new THREE.Vector3(ROOM_BOUNDS.minX + 1.6, meta.mesh.position.y, ROOM_BOUNDS.minZ + 1.6);
    if (meta.type === "stage") {
      start.y = getStageStackY(meta, start, 0);
    }

    meta.mesh.position.copy(start);
    objs.push(meta);
    objectsGroup.add(meta.mesh);
    select(meta);
    setWarning(null);
    pushHistory();
  }

  function rotateSelected(direction: "left" | "right") {
    const meta = selectedId ? objs.find((obj) => obj.id === selectedId) : undefined;
    if (!meta) return;

    const step = getRotationStep(snapOn);
    const currentDeg = getRotationDeg(meta);
    const baseDeg = snapOn ? snapValue(currentDeg, ROTATE_SNAP_DEG) : currentDeg;
    const nextDeg = baseDeg + (direction === "left" ? step : -step);
    const finalDeg = snapOn ? snapValue(nextDeg, ROTATE_SNAP_DEG) : nextDeg;

    const prevRotation = meta.mesh.rotation.y;
    const prevPosY = meta.mesh.position.y;
    meta.mesh.rotation.y = THREE.MathUtils.degToRad(finalDeg);

    if (meta.type === "stage") {
      meta.mesh.position.y = getStageStackY(meta, meta.mesh.position, finalDeg);
    }

    const status = getPlacementStatus(meta, meta.mesh.position, finalDeg);
    if (!status.ok) {
      meta.mesh.rotation.y = prevRotation;
      meta.mesh.position.y = prevPosY;
      colorize(meta, "selected");
      setWarning(`Rotation blocked: ${status.reason}`);
      return;
    }

    colorize(meta, "selected");
    setWarning(null);
    notifySelection(meta);
    pushHistory();
  }

  function deleteSelected() {
    const meta = selectedId ? objs.find((obj) => obj.id === selectedId) : undefined;
    if (!meta) return;

    disposeObject(meta.mesh);
    const idx = objs.findIndex((obj) => obj.id === meta.id);
    if (idx >= 0) objs.splice(idx, 1);

    clearSelection();
    setWarning(null);
    pushHistory();
  }

  function startDrag(meta: ObjMeta, pointerId: number, offset: THREE.Vector3) {
    dragging.active = true;
    dragging.target = meta;
    dragging.pointerId = pointerId;
    dragging.offset = offset;
    dragging.lastGoodPos = meta.mesh.position.clone();
    dragging.startPos = meta.mesh.position.clone();
    dragging.valid = true;
    hoveredId = null;
    updateHoverOutline(undefined);

    const ghost = createGhostMesh(meta);
    ghost.position.copy(meta.mesh.position);
    ghost.rotation.copy(meta.mesh.rotation);
    scene.add(ghost);
    dragging.preview = ghost;
    meta.mesh.visible = false;

    rtsControls.setPanInputsEnabled(false);

    renderer.domElement.setPointerCapture?.(pointerId);
  }

  function updateDrag(ev: PointerEvent) {
    if (!dragging.active || !dragging.target || !dragging.preview) return;

    setMouseFromEvent(ev);
    const floorHit = intersectFloor();
    const offset = dragging.offset ?? new THREE.Vector3();
    const nextPos = floorHit.clone().add(offset);

    applySnapPosition(nextPos, snapOn);
    if (dragging.target.type === "stage") {
      nextPos.y = getStageStackY(dragging.target, nextPos, getRotationDeg(dragging.target));
    } else {
      nextPos.y = dragging.target.mesh.position.y;
    }

    const status = getPlacementStatus(dragging.target, nextPos);
    dragging.valid = status.ok;
    dragging.preview.position.copy(nextPos);
    dragging.preview.rotation.copy(dragging.target.mesh.rotation);
    updateGhostColor(dragging.preview, status.ok);

    if (status.ok) {
      dragging.lastGoodPos = nextPos.clone();
      setWarning(null);
    } else {
      setWarning(`Placement blocked: ${status.reason}`);
    }
  }

  function endDrag(apply: boolean) {
    if (!dragging.active || !dragging.target) return;

    const meta = dragging.target;
    const finalPos = dragging.lastGoodPos?.clone() ?? meta.mesh.position.clone();
    const startPos = dragging.startPos?.clone();

    if (dragging.preview) {
      disposeObject(dragging.preview);
      dragging.preview = undefined;
    }

    meta.mesh.visible = true;

    if (apply && dragging.valid) {
      meta.mesh.position.copy(finalPos);
      colorize(meta, "selected");
    } else if (startPos) {
      meta.mesh.position.copy(startPos);
      colorize(meta, "selected");
    }

    if (renderer.domElement && dragging.pointerId !== undefined) {
      renderer.domElement.releasePointerCapture?.(dragging.pointerId);
    }

    dragging.active = false;
    dragging.target = undefined;
    dragging.pointerId = undefined;
    dragging.offset = undefined;
    dragging.lastGoodPos = undefined;
    dragging.startPos = undefined;
    dragging.valid = undefined;

    rtsControls.setPanInputsEnabled(true);
    setWarning(null);

    if (apply && startPos && !startPos.equals(meta.mesh.position)) {
      notifySelection(meta);
      pushHistory();
    }
  }

  function handlePointerDown(ev: PointerEvent) {
    if (mode !== "edit") return;
    if (ev.button !== 0) return;

    setMouseFromEvent(ev);
    const hits = intersectObjects();
    const picked = hits.find((hit) => findMeta(hit.object));

    if (!picked) {
      clearSelection();
      return;
    }

    const meta = findMeta(picked.object);
    if (!meta) return;

    select(meta);

    const floorHit = intersectFloor();
    const offset = meta.mesh.position.clone().sub(floorHit);
    startDrag(meta, ev.pointerId, offset);
  }

  function handlePointerMove(ev: PointerEvent) {
    if (mode !== "edit") return;

    if (dragging.active) {
      renderer.domElement.style.cursor = "grabbing";
      updateDrag(ev);
      return;
    }

    setMouseFromEvent(ev);
    const hits = intersectObjects();
    const picked = hits.find((hit) => findMeta(hit.object));
    const meta = picked ? findMeta(picked.object) : undefined;

    if (meta && meta.id !== hoveredId) {
      hoveredId = meta.id;
      updateHoverOutline(meta);
    } else if (!meta && hoveredId) {
      hoveredId = null;
      updateHoverOutline(undefined);
    }

    renderer.domElement.style.cursor = meta ? "pointer" : "default";
  }

  function handlePointerUp(ev: PointerEvent) {
    if (mode !== "edit") return;
    if (!dragging.active) return;
    endDrag(true);
    renderer.domElement.style.cursor = "default";
  }

  function handlePointerLeave() {
    if (dragging.active) return;
    hoveredId = null;
    updateHoverOutline(undefined);
    renderer.domElement.style.cursor = "default";
  }

  function shouldIgnoreKey(event: KeyboardEvent) {
    const target = event.target as HTMLElement | null;
    if (!target) return false;
    if (target.isContentEditable) return true;
    const tagName = target.tagName;
    return tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT";
  }

  function handleKeyDown(ev: KeyboardEvent) {
    if (shouldIgnoreKey(ev)) return;

    const key = ev.key.toLowerCase();

    if (ev.code === "Space") {
      const meta = selectedId ? objs.find((obj) => obj.id === selectedId) : undefined;
      if (meta) {
        ev.preventDefault();
        rtsControls.focusOn(meta.mesh.position);
      }
      return;
    }

    if (mode !== "edit") return;

    if (ev.key === "Delete" || ev.key === "Backspace") {
      ev.preventDefault();
      deleteSelected();
      return;
    }

    if (key === "q") {
      rotateSelected("left");
      return;
    }

    if (key === "e") {
      rotateSelected("right");
    }
  }

  const pointerDownOptions: AddEventListenerOptions = { capture: true };
  renderer.domElement.addEventListener("pointerdown", handlePointerDown, pointerDownOptions);
  renderer.domElement.addEventListener("pointermove", handlePointerMove);
  renderer.domElement.addEventListener("pointerup", handlePointerUp);
  renderer.domElement.addEventListener("pointerleave", handlePointerLeave);
  window.addEventListener("keydown", handleKeyDown);

  const clock = new THREE.Clock();
  let raf = 0;
  const tick = () => {
    raf = requestAnimationFrame(tick);
    const delta = clock.getDelta();
    rtsControls.update(delta);
    updateSelectionOutline();
    renderer.render(scene, camera);
  };
  tick();

  function resize() {
    const { w, h } = getHostSize(host);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }

  function setMode(nextMode: EngineMode) {
    mode = nextMode;

    if (mode !== "edit" && dragging.active) {
      endDrag(false);
    }

    rtsControls.setPanInputsEnabled(!dragging.active);

    if (mode !== "edit") {
      renderer.domElement.style.cursor = "default";
      hoveredId = null;
      updateHoverOutline(undefined);
    }
  }

  function setSnap(value: boolean) {
    snapOn = value;
  }

  function setGrid(value: boolean) {
    grid.visible = value;
  }

  function setFov(value: number) {
    camera.fov = value;
    camera.updateProjectionMatrix();
  }

  function setPanSpeed(speed: number) {
    rtsControls.setPanSpeed(speed);
  }

  function dispose() {
    cancelAnimationFrame(raf);
    if (dragging.active) {
      endDrag(false);
    }
    rtsControls.dispose();
    renderer.domElement.removeEventListener("pointerdown", handlePointerDown, pointerDownOptions);
    renderer.domElement.removeEventListener("pointermove", handlePointerMove);
    renderer.domElement.removeEventListener("pointerup", handlePointerUp);
    renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
    window.removeEventListener("keydown", handleKeyDown);

    disposeObject(grid);
    disposeObject(roomGroup);
    disposeObject(selectionOutline);
    disposeObject(hoverOutline);
    clearObjects();
    renderer.dispose();

    if (renderer.domElement.parentElement) {
      renderer.domElement.parentElement.removeChild(renderer.domElement);
    }
  }

  // seed history
  pushHistory();
  setMode(mode);
  setWarning(null);
  notifyHistory();

  return {
    setMode,
    setSnap,
    setGrid,
    setFov,
    setPanSpeed,
    addObject,
    rotateSelected,
    deleteSelected,
    undo,
    redo,
    resize,
    dispose
  };
}

function buildRoom() {
  const group = new THREE.Group();
  const ambient = new THREE.AmbientLight(0xffffff, 0.35);
  group.add(ambient);

  const hemi = new THREE.HemisphereLight(0xe7f2ff, 0xaab3bd, 0.75);
  group.add(hemi);

  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(10, 14, 8);
  dir.castShadow = true;
  dir.shadow.mapSize.set(1024, 1024);
  dir.shadow.camera.left = -14;
  dir.shadow.camera.right = 14;
  dir.shadow.camera.top = 14;
  dir.shadow.camera.bottom = -14;
  dir.shadow.camera.near = 1;
  dir.shadow.camera.far = 40;
  group.add(dir);

  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(-8, 6, -6);
  group.add(fill);

  const floorGeom = new THREE.PlaneGeometry(ROOM_WIDTH_X, ROOM_LENGTH_Z);
  const floorMat = new THREE.MeshStandardMaterial({ color: 0xcfd6dd, metalness: 0.0, roughness: 0.9 });
  const floor = new THREE.Mesh(floorGeom, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  floor.receiveShadow = true;
  group.add(floor);

  const wallThickness = 0.12;
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xe7ebef, metalness: 0.0, roughness: 0.92 });

  const wallLeft = new THREE.Mesh(
    new THREE.BoxGeometry(wallThickness, ROOM_HEIGHT_Y, ROOM_LENGTH_Z + wallThickness * 2),
    wallMat
  );
  wallLeft.position.set(ROOM_BOUNDS.minX - wallThickness / 2, ROOM_HEIGHT_Y / 2, 0);
  wallLeft.receiveShadow = true;
  group.add(wallLeft);

  const wallRight = new THREE.Mesh(
    new THREE.BoxGeometry(wallThickness, ROOM_HEIGHT_Y, ROOM_LENGTH_Z + wallThickness * 2),
    wallMat
  );
  wallRight.position.set(ROOM_BOUNDS.maxX + wallThickness / 2, ROOM_HEIGHT_Y / 2, 0);
  wallRight.receiveShadow = true;
  group.add(wallRight);

  const wallFront = new THREE.Mesh(
    new THREE.BoxGeometry(ROOM_WIDTH_X + wallThickness * 2, ROOM_HEIGHT_Y, wallThickness),
    wallMat
  );
  wallFront.position.set(0, ROOM_HEIGHT_Y / 2, ROOM_BOUNDS.minZ - wallThickness / 2);
  wallFront.receiveShadow = true;
  group.add(wallFront);

  const wallBack = new THREE.Mesh(
    new THREE.BoxGeometry(ROOM_WIDTH_X + wallThickness * 2, ROOM_HEIGHT_Y, wallThickness),
    wallMat
  );
  wallBack.position.set(0, ROOM_HEIGHT_Y / 2, ROOM_BOUNDS.maxZ + wallThickness / 2);
  wallBack.receiveShadow = true;
  group.add(wallBack);

  const pillarGeom = new THREE.CylinderGeometry(0.35, 0.35, 6, 18);
  const pillarMat = new THREE.MeshStandardMaterial({ color: 0xb8c0c8, roughness: 0.9 });
  const pillar1 = new THREE.Mesh(pillarGeom, pillarMat);
  pillar1.position.set(-4, 3, 2);
  pillar1.castShadow = true;
  pillar1.receiveShadow = true;
  group.add(pillar1);

  const pillar2 = new THREE.Mesh(pillarGeom, pillarMat);
  pillar2.position.set(4, 3, -1);
  pillar2.castShadow = true;
  pillar2.receiveShadow = true;
  group.add(pillar2);

  return group;
}

function buildGrid(width: number, length: number, step: number) {
  const points: number[] = [];
  const halfW = width / 2;
  const halfL = length / 2;
  const y = 0.01;

  for (let x = -halfW; x <= halfW + 0.001; x += step) {
    points.push(x, y, -halfL, x, y, halfL);
  }

  for (let z = -halfL; z <= halfL + 0.001; z += step) {
    points.push(-halfW, y, z, halfW, y, z);
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x8a96a3, transparent: true, opacity: 0.35 });
  return new THREE.LineSegments(geom, mat);
}
