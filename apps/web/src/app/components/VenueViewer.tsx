"use client";

import React, { useEffect, useRef, useState } from "react";
import {
  createVenueEngine,
  type EngineMode,
  type HistoryState,
  type PlaceType,
  type SelectedInfo
} from "./venueEngine";

const SPEED_PRESETS = [
  { label: "Slow", value: 3.2 },
  { label: "Normal", value: 4.5 },
  { label: "Fast", value: 6.2 }
] as const;

export default function VenueViewer() {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<ReturnType<typeof createVenueEngine> | null>(null);

  const [mode, setMode] = useState<EngineMode>("edit");
  const [snapOn, setSnapOn] = useState(true);
  const [gridOn, setGridOn] = useState(true);
  const [fov, setFov] = useState(90);
  const [speedIndex, setSpeedIndex] = useState(1);
  const [selected, setSelected] = useState<SelectedInfo | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [historyState, setHistoryState] = useState<HistoryState>({ canUndo: false, canRedo: false });

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;

    const engine = createVenueEngine({
      host,
      mode,
      snap: snapOn,
      grid: gridOn,
      fov,
      panSpeed: SPEED_PRESETS[speedIndex].value,
      onSelect: setSelected,
      onWarning: setWarning,
      onHistoryChange: setHistoryState
    });

    engineRef.current = engine;

    const onResize = () => engine.resize();
    window.addEventListener("resize", onResize);

    return () => {
      window.removeEventListener("resize", onResize);
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    engineRef.current?.setMode(mode);
  }, [mode]);

  useEffect(() => {
    engineRef.current?.setSnap(snapOn);
  }, [snapOn]);

  useEffect(() => {
    engineRef.current?.setGrid(gridOn);
  }, [gridOn]);

  useEffect(() => {
    engineRef.current?.setFov(fov);
  }, [fov]);

  useEffect(() => {
    engineRef.current?.setPanSpeed(SPEED_PRESETS[speedIndex].value);
  }, [speedIndex]);

  const addObject = (type: PlaceType) => engineRef.current?.addObject(type);
  const rotateLeft = () => engineRef.current?.rotateSelected("left");
  const rotateRight = () => engineRef.current?.rotateSelected("right");
  const deleteSelected = () => engineRef.current?.deleteSelected();
  const undo = () => engineRef.current?.undo();
  const redo = () => engineRef.current?.redo();

  return (
    <div className="relative h-[calc(100vh-120px)] w-full overflow-hidden rounded-2xl border border-slate-200/20 bg-slate-950/40">
      <div ref={hostRef} className="absolute inset-0" />

      <div className="absolute inset-x-0 top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-slate-900/70 px-4 py-3 backdrop-blur">
        <div className="flex flex-wrap items-center gap-2 text-white">
          <button
            className={`rounded-lg px-3 py-2 text-sm font-semibold ${
              mode === "nav" ? "bg-sky-500/80 text-white" : "bg-white/10 text-white/80 hover:bg-white/15"
            }`}
            onClick={() => setMode((current) => (current === "nav" ? "edit" : "nav"))}
          >
            Mode: {mode === "nav" ? "Nav" : "Edit"}
          </button>
          <button
            className={`rounded-lg px-3 py-2 text-sm ${
              snapOn ? "bg-emerald-500/70 text-white" : "bg-white/10 text-white/80 hover:bg-white/15"
            }`}
            onClick={() => setSnapOn((value) => !value)}
          >
            Snap: {snapOn ? "ON" : "OFF"}
          </button>
          <button
            className={`rounded-lg px-3 py-2 text-sm ${
              gridOn ? "bg-indigo-500/70 text-white" : "bg-white/10 text-white/80 hover:bg-white/15"
            }`}
            onClick={() => setGridOn((value) => !value)}
          >
            Grid: {gridOn ? "ON" : "OFF"}
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            className="rounded-lg bg-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={rotateLeft}
            disabled={!selected}
          >
            Rotate Left
          </button>
          <button
            className="rounded-lg bg-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={rotateRight}
            disabled={!selected}
          >
            Rotate Right
          </button>
          <button
            className="rounded-lg bg-rose-500/30 px-3 py-2 text-sm text-white hover:bg-rose-500/45 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={deleteSelected}
            disabled={!selected}
          >
            Delete
          </button>
          <button
            className="rounded-lg bg-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={undo}
            disabled={!historyState.canUndo}
          >
            Undo
          </button>
          <button
            className="rounded-lg bg-white/10 px-3 py-2 text-sm text-white/80 hover:bg-white/15 disabled:cursor-not-allowed disabled:opacity-40"
            onClick={redo}
            disabled={!historyState.canRedo}
          >
            Redo
          </button>
        </div>
      </div>

      <div className="absolute left-4 top-20 z-10 w-60 rounded-2xl border border-white/10 bg-slate-900/70 p-4 text-white shadow-lg backdrop-blur">
        <div className="text-sm font-semibold uppercase tracking-[0.2em] text-white/60">Inventory</div>
        <div className="mt-4 flex flex-col gap-3">
          <button
            className="group flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-left hover:bg-white/10"
            onClick={() => addObject("stage")}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/10">
              <span className="h-4 w-6 rounded-sm border border-white/60" />
            </span>
            <div>
              <div className="text-sm font-semibold">Stage</div>
              <div className="text-xs text-white/60">3.2m x 2.2m block</div>
            </div>
          </button>
          <button
            className="group flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-left hover:bg-white/10"
            onClick={() => addObject("table")}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/10">
              <span className="h-4 w-4 rounded-full border border-white/60" />
            </span>
            <div>
              <div className="text-sm font-semibold">Round Table</div>
              <div className="text-xs text-white/60">0.6m radius</div>
            </div>
          </button>
          <button
            className="group flex w-full items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3 py-3 text-left hover:bg-white/10"
            onClick={() => addObject("chair")}
          >
            <span className="flex h-10 w-10 items-center justify-center rounded-lg bg-white/10">
              <span className="h-4 w-4 rounded-sm border border-white/60" />
            </span>
            <div>
              <div className="text-sm font-semibold">Chair</div>
              <div className="text-xs text-white/60">0.45m footprint</div>
            </div>
          </button>
        </div>
      </div>

      <div className="absolute right-4 top-20 z-10 w-64 rounded-2xl border border-white/10 bg-slate-900/70 p-4 text-white shadow-lg backdrop-blur">
        <div className="text-sm font-semibold uppercase tracking-[0.2em] text-white/60">Properties</div>
        <div className="mt-4 space-y-3 text-sm">
          <div className="flex items-center justify-between text-white/70">
            <span>Type</span>
            <span className="text-white">{selected ? selected.type : "None"}</span>
          </div>
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-[0.2em] text-white/50">Position</div>
            <div className="grid grid-cols-3 gap-2 text-xs">
              <div className="rounded-lg bg-white/5 px-2 py-2">
                <div className="text-white/50">X</div>
                <div className="text-white">{selected ? selected.position.x.toFixed(2) : "--"}</div>
              </div>
              <div className="rounded-lg bg-white/5 px-2 py-2">
                <div className="text-white/50">Y</div>
                <div className="text-white">{selected ? selected.position.y.toFixed(2) : "--"}</div>
              </div>
              <div className="rounded-lg bg-white/5 px-2 py-2">
                <div className="text-white/50">Z</div>
                <div className="text-white">{selected ? selected.position.z.toFixed(2) : "--"}</div>
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-[0.2em] text-white/50">Rotation</div>
            <div className="rounded-lg bg-white/5 px-3 py-2 text-white">
              {selected ? `${selected.rotationDeg.toFixed(1)} deg` : "--"}
            </div>
          </div>
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-[0.2em] text-white/50">Camera</div>
            <label className="flex items-center justify-between gap-3 text-xs text-white/70">
              <span>FOV</span>
              <input
                type="range"
                min={70}
                max={110}
                step={1}
                value={fov}
                onChange={(event) => setFov(Number(event.target.value))}
                className="w-24 accent-sky-400"
              />
              <span className="w-10 text-right text-white">{Math.round(fov)}</span>
            </label>
            <label className="flex items-center justify-between gap-3 text-xs text-white/70">
              <span>Pan Speed</span>
              <input
                type="range"
                min={0}
                max={SPEED_PRESETS.length - 1}
                step={1}
                value={speedIndex}
                onChange={(event) => setSpeedIndex(Number(event.target.value))}
                className="w-24 accent-sky-400"
              />
              <span className="w-10 text-right text-white">{SPEED_PRESETS[speedIndex].label}</span>
            </label>
          </div>
        </div>
      </div>

      <div className="absolute inset-x-0 bottom-0 z-10 flex items-center justify-between gap-3 border-t border-white/10 bg-slate-900/70 px-4 py-2 text-xs text-white/70 backdrop-blur">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-white">Mode: {mode === "nav" ? "Nav" : "Edit"}</span>
          <span>Snap: {snapOn ? "ON" : "OFF"}</span>
          <span>Grid: {gridOn ? "ON" : "OFF"}</span>
        </div>
        <div className={warning ? "text-rose-200" : "text-white/60"}>
          {warning ?? "All clear"}
        </div>
      </div>
    </div>
  );
}
