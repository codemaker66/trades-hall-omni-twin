# trades-hall-omni-twin

## Venue Viewer Demo

Local Vite + React + Three.js viewer demo with a placeholder venue, measurement tool, and zone overlays.

## How to run locally

```bash
npm install
npm run dev
```

Then open the URL shown in the terminal (typically http://localhost:5173).

## Web App Share Snapshot Storage

The Next.js app (`apps/web`) now supports short share links via `GET/POST /api/share`.

- Default mode: `file` (persists snapshots to `.data/share-snapshots.json` under repo root)
- Optional mode: `memory` (in-process only)

Environment variables:

```bash
# Optional: "file" (default) or "memory"
SHARE_SNAPSHOT_STORE=file

# Optional: custom snapshot file location when SHARE_SNAPSHOT_STORE=file
SHARE_SNAPSHOT_FILE_PATH=./.data/share-snapshots.json
```
