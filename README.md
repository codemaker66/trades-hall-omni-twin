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
# Optional: "file" (default), "memory", or "redis"
SHARE_SNAPSHOT_STORE=file

# Optional: custom snapshot file location when SHARE_SNAPSHOT_STORE=file
SHARE_SNAPSHOT_FILE_PATH=./.data/share-snapshots.json

# Required when SHARE_SNAPSHOT_STORE=redis
# Falls back to REDIS_URL if omitted.
SHARE_SNAPSHOT_REDIS_URL=redis://localhost:6379

# Optional Redis key prefix when SHARE_SNAPSHOT_STORE=redis
SHARE_SNAPSHOT_REDIS_PREFIX=omnitwin:share:
```

When using `SHARE_SNAPSHOT_STORE=redis`, install the Redis client in the web workspace:

```bash
npm install -w apps/web redis
```
