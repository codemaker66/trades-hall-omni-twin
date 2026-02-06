# OmniTwin Codebase Status

> Last updated: 2026-02-06

## Current Stack

| Layer | Technology | Version | Target (Master Command) | Migration Needed? |
|-------|-----------|---------|------------------------|-------------------|
| Monorepo | **Turborepo** | 2.8.3 | Nx | No — Turborepo is simpler and working well |
| Package Manager | **npm** | 11.6.0 | pnpm | No — npm workspaces are stable; migration optional |
| Frontend | **Next.js** (App Router) | 16.1.1 | 15+ | Exceeds target |
| React | **React 19** | 19.2.4 | 19 | Matches |
| 3D Engine | **Three.js + R3F** | 0.164 + 9.5 | r182+ R3F v9 | Three.js could be upgraded; R3F v9 matches |
| Styling | **Tailwind CSS** | 4.1.18 | v4 | Matches |
| Database | **PostgreSQL + PostGIS** | 16-3.4 (Docker) | PostgreSQL + PostGIS | Matches |
| ORM | **Drizzle ORM** | 0.44.0 | Drizzle or Prisma | Matches |
| Backend | **Hono** | 4.7.0 | Next.js API Routes | Keep Hono — separate server is better for scale |
| Auth | **Custom (argon2 + sessions)** | — | NextAuth / Auth.js / Clerk | Custom is fine for now; can add OAuth later |
| Event Sourcing | **Custom** (packages/engine) | — | Emmett | Custom is well-tested and working; no migration |
| ECS | **bitECS** | 0.3.40 | — | Not in master command; bonus capability |
| Cache | **Redis** (ioredis) | 7-alpine | — | Working |
| Event Streaming | **Redpanda** (Kafka-compat) | latest (Docker) | — | Available but unused |
| Language | **TypeScript strict** | 5.5.4 | TypeScript strict | Matches |
| Testing | **Vitest** | 4.0.18 | — | Working (260 tests passing) |
| CI | **GitHub Actions** | — | — | Working (lint, typecheck, test, build) |

### Stack Recommendation

**Keep the current stack.** Turborepo + npm + Hono + custom event sourcing are all working and well-tested. No rip-and-replace needed. Adapt the master command phases to the existing architecture rather than migrating tools.

---

## What Works

### 3D Venue Editor (apps/web)
- Full 3D scene with procedurally-generated Trades Hall (21m x 10m x 6m)
- 4 furniture types: round table, trestle table, chair, platform
- Drag-and-drop with grid snapping, trestle end-to-end snapping, platform bounds
- Multi-select (click, shift-click, area drag)
- Transform gizmo (translate + rotate modes)
- Group/ungroup furniture items
- Undo/redo with history batching (50+ step history)
- RTS-style camera (orbit, zoom, edge-pan, WASD)
- Keyboard shortcuts (Ctrl+Z, G, R, Del, etc.)
- Inventory tracking with capacity warnings
- Scenario save/load/delete/rename with status (draft/review/approved)
- Project export (JSON) and import (replace/merge modes)
- Shareable project links (inline URL encoding + snapshot API)
- Chair prompt modal for bulk placement
- Dark theme UI with Framer Motion animations
- Toast notification system
- Error boundary

### Event Sourcing Engine (packages/engine)
- 19 domain event types, 17 command types
- Pure command validator (returns Result<T, E>)
- Pure state projector (applyEvent, projectState)
- Command handler (validates + maps to events, assigns UUIDs + versions)
- Undo manager with item snapshots and compensating events
- Store adapter bridging engine with Zustand
- Event migration system (schema versioning)

### ECS / Spatial Engine (packages/engine/ecs)
- bitECS world with 8 component types (Position, Rotation, Scale, BoundingBox, FurnitureTag, GroupMember, Selectable, Draggable)
- Spatial hash with configurable cell size
- AABB collision detection (broadphase + narrowphase)
- Rectangle and point selection
- Grid/height snapping, nearest-neighbor, K-nearest search
- EcsBridge: domain items <-> ECS entities (syncAll, addItem, removeItem, moveItem)
- EntityIdMap: bidirectional string <-> number mapping

### Database Layer (packages/db)
- Drizzle ORM with PostgreSQL
- Schema: users, sessions, oauth_accounts, venues, venue_permissions, venue_events, venue_snapshots
- Event store with optimistic concurrency control
- Snapshot support for fast state reconstruction
- 4-tier venue permissions (owner, editor, viewer, commenter)

### Auth (apps/server)
- Hono API server with auth routes
- Password hashing (argon2)
- Session management
- Auth middleware

### Infrastructure
- Docker Compose: web, server, postgres, redis, redpanda
- Dev override with hot-reload volumes
- CI pipeline: lint -> typecheck -> test -> build
- Feature flags system (7 flags, env + localStorage overrides)
- Redis cache abstraction (get, set, del, publish, subscribe)

### Tests (260 total)
- 182 engine tests (command-validator, command-handler, projector, undo-manager, event-migrator, ecs, ecs-systems, ecs-bridge)
- 78 web tests (drag, selection, furniture, history)

---

## What's Partially Implemented

| Feature | State | What's Missing |
|---------|-------|---------------|
| Auth UI | Login/register pages exist | OAuth providers, session refresh, invite system |
| Feature flags | Config + hook + admin panel | Server-side flag evaluation |
| Server API | Auth routes only | No CRUD for venues, events, layouts, bookings |
| Furniture catalog | Inventory sidebar in UI | No persistent catalog DB table, no model upload |
| Design system | Button, Input, Modal, Toast, Tooltip | Missing: Select, Card, Badge, Avatar, DataTable, EmptyState |
| CI pipeline | Web only | Server not tested/built in CI |

---

## What's Placeholder / Minimal

| Item | Status |
|------|--------|
| @omni-twin/shared | Single line: `type SharedHealth = "ok"` |
| apps/server routes | Auth scaffolding only, no business logic |
| Redpanda (Kafka) | Docker service defined, not connected to anything |
| README.md | Minimal, references old Vite setup |

---

## What's Missing Entirely

### From Master Command Phase 1 (Foundation)
- [ ] Database tables: floor_plans, events (bookings), proposals, bookings, furniture_catalog
- [ ] CRUD API routes for all entities
- [ ] Zod validation schemas (shared client/server)
- [ ] Rate limiting on public endpoints
- [ ] API integration tests

### From Master Command Phase 2 (Core UI)
- [ ] Dashboard page with metrics
- [ ] Venue list/detail/edit pages
- [ ] Event & booking management (calendar, Kanban board)
- [ ] Proposal builder
- [ ] Multi-page app shell with sidebar navigation

### From Master Command Phase 3 (2D Floor Plan Editor)
- [ ] 2D canvas editor (Konva.js or SVG)
- [ ] Room boundary drawing
- [ ] Background image upload & scaling
- [ ] Smart guides (Figma-style alignment)
- [ ] Floor plan templates
- [ ] PNG/PDF export with dimensions and legend

### From Master Command Phase 4 (3D Visualization)
- [x] R3F scene setup
- [x] Furniture rendering
- [x] Camera controls
- [ ] WebGPU renderer (using WebGL)
- [ ] InstancedMesh optimization
- [ ] LOD system
- [ ] glTF/GLB asset pipeline
- [ ] Draco/KTX2 compression

### From Master Command Phase 5 (Collaboration)
- [ ] Yjs CRDT integration
- [ ] WebSocket backend (Hocuspocus)
- [ ] Presence & awareness (live cursors)
- [ ] Conflict resolution
- [ ] Offline support (y-indexeddb)

### From Master Command Phase 6 (Polish)
- [ ] WCAG 2.2 AA accessibility audit
- [ ] Screen reader announcements for 3D actions
- [ ] Lighthouse performance > 90
- [ ] Onboarding tour
- [ ] Security hardening (CSP, CORS, CSRF)
- [ ] Structured logging
- [ ] Error monitoring (Sentry)

---

## File Inventory Summary

| Location | Files | Purpose |
|----------|-------|---------|
| apps/web/src/app/components/ | ~25 | 3D scene, furniture, camera, selection, UI overlays |
| apps/web/src/app/components/venue-viewer/ | ~12 | Header, sidebar, toolbar, modals, hooks, utils |
| apps/web/src/store/ | ~8 | Zustand slices (furniture, selection, inventory, scenarios, history) |
| apps/web/src/app/(auth)/ | 2 | Login, register pages |
| apps/web/src/app/api/ | 2 | Share snapshot API |
| apps/server/src/ | 5 | Hono server, auth routes/middleware/sessions/password |
| packages/types/src/ | 5 | Domain types, events, commands, auth, schema versions |
| packages/engine/src/ | 16+8 | Event sourcing + ECS engine with tests |
| packages/db/src/ | 7 | Drizzle schema, event store, client |
| packages/config/src/ | 2 | Feature flags |
| packages/cache/src/ | 3 | Redis cache abstraction |
| packages/shared/src/ | 1 | Placeholder |
