# OmniTwin Progress Tracker

> Living checklist mirroring VENUE_PLATFORM_MASTER_COMMAND.md phases.
> Updated after each session.

---

## Pre-Flight: Codebase Assessment

- [x] Inventory existing codebase
- [x] Document findings in STATUS.md
- [x] Identify current stack vs target stack
- [x] Create PROGRESS.md (this file)

**Stack decision**: Keep Turborepo + npm + Hono + custom event sourcing. No migration to Nx/pnpm/Emmett needed — current stack is working and well-tested.

---

## Phase 1: Foundation & Data Architecture

### 1.1 Project Structure
- [x] Monorepo with workspace packages (Turborepo, not Nx — keeping as-is)
- [x] TypeScript strict mode, ESLint, Prettier
- [ ] `.env.example` with all required env vars documented — exists but incomplete
- [x] `packages/db/` with Drizzle ORM schema

### 1.2 Database Schema
- [x] `users` table (id, email, name, role, passwordHash, avatarUrl, onboarding, preferences, timestamps)
- [x] `sessions` table (id, userId, expiresAt)
- [x] `oauth_accounts` table (provider, providerAccountId, tokens)
- [x] `venues` table (id, name, slug, description, address, lat/lng, capacity, sqft, type, pricing, amenities, images, status)
- [x] `venue_permissions` table (venueId, userId, role)
- [x] `venue_events` table (event sourcing — id, venueId, version, type, payload, userId, timestamp)
- [x] `venue_snapshots` table (venueId, version, state)
- [x] `floor_plans` table (id, venueId, name, version, dimensions, objects JSONB, is_template)
- [x] `occasions` table (bookings — id, venueId, organizerId, name, type, dates, guest_count, status, floor_plan_id)
- [x] `proposals` table (id, occasionId, venueId, status, pricing, message, valid_until)
- [x] `bookings` table (id, occasionId, venueId, proposalId, status, amounts, contract)
- [x] `furniture_catalog` table (id, name, category, model_url, dimensions, capacity, stackable)

### 1.3 Authentication & Authorization
- [x] Auth routes (register, login, logout, me)
- [x] Password hashing (argon2)
- [x] Session management (cookie-based)
- [x] Roles defined in DB (owner, editor, viewer, commenter)
- [x] RBAC middleware: `requireRole()` for global roles, `requireVenueAccess()` for venue permissions
- [ ] Invite system (owners invite managers, planners share view-only links)

### 1.4 API Layer
- [x] Share snapshot API (POST /api/share)
- [x] CRUD: venues (GET list, POST create, GET detail, PATCH update, DELETE archive)
- [x] CRUD: floor_plans (GET list, POST create, GET detail, PATCH update, DELETE)
- [x] CRUD: occasions (GET list, POST create, GET detail, PATCH update, PATCH status, DELETE)
- [ ] CRUD: proposals
- [ ] CRUD: bookings
- [x] CRUD: furniture_catalog (GET list, POST create, GET detail, PATCH update, DELETE)
- [x] Zod validation schemas (shared client/server in @omni-twin/shared)
- [x] Consistent error response format (parseBody + isResponse helper)
- [ ] Rate limiting on public endpoints

### 1.5 Tests for Phase 1
- [x] Engine unit tests (182 tests — command-validator, projector, handler, undo, ECS)
- [x] Zod schema unit tests (29 tests — venues, floor plans, occasions, catalog)
- [ ] API route integration tests (CRUD operations — needs running PostgreSQL)
- [ ] Database migration tests (up/down — needs running PostgreSQL)

---

## Phase 2: Core UI & Venue Management

### 2.1 Layout & Navigation
- [x] App shell with sidebar navigation (collapsible)
- [x] Dashboard page with key metrics (upcoming events, pending inquiries)
- [x] Responsive design (works on desktop)
- [x] Dark mode (Tailwind)

### 2.2 Venue Management Pages
- [x] Venue list (grid/list toggle, search, filter)
- [x] Venue detail/edit form (all fields, pricing, amenities)
- [ ] Venue public profile preview
- [ ] Image upload for venues

### 2.3 Event & Booking Management
- [x] Event list (list view with table)
- [ ] Event detail page
- [x] Booking pipeline (Kanban board)
- [ ] Proposal builder

### 2.4 Furniture Catalog Management
- [x] Inventory sidebar with usage tracking (in VenueViewer)
- [x] Catalog browser (grid view, thumbnails, search, filter)
- [ ] Add/edit items (3D model upload, dimensions, capacity)
- [x] Default furniture catalog (12 common items)

### 2.5 Design System
- [x] Button (primary, ghost, danger)
- [x] IconButton
- [x] Input (text, dark theme)
- [x] Modal
- [x] ConfirmDialog
- [x] Tooltip (with keyboard shortcut display)
- [x] Toast notifications (success, error, info)
- [x] Design tokens (theme.ts — colors, typography, spacing, shadows)
- [x] Select component
- [x] Card component
- [x] Badge component
- [x] Avatar component
- [x] DataTable component
- [x] EmptyState component
- [x] Loading skeletons (Skeleton, SkeletonRow, SkeletonCard)
- [x] UI barrel export (ui/index.ts)

### 2.6 Tests for Phase 2
- [x] Component tests (27 tests — Card, Badge, Avatar, DataTable, EmptyState, Skeleton, Select)
- [ ] E2E tests: create venue, create event, send proposal (Playwright)

---

## Phase 3: 2D Floor Plan Editor

### 3.1 Canvas Setup
- [x] 2D canvas (Konva.js + react-konva)
- [x] Grid with configurable snap (1ft, 6in, 2ft)
- [x] Zoom (scroll wheel toward pointer), pan (pan tool or middle-click)
- [ ] Minimap
- [ ] Ruler/measurement display

### 3.2 Floor Plan Drawing
- [ ] Draw room boundaries (walls)
- [x] Set room dimensions (planWidthFt, planHeightFt in store)
- [ ] Upload background image and scale
- [ ] Multiple rooms/zones

### 3.3 Furniture Placement (2D)
- [x] Click-to-add from catalog sidebar (10 furniture types)
- [x] Snap to grid + smart alignment
- [x] Rotate (45° increments, R key + Shift+R)
- [x] Multi-select (marquee + shift-click), group move
- [ ] Copy/paste, duplicate
- [ ] Right-click context menu
- [x] Undo/redo (50-step history with batch support)
- [x] Delete (Del/Backspace keys)
- [x] Keyboard shortcuts (Ctrl+Z, Ctrl+Shift+Z, Ctrl+A, Escape, R, Del)

### 3.4 Smart Features
- [x] Auto-count (live chair/table/seat count in toolbar)
- [x] Capacity validation warnings (banner at bottom)
- [x] Spacing checker (red lines between items closer than 3ft)
- [ ] Fire code helper (exit path, aisle width)

### 3.5 Floor Plan Templates
- [ ] Save as template
- [x] Pre-built templates: Theater (120), Banquet (80), Classroom (40), Cocktail (60), U-Shape (24), Ceremony (100), Boardroom (16)
- [x] Apply template then customize (loadTemplate action with undo)

### 3.6 Export & Sharing
- [x] PNG export (2x resolution via Konva stage)
- [x] Legend generator (text summary with dimensions, counts, item breakdown)
- [ ] PDF export with title block
- [ ] Shareable view-only link
- [ ] Print-optimized view

### 3.7 Tests for Phase 3
- [x] Store CRUD tests (add, remove, update, rotate, selection)
- [x] Undo/redo + batch integration tests
- [x] Snap-to-grid helper tests
- [x] Template validation tests
- [x] Metrics calculation tests
- [x] Viewport control tests
- [x] Export legend tests
- [x] 39 tests total

---

## Phase 4: 3D Visualization

### 4.1 Three.js / R3F Setup
- [x] Three.js 0.164 + R3F 9.5 + drei 10.7 installed
- [x] WebGL renderer (default)
- [ ] WebGPU renderer with fallback
- [ ] `<PerformanceMonitor>` from drei
- [x] Lazy-loaded 3D view (Three.js only initialized when switching to 3D tab)

### 4.2 Scene Construction
- [x] Hall model (extruded walls, floor, doors, windows, dome) — full 3D editor
- [x] Furniture rendering (round table, trestle table, chair, platform)
- [x] Lighting (ambient + spot + point + contact shadows)
- [x] Floor plane with procedural wood texture
- [x] 3D Preview: converts 2D floor plan items to 3D scene (coordinateBridge.ts)
- [x] Fallback colored boxes for items without 3D models (decor, equipment)
- [ ] Load GLB models from furniture catalog

### 4.3 3D Navigation
- [x] RTS camera controls (orbit, zoom, edge-pan, WASD) — full 3D editor
- [x] MapControls in 3D preview (orbit, pan, zoom)
- [ ] First-person walkthrough mode
- [x] Preset camera angles (top-down, entrance, stage, perspective)
- [x] Smooth animated transitions (easeInOutCubic camera lerp)

### 4.4 Bidirectional Sync
- [x] 2D ↔ 3D toggle view (FloorPlanEditor tabs)
- [x] 2D floor plan items reflected in 3D (coordinate bridge: feet→meters, centered origin)
- [x] Category→FurnitureType mapping (round-table, trestle-table, chair, platform)
- [x] Object selection in 3D (click to select, highlight) — full 3D editor

### 4.5 Performance Optimization
- [x] Shared geometries + materials (module-level, reused)
- [x] React.memo on all furniture components
- [x] Object pooling (threePool.ts)
- [x] InstancedMesh for repeated furniture (chairs: 6 draw calls, round tables: 2, trestle: 3)
- [ ] LOD system (simplified meshes when zoomed out)
- [x] DPR capping ([1, 2])

### 4.6 Asset Pipeline
- [ ] GLB/GLTF upload with validation
- [ ] Thumbnail generation
- [ ] Dimension extraction
- [ ] Draco/KTX2 compression
- [ ] Object storage (S3/R2)

### 4.7 Tests for Phase 4
- [x] Drag & selection tests (78 web tests) — full 3D editor
- [x] 2D-to-3D coordinate conversion tests (26 tests)
- [x] Category mapping, rotation conversion, floor dimensions, camera presets
- [ ] Performance benchmark (200-object scene < 200 draw calls)

---

## Phase 5: Real-Time Collaboration

### 5.1 Yjs Integration
- [x] Install Yjs v13.6+ (yjs, y-websocket, y-indexeddb, y-protocols)
- [x] Y.Doc per floor plan (yjsModel.ts)
- [x] Y.Map per furniture item, Y.Array for item list
- [x] Typed observation events (observeDeep on items array)

### 5.2 WebSocket Backend
- [x] y-websocket client provider (useCollaboration hook)
- [ ] Hocuspocus server (server-side WebSocket — future)
- [ ] Auth hook (validate session, check access)
- [ ] Persistence hook (save Y.Doc to DB, debounced)

### 5.3 Presence & Awareness
- [x] Connected user avatars in toolbar (ConnectionStatusIndicator)
- [x] Remote cursor tracking (awareness protocol — updateCursor/clearCursor)
- [x] Remote selection tracking (updateSelection)
- [x] Deterministic color assignment (12 colorblind-safe colors)
- [ ] Render remote cursors on canvas (colored circles)

### 5.4 Conflict Handling
- [x] CRDT auto-merge (Yjs Y.Map per-property LWW)
- [x] Echo suppression (bidirectional bridge prevents loops)
- [x] Incremental reconciliation (property-level diff in yjsBridge)
- [ ] Visual indicator for remote changes
- [ ] Optimistic drag locking

### 5.5 Offline Support
- [x] y-indexeddb for local persistence
- [x] Queue offline changes, sync on reconnect (Yjs built-in)
- [x] Connection status indicator (connected/connecting/offline with colored dot)

### 5.6 Tests for Phase 5
- [x] Y.Doc data model CRUD tests (16 tests)
- [x] Yjs ↔ Zustand bridge tests (7 tests — push, pull, bidirectional, echo suppression)
- [x] Two-client sync tests (3 tests — sync, concurrent edits, LWW convergence)
- [x] Awareness/presence tests (4 tests — colors, local state, remote filtering)
- [x] 30 tests total
- [ ] Load test (10 concurrent editors)

---

## Phase 6: Polish, Accessibility & Production Readiness

### 6.1 Accessibility
- [x] Keyboard navigation (useEditorKeyboard hook — Tab cycle, arrow nudge, Del, R, Ctrl+Z/A, Escape)
- [x] Tab through furniture items, arrow keys to nudge, Shift for 5x speed
- [ ] Accessible table/list view for 3D objects
- [ ] WCAG 2.2 AA color contrast audit
- [x] aria-live regions (metrics bar, capacity warning role=alert)
- [ ] Numeric input fields as drag alternative (WCAG 2.5.7)
- [x] Screen reader announcements (ScreenReaderProvider + useAnnounce hook)
- [ ] Color-blind safe palette (Okabe-Ito)
- [x] `prefers-reduced-motion` support (CSS global — disables all animations/transitions)
- [ ] High contrast mode
- [x] eslint-plugin-jsx-a11y recommended rules enabled
- [x] Focus-visible ring (indigo-50, 2px offset)
- [x] Skip-to-content link styles

### 6.2 Performance Audit
- [ ] Lighthouse > 90 on all pages
- [ ] LCP < 2.5s, FID < 100ms, CLS < 0.1
- [x] Code-splitting (3D view lazy-loaded via next/dynamic)
- [ ] Image optimization (WebP/AVIF)
- [ ] Database query optimization (indexes, slow queries)
- [ ] API caching

### 6.3 Error Handling & Edge Cases
- [x] Global error boundary
- [x] Toast notifications (save success/failure)
- [x] Auto-save with visual indicator (debounced localStorage, saving/saved/error dot)
- [ ] Session expiry handling
- [ ] Rate limit error handling
- [ ] Large floor plan handling (> 500 objects)

### 6.4 Onboarding
- [ ] Guided tour (5 steps max)
- [ ] Endowed progress
- [ ] Empty states with CTAs
- [ ] Tooltip hints for non-obvious features
- [ ] Sample venue for exploration

### 6.5 Security Audit
- [ ] Input sanitization
- [ ] File upload validation (server-side)
- [x] SQL injection prevention (Drizzle ORM handles parameterization)
- [x] XSS prevention (React auto-escaping, no dangerouslySetInnerHTML)
- [ ] CSRF protection
- [x] Secure headers (CSP, HSTS, X-Frame-Options, X-Content-Type-Options, X-XSS-Protection, Referrer-Policy, Permissions-Policy)
- [ ] Shareable link token security

### 6.6 DevOps & Deployment
- [x] Docker Compose (web, server, postgres, redis, redpanda)
- [x] CI pipeline (lint, typecheck, test, build)
- [ ] CI: add server to pipeline
- [ ] Database migration strategy (up/down, seed data)
- [ ] Staging + production environment config
- [ ] Health check endpoint
- [ ] Structured logging (JSON, request IDs)
- [ ] Error monitoring (Sentry)

### 6.7 Tests for Phase 6
- [x] Keyboard navigation tests (Tab cycle, nudge, Shift speed, locked skip, delete, rotate, select all, escape, undo/redo — 18 tests)
- [x] Auto-save tests (localStorage save/load, no-overwrite, corrupt data, clear — 5 tests)
- [x] Security header validation tests (CSP, HSTS max-age — 2 tests)
- [x] 25 tests total

### 6.8 Documentation
- [ ] README.md (setup, architecture, env vars)
- [ ] API documentation (from Zod/OpenAPI)
- [ ] CONTRIBUTING.md

---

## Transcendent Engineering Additions

### T2: Incremental Computation Framework
- [x] `IncrementalGraph` class with input/derived/observer nodes
- [x] Height-based topological scheduling (dirty nodes processed lowest-first)
- [x] Change propagation with cutoff (if derived produces same value, stop)
- [x] Dynamic dirty propagation (not recursive upfront — enables proper cutoff)
- [x] Batch stabilization (set multiple inputs, stabilize once)
- [x] Observer nodes with side-effect callbacks
- [x] Custom equality functions for structural comparison
- [x] Metrics: totalRecomputations(), resetCounters(), size, stabilizeCount
- [x] 19 tests (unit + 4 property-based)

### T1: Constraint Solver for Automatic Layout Generation
- [x] Types: RoomConfig, FurnitureSpec, Placement, LayoutRequest/Result, Violation
- [x] LayoutGrid: discretized room (6-inch cells), obstacle/exit zone marking, occupy/vacate/snapshot
- [x] Hard constraints: no-overlap, bounds, obstacle collision, exit clearance, aisle width
- [x] Soft objectives: capacity utilization, space coverage, sightline raycasting, symmetry, exit access
- [x] Greedy placement with MRV heuristic (fixed-zone → wall-adjacent → large items → small)
- [x] Simulated annealing optimizer (position perturbation + rotation flip with Metropolis acceptance)
- [x] Deterministic PRNG (Mulberry32) for reproducible results
- [x] Public API: solve(), validate(), score()
- [x] 38 tests (32 unit + 6 property-based: hard constraint satisfaction, bounds, overlap, capacity, scores, determinism)

### T6: Custom Binary Protocol for Real-Time Sync
- [x] Wire protocol types: 7 op types (move, rotate, place, remove, scale, batch_move, batch_rotate)
- [x] HLC (Hybrid Logical Clock): tick/receive/compare, 8-byte packed encoding (48-bit wall + 16-bit counter)
- [x] Binary encoder: WireOp → ArrayBuffer via DataView (little-endian, 25 bytes/move vs ~180 JSON)
- [x] Binary decoder: ArrayBuffer → WireOp with offset cursor for streaming
- [x] Batch framing: multi-op frames with [4B length][2B count][ops...] header
- [x] Delta compression: DeltaCompressor with per-object tracking, int16 fixed-point (1mm precision)
- [x] Deadzone filtering: suppress sub-0.5mm position changes during drag
- [x] Furniture type index mapping (0-6, matches ECS FurnitureTag.type)
- [x] 51 tests (45 unit + 7 property-based + 6 benchmarks)
- [x] Benchmark: ~4.8x size reduction, 24% delta compression savings, 6-byte batch overhead

### T3: Custom Spatial CRDT for Collaborative Editing
- [x] Operation types: add, remove, move (displacement), rotate (delta), scale (delta)
- [x] Vec3 math: add, equality with epsilon, constants (ZERO, ONE)
- [x] State reconstruction: deterministic from op set (base from AddOp + sum of deltas)
- [x] Merge function: set union of op logs (commutative, associative, idempotent by construction)
- [x] SpatialDocument: op log + incremental state cache + per-object op tracking
- [x] Add-wins semantics: concurrent add/remove resolved by HLC (add wins on tie)
- [x] Delta sync protocol: state vector exchange, getMissingOps, fullSync
- [x] Op ID generation: replicaId:counter format with parseOpId
- [x] 33 tests (25 unit + 8 property-based CRDT correctness proofs)

### T4: WebGPU Compute Shaders for Real-Time Spatial Analysis
- [x] Types: AABB2D, Point2D, RoomGeometry, AnalysisItem, CollisionResult, SightlineResult, CrowdFlowResult, GpuCapabilities
- [x] Parallel collision detection: WGSL compute shader (workgroup_size 64, atomicOr bitmask) + CPU fallback (O(N²) AABB pairs)
- [x] Sightline analysis: WGSL with chairSightlines + heatmapSightlines entry points, ray-AABB slab intersection
- [x] CPU sightline fallback: per-chair raycasting + grid heatmap with shadow casting behind obstacles
- [x] Crowd flow evacuation: WGSL agent update with nearest-exit pathfinding, obstacle/agent repulsion
- [x] CPU crowd flow: full agent-based sim with density tracking, bottleneck detection, multi-exit support
- [x] Exported from engine index.ts with aliased names (gpuDetectCollisionsCPU, gpuAnalyzeSightlinesCPU, gpuSimulateCrowdFlowCPU)
- [x] 25 tests (collision 5, sightlines 6, crowd flow 6, WGSL validation 3, property-based 5)

### T5: Property-Based Testing (fast-check)
- [x] Incremental framework: consistency, idempotency, minimality, commutativity (4 properties)
- [x] Projector: immutability, version monotonicity, place-remove roundtrip, move idempotency, group-dissolve roundtrip, projectState equivalence (6 properties)
- [x] SpatialHash: insert-query consistency, negative coordinates, removal completeness, size tracking, cell boundaries (5 properties)
- [x] AABB collision: symmetry, self-overlap, separation detection, containment (4 properties)
- [x] Snapping: grid idempotency, proximity bounds, alignment, height idempotency, K-nearest ordering (5 properties)
- [x] ~2,400 generated test cases total across all properties
- [x] 20 property-based test entries

---

## Session Log

| Date | Phase | Work Done |
|------|-------|-----------|
| 2026-02-06 | Pre-Flight | Codebase inventory, STATUS.md, PROGRESS.md, stack assessment |
| 2026-02-06 | Phase 1 | DB schema (5 new tables + enhanced venues/users), RBAC middleware, CRUD API routes (4 domains), Zod schemas, 29 schema tests |
| 2026-02-06 | Phase 2 | Design system (7 new components), app shell with sidebar, dashboard, venue list/detail, events page with pipeline, catalog page, 27 component tests |
| 2026-02-06 | Phase 3 | 2D floor plan editor (Konva canvas, grid snap, 10 furniture types, multi-select, drag, undo/redo, 7 templates, spacing checker, capacity warnings, PNG export, legend), 39 tests |
| 2026-02-06 | Phase 4 | 2D/3D toggle view, coordinate bridge (feet↔meters), 3D preview with InstancedMesh, camera presets with animated transitions, lazy-loaded Three.js, 26 tests |
| 2026-02-06 | Phase 5 | Yjs collaboration infrastructure (Y.Doc model, bidirectional Yjs↔Zustand bridge with echo suppression, presence/awareness system, y-websocket provider, y-indexeddb offline persistence, ConnectionStatusIndicator UI), 30 tests |
| 2026-02-07 | Phase 6 | Accessibility (prefers-reduced-motion, eslint-plugin-jsx-a11y, keyboard nav hook, ARIA live regions, screen reader announcements), auto-save with indicator, security headers (CSP/HSTS/XSS), Modal useId fix, 25 tests |
| 2026-02-07 | T2+T5 | Incremental computation framework (IncrementalGraph with height-based topological scheduling, cutoff propagation, batch stabilization, observers). Property-based tests (fast-check): 19 incremental tests + 20 property tests across projector, spatial hash, AABB collision, and snapping systems. Total: 475 tests (221 engine + 29 shared + 225 web). |
| 2026-02-07 | T1 | Constraint solver for automatic layout generation: grid discretization, hard constraints (overlap, bounds, obstacle, exit clearance, aisle width), soft objectives (sightlines, symmetry, exit access), greedy MRV placement + simulated annealing optimizer. 38 tests. Total: 513 tests (259 engine + 29 shared + 225 web). |
| 2026-02-07 | T6 | Custom binary wire protocol: HLC clock, binary encoder/decoder (DataView), batch framing, delta compression (int16 fixed-point), deadzone filtering. 51 tests. Total: 564 tests (259 engine + 29 shared + 225 web + 51 wire-protocol). |
| 2026-02-07 | T3 | Spatial intent CRDT: displacement-based merge (both concurrent intents preserved), SpatialDocument with op log + incremental cache, delta sync protocol, add-wins semantics. 33 tests including 8 property-based CRDT proofs (SEC, commutativity, associativity, idempotency, convergence). Total: 597 tests. |
| 2026-02-07 | T4 | WebGPU compute shaders for spatial analysis: WGSL shaders + CPU fallbacks for parallel collision detection, sightline analysis with heatmaps, crowd flow evacuation simulation. 25 tests including property-based. Total: 622 tests (284 engine + 29 shared + 225 web + 51 wire-protocol + 33 spatial-crdt). |
