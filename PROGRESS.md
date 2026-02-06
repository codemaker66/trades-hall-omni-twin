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

### 4.2 Scene Construction
- [x] Hall model (extruded walls, floor, doors, windows, dome)
- [x] Furniture rendering (round table, trestle table, chair, platform)
- [x] Lighting (ambient + spot + point + contact shadows)
- [x] Floor plane with procedural wood texture
- [ ] Load GLB models from furniture catalog
- [ ] Fallback colored boxes with labels for missing models

### 4.3 3D Navigation
- [x] RTS camera controls (orbit, zoom, edge-pan, WASD)
- [ ] First-person walkthrough mode
- [ ] Preset camera angles (top-down, entrance, stage)
- [ ] Smooth animated transitions

### 4.4 Bidirectional Sync
- [ ] 2D changes reflected in 3D (needs 2D editor first)
- [x] Object selection in 3D (click to select, highlight)

### 4.5 Performance Optimization
- [x] Shared geometries + materials (module-level, reused)
- [x] React.memo on all furniture components
- [x] Object pooling (threePool.ts)
- [ ] InstancedMesh for repeated furniture
- [ ] LOD system (simplified meshes when zoomed out)
- [ ] DPR capping (1.0 desktop, 1.5 mobile)

### 4.6 Asset Pipeline
- [ ] GLB/GLTF upload with validation
- [ ] Thumbnail generation
- [ ] Dimension extraction
- [ ] Draco/KTX2 compression
- [ ] Object storage (S3/R2)

### 4.7 Tests for Phase 4
- [x] Drag & selection tests (78 web tests)
- [ ] 2D-to-3D coordinate conversion tests
- [ ] Performance benchmark (200-object scene < 200 draw calls)

---

## Phase 5: Real-Time Collaboration

### 5.1 Yjs Integration
- [ ] Install Yjs v13.6+
- [ ] Y.Doc per floor plan
- [ ] Y.Map per furniture item, Y.Array for item list
- [ ] Typed observation events

### 5.2 WebSocket Backend
- [ ] Hocuspocus server
- [ ] Auth hook (validate session, check access)
- [ ] Persistence hook (save Y.Doc to DB, debounced)
- [ ] Load hook (restore from DB)

### 5.3 Presence & Awareness
- [ ] Connected user avatars in toolbar
- [ ] Remote cursors on canvas (colored by user)
- [ ] Remote selection highlights
- [ ] Awareness throttled to 10-15Hz

### 5.4 Conflict Handling
- [ ] CRDT auto-merge
- [ ] Visual indicator for remote changes
- [ ] Optimistic drag locking

### 5.5 Offline Support
- [ ] y-indexeddb for local persistence
- [ ] Queue offline changes, sync on reconnect
- [ ] Connection status indicator

### 5.6 Tests for Phase 5
- [ ] Two-client sync integration test
- [ ] Offline/reconnect test
- [ ] Load test (10 concurrent editors)

---

## Phase 6: Polish, Accessibility & Production Readiness

### 6.1 Accessibility
- [ ] All interactive elements keyboard-accessible
- [ ] Tab through furniture, arrow keys to nudge
- [ ] Accessible table/list view for 3D objects
- [ ] WCAG 2.2 AA color contrast
- [ ] aria-label, aria-live regions
- [ ] Numeric input fields as drag alternative (WCAG 2.5.7)
- [ ] Screen reader announcements
- [ ] Color-blind safe palette (Okabe-Ito)
- [ ] `prefers-reduced-motion` support
- [ ] High contrast mode

### 6.2 Performance Audit
- [ ] Lighthouse > 90 on all pages
- [ ] LCP < 2.5s, FID < 100ms, CLS < 0.1
- [ ] Bundle analysis and code-splitting
- [ ] Image optimization (WebP/AVIF)
- [ ] Database query optimization (indexes, slow queries)
- [ ] API caching

### 6.3 Error Handling & Edge Cases
- [x] Global error boundary
- [x] Toast notifications (save success/failure)
- [ ] Auto-save with visual indicator
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
- [ ] SQL injection prevention (Drizzle handles)
- [ ] XSS prevention
- [ ] CSRF protection
- [ ] Secure headers (CSP, HSTS)
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

### 6.7 Documentation
- [ ] README.md (setup, architecture, env vars)
- [ ] API documentation (from Zod/OpenAPI)
- [ ] CONTRIBUTING.md

---

## Session Log

| Date | Phase | Work Done |
|------|-------|-----------|
| 2026-02-06 | Pre-Flight | Codebase inventory, STATUS.md, PROGRESS.md, stack assessment |
| 2026-02-06 | Phase 1 | DB schema (5 new tables + enhanced venues/users), RBAC middleware, CRUD API routes (4 domains), Zod schemas, 29 schema tests |
| 2026-02-06 | Phase 2 | Design system (7 new components), app shell with sidebar, dashboard, venue list/detail, events page with pipeline, catalog page, 27 component tests |
| 2026-02-06 | Phase 3 | 2D floor plan editor (Konva canvas, grid snap, 10 furniture types, multi-select, drag, undo/redo, 7 templates, spacing checker, capacity warnings, PNG export, legend), 39 tests |
