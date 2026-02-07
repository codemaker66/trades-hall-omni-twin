# Claude Code Master Command: Venue Planning Platform

> **Purpose**: Feed this document to Claude Code alongside your existing codebase to systematically build it into a production-grade venue planning platform. This is designed to be executed as a phased series of Claude Code sessions — NOT all at once.

> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

>
> **How to use**: Start each Claude Code session by saying: "Read VENUE_PLATFORM_MASTER_COMMAND.md, assess the current state of the codebase, and execute the next incomplete phase." Claude Code should check off completed tasks and pick up where it left off.

---

## Pre-Flight: Codebase Assessment (Run This First)

Before any implementation, Claude Code must:

1. **Inventory the existing codebase** — list every file, its purpose, what framework/libraries are in use, and what functionality exists vs. is missing.
2. **Document findings** in `STATUS.md` at the project root — what works, what's broken, what's placeholder, what's missing entirely.
3. **Identify the current stack** — confirm or propose migration to the target stack below.
4. **Create `PROGRESS.md`** — a living checklist mirroring the phases below, updated after each session.

### Target Stack

| Layer | Technology | Notes |
|-------|-----------|-------|
| Monorepo | Nx | Polyglot orchestration |
| Frontend | Next.js 15+ (App Router) | React 19, Server Components |
| 3D Engine | Three.js r182+ via React Three Fiber v9 | WebGPU with WebGL fallback |
| Collaboration | Yjs v13.6+ CRDTs | Hocuspocus WebSocket backend |
| Database | PostgreSQL + PostGIS | Event-sourced via Emmett |
| Auth | NextAuth.js / Auth.js | Or Clerk — whichever is already in use |
| Styling | Tailwind CSS v4 | |
| Package Manager | pnpm | |
| Language | TypeScript (strict) throughout | |

If the existing codebase uses a different stack, propose a migration path or adapt these instructions to what's already there — don't rip and replace unless the user confirms.

---

## Phase 1: Foundation & Data Architecture

**Goal**: Solid data models, auth, API routes, and database schema. No UI polish yet.

### 1.1 Project Structure
- Set up Nx workspace (if not already) with `apps/web/` (Next.js) and `packages/` for shared code
- Configure TypeScript strict mode, ESLint, Prettier
- Set up `.env.example` with all required env vars documented
- Create `packages/db/` with Drizzle ORM or Prisma schema (whichever the codebase already uses, or Drizzle if greenfield)

### 1.2 Database Schema
Design and implement these core tables:

```
venues
  - id, name, slug, description, address, lat/lng (PostGIS POINT)
  - capacity, square_footage, venue_type (enum)
  - owner_id (FK users), pricing_model, base_price
  - amenities (JSONB), images (JSONB array of URLs)
  - status (draft/published/archived), created_at, updated_at

floor_plans
  - id, venue_id (FK), name, version
  - dimensions (width_ft, height_ft)
  - background_image_url (optional uploaded blueprint)
  - objects (JSONB — array of placed furniture/elements with x,y,z,rotation,scale)
  - is_template (boolean), created_by, created_at, updated_at

events
  - id, venue_id (FK), organizer_id (FK users)
  - name, type (enum: wedding/corporate/social/etc)
  - date_start, date_end, setup_time, teardown_time
  - guest_count, status (inquiry/confirmed/completed/cancelled)
  - floor_plan_id (FK — which layout is assigned)
  - budget, notes, created_at, updated_at

users
  - id, email, name, role (enum: owner/manager/planner/viewer)
  - organization_id (FK, nullable), avatar_url
  - onboarding_completed (boolean), preferences (JSONB)

proposals
  - id, event_id (FK), venue_id (FK)
  - status (draft/sent/viewed/accepted/declined)
  - pricing_breakdown (JSONB), custom_message
  - valid_until, sent_at, viewed_at

bookings
  - id, event_id (FK), venue_id (FK), proposal_id (FK)
  - status (pending/confirmed/deposit_paid/completed/cancelled)
  - total_amount, deposit_amount, payment_status
  - contract_url, signed_at

furniture_catalog
  - id, name, category (enum: table/chair/stage/decor/equipment)
  - model_url (GLB/GLTF path), thumbnail_url
  - dimensions (width, depth, height in feet)
  - capacity (for tables/seating), stackable (boolean)
  - is_custom (boolean), created_by
```

### 1.3 Authentication & Authorization
- Implement auth with role-based access control (RBAC)
- Roles: `owner` (full venue control), `manager` (venue operations), `planner` (event creation, floor plan editing), `viewer` (read-only links for clients)
- Middleware protecting API routes and pages by role
- Invite system: owners can invite managers, planners can share view-only links

### 1.4 API Layer
- RESTful API routes under `app/api/` using Next.js route handlers
- CRUD for all entities above
- Input validation with Zod schemas (shared between client and server)
- Proper error handling with consistent error response format
- Rate limiting on public endpoints

### 1.5 Tests for Phase 1
- Unit tests for all Zod schemas
- Integration tests for API routes (CRUD operations)
- Database migration tests (up and down)

**Phase 1 exit criteria**: Can create users, venues, floor plans, events, proposals, and bookings via API. Auth works. Tests pass.

---

## Phase 2: Core UI & Venue Management

**Goal**: Functional, good-looking UI for managing venues and events. No 3D yet.

### 2.1 Layout & Navigation
- App shell with sidebar navigation (collapsible on mobile)
- Dashboard page showing key metrics (upcoming events, pending inquiries, revenue)
- Responsive design — works on tablet and desktop minimum
- Dark mode support via Tailwind

### 2.2 Venue Management Pages
- **Venue list** — grid/list toggle, search, filter by type/status
- **Venue detail/edit** — form for all venue fields, image upload (to S3/Cloudflare R2 or local), amenity checkboxes
- **Venue public profile preview** — what clients see

### 2.3 Event & Booking Management
- **Event list** — filterable calendar view AND list view
- **Event detail** — full event info, linked floor plan, guest count, status workflow
- **Booking pipeline** — Kanban-style board (inquiry → proposal → confirmed → completed)
- **Proposal builder** — form to create pricing proposals, preview as PDF-like view, send via email (or generate shareable link)

### 2.4 Furniture Catalog Management
- **Catalog browser** — grid view with thumbnails, search, filter by category
- **Add/edit items** — upload 3D model (GLB), set dimensions, capacity, name
- Pre-seed with a default furniture catalog (10-15 common items: round tables, rectangular tables, chairs, podium, stage, dance floor, bar, DJ booth, buffet table, cocktail table, lounge seating, arch, pipe-and-drape)
- For items without 3D models, generate simple geometric placeholders (colored boxes with labels)

### 2.5 Design System
- Build a small component library in `packages/ui/`: Button, Input, Select, Card, Modal, Badge, Avatar, Tooltip, Toast notifications, DataTable, EmptyState
- Consistent spacing, typography, and color tokens
- Loading skeletons for all data-fetching views

### 2.6 Tests for Phase 2
- Component tests for all UI components (React Testing Library)
- E2E tests for critical flows: create venue, create event, send proposal (Playwright)

**Phase 2 exit criteria**: A venue owner can sign up, create a venue, manage events, build proposals, and track bookings — all through a polished UI. No 3D floor plan editor yet.

---

## Phase 3: 2D Floor Plan Editor

**Goal**: A fully functional 2D drag-and-drop floor plan editor. This is the core product differentiator — invest heavily here.

### 3.1 Canvas Setup
- Use HTML Canvas (via Konva.js/react-konva) OR SVG for the 2D editor — choose based on what's already in the codebase, or Konva if greenfield (better for interactive manipulation)
- Grid-based canvas with configurable grid snap (1ft, 6in increments)
- Zoom (scroll wheel + pinch), pan (middle-click drag or two-finger), and minimap
- Ruler/measurement display along edges showing dimensions in feet

### 3.2 Floor Plan Drawing
- Draw room boundaries (walls) — rectangular rooms at minimum, L-shapes as stretch
- Set room dimensions numerically (input fields) or by drawing
- Upload background image (venue blueprint/photo) and scale it to real dimensions
- Multiple rooms/zones per floor plan

### 3.3 Furniture Placement
- Drag items from the furniture catalog sidebar onto the floor plan
- Snap to grid, snap to alignment guides (like Figma's smart guides)
- Rotate (free rotation + snap to 0/45/90), scale (proportional)
- Multi-select (click-drag rectangle, shift-click), group move/rotate
- Copy/paste, duplicate, delete
- Right-click context menu (duplicate, delete, lock, send to back/front)
- Undo/redo (Ctrl+Z / Ctrl+Shift+Z) with at least 50-step history

### 3.4 Smart Features
- **Auto-count**: live counter showing total seats, tables, standing capacity
- **Capacity validation**: warn when guest count exceeds floor plan capacity
- **Spacing checker**: highlight when furniture is placed too close together (configurable minimum spacing, default 3ft between table edges)
- **Fire code helper**: basic checks — clear path to exits (if exits are marked), aisle width minimums

### 3.5 Floor Plan Templates
- Save any floor plan as a reusable template
- Pre-built templates: classroom, banquet rounds, theater, cocktail, U-shape, boardroom, ceremony
- Apply template to a room, then customize

### 3.6 Export & Sharing
- Export floor plan as PNG/PDF (with dimensions, furniture counts, legend)
- Generate shareable view-only link for clients
- Print-optimized view (scaled to paper size with title block)

### 3.7 Tests for Phase 3
- Unit tests for geometry helpers (snap, collision, spacing calculations)
- Integration tests for undo/redo stack
- E2E tests: place furniture, move it, undo, export

**Phase 3 exit criteria**: Users can create floor plans, place and arrange furniture, get live capacity counts, and export/share layouts. Editor feels responsive and intuitive.

---

## Phase 4: 3D Visualization

**Goal**: Add a 3D view of floor plans that users can toggle into. NOT replacing the 2D editor — augmenting it.

### 4.1 Three.js / R3F Setup
- Install Three.js (latest r182+) and React Three Fiber v9
- Set up WebGPU renderer with automatic WebGL fallback
- `await renderer.init()` pattern for async WebGPU initialization
- Performance monitoring with `<PerformanceMonitor>` from drei

### 4.2 Scene Construction
- Convert 2D floor plan data into a 3D scene:
  - Room: extruded walls (8ft default height), floor plane with optional texture
  - Furniture: load GLB models from furniture catalog, position/rotate to match 2D layout
  - For items without GLB models: render as colored 3D boxes with label text
- Lighting: ambient + directional sun, optional configurable accent lights
- Ground plane with subtle grid

### 4.3 3D Navigation
- Orbit controls (mouse drag to rotate, scroll to zoom)
- First-person walkthrough mode (WASD + mouse look) — optional, implement if time allows
- Preset camera angles: top-down, entrance view, stage view
- Smooth animated transitions between camera positions

### 4.4 Bidirectional Sync
- Changes in 2D editor instantly reflected in 3D view
- Optionally allow object selection in 3D view (click to select, highlight in both views)
- Do NOT implement drag-and-drop in 3D — keep editing in 2D, viewing in 3D

### 4.5 Performance Optimization
- `InstancedMesh` for repeated furniture items (all chairs as one draw call, etc.)
- LOD (Level of Detail) for complex models — simplified meshes when zoomed out
- Frustum culling (R3F handles this by default)
- Target: < 200 draw calls, < 500K visible triangles, 60fps on mid-range laptop
- Cap device pixel ratio: 1.0 desktop, 1.5 mobile
- Lazy-load 3D view — don't initialize Three.js until user clicks "3D View" tab

### 4.6 Asset Pipeline
- Accept GLB/GLTF uploads for furniture models
- On upload: validate, generate thumbnail, extract dimensions
- Compress with Draco (geometry) and KTX2 (textures) if build pipeline supports it — otherwise defer to a future optimization pass
- Store processed assets in object storage (S3/R2)

### 4.7 Tests for Phase 4
- Unit tests for 2D-to-3D coordinate conversion
- Visual regression tests (screenshot comparison) if feasible
- Performance benchmark test: load a 200-object scene, assert < 200 draw calls

**Phase 4 exit criteria**: Users can toggle between 2D editor and 3D visualization. 3D view accurately reflects the floor plan. Performance is smooth on modern hardware.

---

## Phase 5: Real-Time Collaboration

**Goal**: Multiple users can edit the same floor plan simultaneously, Figma-style.

### 5.1 Yjs Integration
- Install Yjs v13.6+ and set up a Y.Doc per floor plan
- Data model: `Y.Map` per furniture object with flat keys (`posX`, `posY`, `rotation`, `type`, `catalogItemId`, etc.)
- `Y.Array` for the list of objects in the floor plan
- Use `YKeyValue` from `y-utility` for frequently updated properties to minimize doc size

### 5.2 WebSocket Backend
- Set up Hocuspocus server (can run as a separate Node process or integrated into Next.js custom server)
- Authentication hook: validate JWT, check user has edit access to this floor plan
- Persistence hook: save Y.Doc snapshots to database on change (debounced, every 5-10 seconds)
- Load hook: restore Y.Doc from database on connection

### 5.3 Presence & Awareness
- Show connected users' avatars in the editor toolbar
- Display other users' cursors on the 2D canvas (colored by user)
- Show selection highlights for objects other users are currently manipulating
- Throttle awareness updates to 10-15Hz

### 5.4 Conflict Handling
- CRDT handles merge automatically — no manual conflict resolution needed
- Add visual indicator when an object was moved by another user (brief flash/highlight)
- Lock objects while being dragged (optimistic — release if user disconnects)

### 5.5 Offline Support
- Use `y-indexeddb` for local persistence
- Queue changes while offline, sync automatically on reconnect
- Show connection status indicator (connected/reconnecting/offline)

### 5.6 Tests for Phase 5
- Integration tests: two clients connect, one moves an object, other sees the change
- Offline/reconnection test: disconnect, make changes, reconnect, verify sync
- Load test: 10 concurrent editors on one floor plan

**Phase 5 exit criteria**: Two or more users can edit the same floor plan simultaneously with live cursors, presence, and no data loss. Offline editing works.

---

## Phase 6: Polish, Accessibility & Production Readiness

**Goal**: Take everything from "works" to "works well, looks great, is accessible, and is production-ready."

### 6.1 Accessibility
- All interactive elements keyboard-accessible
- For 2D editor: Tab through furniture items, arrow keys to nudge, Enter to select, Delete to remove
- For 3D view: provide a parallel accessible table/list view of all objects with their positions
- WCAG 2.2 AA color contrast throughout
- `aria-label` and `aria-live` regions for dynamic content (capacity counter, save status)
- Dragging alternatives: number input fields for x, y, rotation as alternative to drag-and-drop (WCAG 2.5.7)
- Screen reader announcements for key actions ("Table 5 moved to position 12, 8")
- Color-blind safe palette for object categories (Okabe-Ito palette)
- `prefers-reduced-motion` support: disable canvas animations, 3D camera transitions
- High contrast mode support

### 6.2 Performance Audit
- Lighthouse performance score > 90 on all pages
- Core Web Vitals: LCP < 2.5s, FID < 100ms, CLS < 0.1
- Bundle analysis — code-split aggressively, lazy-load 3D libraries
- Image optimization: WebP/AVIF with fallbacks, responsive srcset
- Database query optimization: add indexes, analyze slow queries
- API response caching where appropriate (venue catalog, furniture catalog)

### 6.3 Error Handling & Edge Cases
- Global error boundary with user-friendly error page
- Toast notifications for save success/failure
- Auto-save with visual indicator (saved/saving/error)
- Handle session expiry gracefully (re-auth without losing work)
- Rate limit error handling (retry with backoff)
- Large floor plan handling (> 500 objects — test and optimize or warn)

### 6.4 Onboarding
- First-time user guided tour (5 steps max): create venue → upload blueprint → place first table → see capacity count → share with client
- Endowed progress: start onboarding at step 2 ("Account created ✓")
- Empty states with clear CTAs on all list views
- Tooltip hints for non-obvious features (keyboard shortcuts, snap behavior)
- Sample venue with pre-built floor plan for new users to explore

### 6.5 Security Audit
- Input sanitization on all user inputs
- File upload validation (type, size, malware scanning if feasible)
- SQL injection prevention (parameterized queries — should already be handled by ORM)
- XSS prevention (React handles most of this, but audit dangerouslySetInnerHTML usage)
- CSRF protection on state-changing API routes
- Secure headers (CSP, HSTS, X-Frame-Options)
- Shareable link tokens: cryptographically random, expirable, revocable

### 6.6 DevOps & Deployment
- Docker Compose for local development (app + database + redis)
- CI pipeline: lint → type-check → test → build → deploy
- Database migration strategy (up/down migrations, seed data)
- Environment configuration for staging and production
- Health check endpoint
- Structured logging (JSON format, request IDs)
- Error monitoring setup (Sentry or similar)

### 6.7 Documentation
- `README.md` with setup instructions, architecture overview, env var reference
- API documentation (auto-generated from Zod schemas or OpenAPI spec)
- `CONTRIBUTING.md` for development workflow

**Phase 6 exit criteria**: App is accessible (WCAG 2.2 AA), performant (Lighthouse > 90), secure, deployable, documented, and onboarding guides new users effectively.

---

## CLAUDE.md Template

Place this at the project root. Claude Code reads it automatically.

```markdown
# CLAUDE.md

## Quick Commands
- `pnpm dev` — start development server
- `pnpm test` — run all tests
- `pnpm lint` — lint and type-check
- `pnpm db:migrate` — run database migrations
- `pnpm db:seed` — seed database with sample data

## Architecture
- Next.js 15 App Router in `apps/web/`
- Shared UI components in `packages/ui/`
- Database layer in `packages/db/`
- All API routes in `apps/web/app/api/`

## Code Style
- TypeScript strict mode, no `any` types
- Zod for all input validation — schemas in `packages/shared/schemas/`
- Server Components by default, `"use client"` only when needed
- Named exports, no default exports (except pages)
- Error handling: never swallow errors, always log and surface to user
- All database queries go through the ORM, never raw SQL in route handlers

## Common Mistakes to Avoid
- Don't import Three.js in Server Components — always lazy-load with dynamic import
- Don't store Y.Doc state in React state — use Yjs bindings
- Don't use `useEffect` for data fetching — use Server Components or React Query
- Always dispose Three.js resources in cleanup functions
- Always validate file uploads server-side, not just client-side

## Testing
- Run `pnpm test` before committing
- New features require at least one integration test
- 2D editor changes require a geometry unit test
- API routes require request/response validation tests

## Status
See PROGRESS.md for current implementation status.
```

---

## Session Management Rules for Claude Code

1. **One phase per session.** Don't try to do Phase 1-6 in one go.
2. **Start each session**: "Read VENUE_PLATFORM_MASTER_COMMAND.md and PROGRESS.md. What phase am I on? Execute the next incomplete phase."
3. **End each session**: Update PROGRESS.md with completed items, dump any context or decisions into a `DECISIONS.md` log.
4. **Clear context** at ~30% usage (~60K tokens). Use the Document & Clear pattern — write progress to a file, `/clear`, resume.
5. **Commit after each sub-section** (e.g., after 1.1, after 1.2, etc.) — not after the entire phase.
6. **Tests before features**: Write failing tests first, then implement to make them pass.
7. **Diffs under 200 lines** per commit. If a task will produce more, break it into sub-tasks.
8. **Don't gold-plate early phases.** Phase 1-2 UI can be functional but plain. Phase 6 is for polish.
