# TRANSCENDENT_ENGINEERING.md — Portfolio-Grade Technical Additions

> **Purpose**: Supplementary to VENUE_PLATFORM_MASTER_COMMAND.md. These are advanced
> engineering additions that elevate the project from "good app" to "this person thinks
> like a systems engineer." Feed to Claude Code AFTER Phase 2 of the main command is
> complete — you need the foundation first.
>
> **What impresses elite firms**: Not feature count. Not UI polish. They want to see
> algorithmic thinking, novel data structures, correctness guarantees, performance
> engineering, and the ability to build abstractions that don't exist yet.

---

## T1: Constraint Solver for Automatic Layout Generation

**Why this is impressive**: You're building a SAT/constraint-satisfaction solver — the
same class of problem that powers SMT solvers, circuit design, and compiler optimization.
Nobody in the venue space has this. It's genuinely novel.

### What to Build

An automatic layout engine that takes high-level intent and produces optimal furniture
arrangements, respecting physical and regulatory constraints.

**Input**: "120 guests, banquet style, stage at north wall, 2 bars, dance floor center,
wheelchair accessible aisles, fire code compliant"

**Output**: A valid, optimized furniture layout — positions and rotations for every object.

### Implementation

Build a **constraint propagation engine with backtracking search**:

```
Constraints (hard — must satisfy):
- No furniture overlaps (AABB + rotated OBB collision)
- Minimum aisle width between furniture groups (ADA: 36", fire code: 44")
- Clear path from every seat to nearest exit (pathfinding)
- Furniture within room boundaries
- Stage/bar/dance floor at specified positions or wall adjacency
- Table-to-chair associations (each table gets N chairs around it)

Objectives (soft — optimize):
- Minimize wasted space (maximize usable area ratio)
- Maximize sightlines to stage/focal point (raycasting)
- Balance server access paths (every table reachable from kitchen)
- Minimize guest travel distance to amenities (bars, bathrooms)
- Aesthetic symmetry score
```

**Algorithm approach — implement in this order**:

1. **Grid discretization**: Divide room into cells (6-inch resolution). Each cell is
   empty, occupied, or blocked (wall/exit).

2. **Arc consistency (AC-3)**: For each furniture piece, compute its valid placement
   domain (all cells where it fits without violating hard constraints). Propagate
   constraints to prune domains.

3. **Backtracking with heuristics**:
   - Variable ordering: MRV (Minimum Remaining Values) — place most constrained items first
   - Value ordering: LCV (Least Constraining Value) — try positions that leave the most
     options for remaining items
   - Conflict-directed backjumping — when stuck, jump back to the variable that caused
     the conflict, not just the previous one

4. **Simulated annealing post-pass**: After finding a valid layout, optimize soft
   objectives by randomly perturbing furniture positions and accepting improvements
   (or occasionally worse positions to escape local minima).

5. **Multi-objective Pareto frontier**: Run the optimizer multiple times with different
   weight vectors, present the user with 3-5 distinct layout options along the
   Pareto frontier (e.g., "maximum capacity" vs "best sightlines" vs "most accessible").

### Rust WASM for Performance

The constraint solver MUST run in a Web Worker compiled from Rust to WASM. This is both
a performance necessity (solving 200+ objects is computationally expensive) and a portfolio
signal (you can write Rust, you understand WASM, you know when to reach for native perf).

```
crates/
  layout-solver/
    src/
      lib.rs          — Public API (solve, validate, score)
      constraints.rs  — Hard constraint definitions + checkers
      objectives.rs   — Soft objective scoring functions
      solver.rs       — AC-3 + backtracking + simulated annealing
      geometry.rs     — OBB collision, raycasting, pathfinding (A*)
      grid.rs         — Discretized room representation
    Cargo.toml        — wasm-bindgen, serde, getrandom/js
```

Build pipeline: `cargo build --target wasm32-unknown-unknown` → `wasm-bindgen --target web`
→ `wasm-opt -O3`

Expose to JS as an async function that posts progress updates (% complete, current best
score) back to the main thread via `postMessage`.

### The "Wow" Demo

User clicks "Auto Layout", types natural language intent, watches furniture animate into
position over 2-3 seconds as the solver converges. Show a real-time score dashboard:
capacity utilization %, accessibility score, sightline coverage %, fire code compliance
(pass/fail per exit).

---

## T2: Incremental Computation Framework (à la Jane Street's Incremental)

**Why this is impressive**: Jane Street literally built a library called `Incremental`
(open source in OCaml) that is central to their trading systems. Building an analogous
system in TypeScript for your venue engine shows you understand their core engineering
philosophy: don't recompute what hasn't changed.

### What to Build

A reactive dependency graph that tracks which computations depend on which inputs, and
only recomputes the minimum necessary subgraph when an input changes.

### Where It Applies

Your venue editor has dozens of derived computations:
- Total capacity (depends on: all furniture positions + types)
- Fire code compliance (depends on: furniture positions + exit locations + aisle widths)
- Sightline scores (depends on: furniture positions + stage location)
- Cost estimate (depends on: furniture counts + pricing data)
- Collision set (depends on: all furniture bounding boxes)
- Accessibility paths (depends on: furniture positions + door locations)

Currently, if you move ONE chair, you'd naively recompute ALL of these. With incremental
computation, moving a chair only recomputes the collision set for nearby objects, the
capacity count for that table, and the aisle width check for adjacent paths.

### Implementation

```typescript
// Core API — inspired by Jane Street's Incremental
const graph = new IncrementalGraph();

// Input nodes (things the user changes)
const furniturePositions = graph.input<Map<string, Position>>(initialPositions);
const roomBoundary = graph.input<Polygon>(initialBoundary);
const exitLocations = graph.input<Point[]>(initialExits);

// Derived nodes (automatically recomputed when dependencies change)
const spatialIndex = graph.derive([furniturePositions], (positions) => {
  return buildRTree(positions);  // Only rebuilds when positions change
});

const collisions = graph.derive([spatialIndex], (index) => {
  return findAllCollisions(index);
});

const aisleWidths = graph.derive([spatialIndex, roomBoundary], (index, boundary) => {
  return computeMinAisleWidths(index, boundary);
});

const fireCodeCompliance = graph.derive(
  [aisleWidths, exitLocations, furniturePositions],
  (aisles, exits, positions) => {
    return checkFireCode(aisles, exits, positions);
  }
);

// When user moves a chair:
furniturePositions.set(newPositions);
graph.stabilize();  // Only recomputes the affected subgraph
// fireCodeCompliance.value is now up-to-date
```

### Internal Architecture

- **Dependency DAG**: Nodes are computations, edges are data dependencies
- **Height-based scheduling**: Process nodes in topological order (lowest height first)
  to ensure all dependencies are fresh before recomputing a node
- **Change propagation with cutoff**: If a node recomputes but produces the same value
  as before (structural equality), don't propagate further — "cutoff"
- **Adjustable nodes**: Input nodes that the user can change at any time
- **Observer nodes**: Leaf nodes that trigger side effects (UI updates, validation
  warnings, analytics)
- **Batch stabilization**: Collect multiple input changes, then stabilize once —
  critical for drag operations where position changes 60x/second

### Testing

Property-based tests (use fast-check):
- **Consistency**: After stabilize(), every derived node's value equals what you'd get
  from recomputing from scratch
- **Minimality**: Count recomputations; assert that only nodes in the affected subgraph
  were recomputed
- **Idempotency**: Calling stabilize() twice without input changes produces zero
  recomputations
- **Commutativity**: Changing inputs A then B produces the same result as B then A

---

## T3: Custom Spatial CRDT for Collaborative Editing

**Why this is impressive**: Instead of just using Yjs (which anyone can npm install),
you design and implement a CRDT specifically optimized for spatial data. This shows
distributed systems knowledge at a theoretical level.

### The Problem with Generic CRDTs for Spatial Data

Yjs Y.Map stores key-value pairs with last-writer-wins semantics. This means:
- If User A moves a table to (10, 5) and User B simultaneously moves it to (3, 8),
  one user's move is silently lost
- No concept of "spatial intent" — the system can't merge "A moved it left, B moved
  it up" into a combined displacement

### What to Build

A **Spatial Intent CRDT** that merges concurrent moves as displacement vectors rather
than absolute positions:

```
User A: move(table1, dx: -5, dy: 0)   — "move left 5"
User B: move(table1, dx: 0, dy: +3)   — "move up 3"
Merged result: table1 position += (-5, +3)  — both intents preserved
```

For concurrent operations on DIFFERENT properties (A moves, B rotates), both apply
cleanly (commutative).

For concurrent operations on the SAME property with conflicting intent (A moves left,
B moves right), use a **vector average with recency weighting** — more recent
operations get slightly higher weight, converging toward the last-active user's intent
while preserving some of both.

### Implementation

```
packages/
  spatial-crdt/
    src/
      clock.ts           — Hybrid Logical Clock (HLC) for causal ordering
      operation.ts        — Operation types (MoveOp, RotateOp, ScaleOp, AddOp, RemoveOp)
      state.ts            — Per-object state with vector clock per property
      merge.ts            — Merge function: commutative, associative, idempotent
      document.ts         — Collection of objects, operation log, state reconstruction
      sync.ts             — State vector exchange, delta sync protocol
      serialization.ts    — Binary encoding for network efficiency (not JSON)
```

### Mathematical Properties to Prove (via property-based tests)

These are the CRDT correctness properties. Proving them is what makes this impressive:

1. **Strong Eventual Consistency**: Two replicas that have received the same set of
   operations (in any order) have identical state
2. **Commutativity**: merge(A, B) === merge(B, A)
3. **Associativity**: merge(merge(A, B), C) === merge(A, merge(B, C))
4. **Idempotency**: merge(A, A) === A
5. **Convergence**: All replicas converge regardless of operation delivery order,
   including partitions and message reordering

Write these as fast-check properties that generate random operation sequences and
verify the properties hold across thousands of random test cases.

### Use Alongside Yjs (Not Instead Of)

Use your spatial CRDT for furniture positions/rotations (the novel part), and standard
Yjs for everything else (metadata, chat, non-spatial state). This is pragmatic
engineering — build novel solutions where they add value, use battle-tested libraries
everywhere else.

---

## T4: WebGPU Compute Shaders for Real-Time Spatial Analysis

**Why this is impressive**: GPU compute for non-rendering tasks shows you understand
parallel computation, SIMD thinking, and hardware-aware optimization. Most web devs
have never touched compute shaders.

### What to Build

Move these computations to WebGPU compute shaders (with CPU fallback):

1. **Parallel collision detection**: Check all furniture pairs simultaneously on GPU.
   N objects = N*(N-1)/2 pairs, each checked independently — embarrassingly parallel.
   Returns a collision bitmask in one dispatch.

2. **Sightline analysis**: For each seat, cast rays toward the stage/focal point.
   Report percentage of seats with unobstructed view. Visualize as a heatmap overlay
   on the floor plan (green = clear sightline, red = obstructed).

3. **Crowd flow simulation**: Agent-based model where each guest is a particle.
   Simulate evacuation: all agents pathfind to nearest exit simultaneously. Identify
   bottlenecks where agent density exceeds threshold. This is a real safety analysis
   tool, not a gimmick.

4. **Acoustic ray tracing**: Cast sound rays from speaker/stage positions, bounce off
   walls (reflection), identify dead spots and echo points. Visualize as a volume
   heatmap. This is genuinely novel for a web-based venue tool.

### Implementation Pattern

```typescript
// All compute shaders follow this pattern:
// 1. Upload data to GPU storage buffers
// 2. Dispatch compute shader
// 3. Read results back to CPU

// Example: Parallel collision detection
const collisionShader = /* wgsl */`
  struct AABB { minX: f32, minY: f32, maxX: f32, maxY: f32 }

  @group(0) @binding(0) var<storage, read> boxes: array<AABB>;
  @group(0) @binding(1) var<storage, read_write> collisions: array<u32>;

  @compute @workgroup_size(64)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let n = arrayLength(&boxes);
    if (i >= n) { return; }

    for (var j = i + 1u; j < n; j++) {
      if (aabbOverlap(boxes[i], boxes[j])) {
        atomicOr(&collisions[i], 1u << j);
      }
    }
  }
`;
```

Each analysis produces a visual overlay that can be toggled on/off in the editor.
The crowd flow sim should animate in real-time (agents moving as particles).

**CPU fallback**: Every compute shader must have an equivalent TypeScript implementation
that runs when WebGPU is unavailable. Same API, same results, just slower.

---

## T5: Property-Based Testing & Formal Verification Thinking

**Why this is impressive**: Jane Street uses property-based testing extensively
(Quickcheck in OCaml). Demonstrating this discipline — and having it catch real bugs —
is a massive signal.

### What to Test with fast-check

Install `fast-check` and write property-based tests for:

**Event sourcing correctness**:
```typescript
// Property: projecting events in any order produces the same state
fc.assert(fc.property(
  fc.array(arbitraryDomainEvent()),
  (events) => {
    const state1 = projectState(events);
    const state2 = projectState(shuffle(events));
    expect(state1).toDeepEqual(state2);
  }
));

// Property: undo(apply(event)) === original state
fc.assert(fc.property(
  arbitraryState(), arbitraryDomainEvent(),
  (state, event) => {
    const after = applyEvent(state, event);
    const restored = applyEvent(after, computeInverse(event));
    expect(restored).toDeepEqual(state);
  }
));
```

**Constraint solver correctness**:
```typescript
// Property: solver output always satisfies all hard constraints
fc.assert(fc.property(
  arbitraryRoomConfig(), arbitraryFurnitureSet(),
  (room, furniture) => {
    const layout = solve(room, furniture);
    if (layout === null) return true;  // unsatisfiable is valid
    expect(checkAllHardConstraints(room, layout)).toBe(true);
  }
));
```

**Spatial CRDT correctness** (see T3 above for the four properties).

**Incremental framework correctness** (see T2 above for properties).

**Geometry utilities**:
```typescript
// Property: rotating a shape 360 degrees returns to original
// Property: AABB of a rotated rectangle contains all corners
// Property: point-in-polygon is consistent with winding number
```

### Target: 500+ Property-Based Test Cases

This isn't vanity — each property test runs hundreds of random inputs. If your suite has
50 properties × 100 iterations each, that's 5,000 test executions finding edge cases
that unit tests never would. Document any bugs found by property tests in a
`PROPERTY_TEST_FINDINGS.md` — this is gold for interviews.

---

## T6: Custom Binary Protocol for Real-Time Sync

**Why this is impressive**: Designing a wire protocol shows systems-level thinking.
JSON over WebSocket is what everyone does. A custom binary protocol that's 10x smaller
and faster is what systems engineers do.

### What to Build

A compact binary encoding for spatial operations optimized for venue editing:

```
Operation wire format (variable length, ~12-40 bytes vs ~200+ bytes JSON):

[1 byte]  op_type (0x01=move, 0x02=rotate, 0x03=add, 0x04=remove, 0x05=resize)
[8 bytes] hlc_timestamp (hybrid logical clock — 48-bit wall clock + 16-bit counter)
[4 bytes] object_id (uint32 entity ID from ECS, not full UUID on wire)
[payload] varies by op_type:
  move:   [4B float32 dx] [4B float32 dy] [4B float32 dz]  = 12 bytes
  rotate: [4B float32 angle]                                 = 4 bytes
  add:    [2B type_enum] [4B x] [4B y] [4B z] [4B rotation] = 18 bytes
  remove: (no payload)                                        = 0 bytes

Total move operation: 1 + 8 + 4 + 12 = 25 bytes (vs ~180 bytes JSON)
```

### Implementation

```
packages/
  wire-protocol/
    src/
      encoder.ts   — Operation → ArrayBuffer (DataView writes)
      decoder.ts   — ArrayBuffer → Operation (DataView reads)
      batch.ts     — Batch multiple operations into one frame with length prefixes
      compress.ts  — Delta encoding for position updates (store diff from last known)
      benchmark.ts — Compare encode/decode speed and size vs JSON.stringify/parse
```

Write a benchmark that processes 10,000 operations and compares:
- Size: binary vs JSON (expect ~7-10x reduction)
- Encode speed: binary vs JSON.stringify (expect ~3-5x faster)
- Decode speed: binary vs JSON.parse (expect ~2-4x faster)

Include the benchmark results in the README with actual numbers.

---

## T7: Time-Travel Debugger with Branching Timelines

**Why this is impressive**: You already have event sourcing. A visual time-travel
debugger that lets you scrub through the entire history of a floor plan, branch into
alternative timelines, and diff between branches is technically sophisticated and
visually stunning.

### What to Build

A timeline UI (think video editor scrubber) at the bottom of the venue editor:

- **Scrub bar**: Drag to any point in history. Floor plan reconstructs to that moment
  via event replay (use snapshots every 50 events for performance).
- **Event markers**: Visual dots on the timeline showing when events occurred, colored
  by type (blue=move, green=add, red=remove, yellow=rotate).
- **Branch**: Click "Branch" at any point to create an alternative timeline. Both
  timelines are preserved. Switch between them freely.
- **Diff view**: Select two timeline points (or two branches) and see a visual diff —
  objects that moved are shown with ghost outlines at their old position and arrows
  to their new position. Added objects glow green, removed objects shown as red ghosts.
- **Merge branches**: Select two branches and merge them. Use the spatial CRDT (T3) to
  resolve conflicts. Show the user what was auto-merged and what needs manual resolution.

### The Technical Depth

The event store already exists. The novel part is:
1. **Efficient reconstruction**: Snapshot + replay with binary search for nearest
   snapshot to target timestamp
2. **Branch data structure**: A persistent (immutable) tree where branches share common
   ancestry — don't duplicate the shared event history
3. **Visual diff algorithm**: Compute the set difference of two states, classify each
   object as added/removed/moved/modified, compute displacement vectors for moved objects
4. **Branch merge**: Three-way merge using common ancestor state as base

---

## T8: Performance Observatory

**Why this is impressive**: Building your own profiling and observability system shows
mature engineering thinking. Don't just make it fast — prove it's fast with data.

### What to Build

A developer-facing performance dashboard (toggled via feature flag) showing real-time:

- **Frame budget breakdown**: Stacked bar showing ms spent on: render, physics/collision,
  CRDT sync, UI updates, idle — per frame, rolling 120-frame window
- **Memory waterfall**: Track GPU memory (geometries, textures, buffers), JS heap, and
  WASM linear memory over time. Alert on leaks (monotonically increasing allocation
  without corresponding frees).
- **Network efficiency**: Operations/second, bytes/second, compression ratio,
  round-trip latency histogram
- **Solver metrics** (for T1): Constraint propagation steps, backtrack count, solution
  quality score over time as annealing progresses
- **Incremental graph stats** (for T2): Nodes stabilized per cycle, cutoff ratio
  (what % of nodes were skipped), max propagation depth

Export all metrics as JSON for inclusion in documentation / README performance section.

---

## T9: NVIDIA Cosmos World Foundation Model Integration

**Why this is impressive**: You're integrating a frontier AI world model into a
production web app. Nobody in the venue space — or almost any web app space — is doing
this. It's the kind of thing that makes someone say "wait, that's running in a browser?"

### What to Build

A "Photorealistic Preview" feature: the user designs a layout in the 2D/3D editor,
clicks "Generate Realistic Preview," and receives a photorealistic video walkthrough
of their venue configuration generated by NVIDIA Cosmos.

### Pipeline

```
User's floor plan state
  → Render a depth map + segmentation map from the Three.js scene (server-side or via
    OffscreenCanvas)
  → Send to Cosmos Transfer2.5 API (converts structured spatial input to photorealistic
    video)
  → Stream result back to client as progressive MP4
  → Display in a modal with playback controls

Alternative (simpler, text-only):
  → Serialize floor plan to natural language description:
    "A 21m x 10m hall with arched ceiling. 12 round tables of 8 seating 96 guests.
     Stage with podium at the north wall. Two bars flanking the entrance. Warm
     pendant lighting. Hardwood floors."
  → Send to Cosmos Predict2.5 (text-to-world video generation)
  → Display generated walkthrough video
```

### Implementation

```
apps/
  web/
    src/
      app/api/cosmos/
        generate/route.ts    — POST: accepts scene description or depth/seg maps,
                               calls NIM endpoint, returns job ID
        status/[id]/route.ts — GET: poll job status, return video URL when complete
      components/
        cosmos/
          RealisticPreview.tsx   — Modal with generate button, progress bar, video player
          SceneSerializer.ts     — Convert floor plan state to text description
          DepthMapRenderer.ts    — Render Three.js scene to depth map (WebGL readPixels)
          SegmentationMap.ts     — Color-code objects by category for Transfer2.5 input
```

### Infrastructure

**NIM deployment** (NVIDIA Inference Microservices):
- Docker container with Cosmos model + Triton Inference Server
- Requires H100 or H200 GPU (use cloud: AWS p5 instances, Azure ND H100, or
  NVIDIA DGX Cloud)
- RESTful API — POST video generation request, poll for completion
- Async pattern: generation takes 10-60 seconds depending on length/resolution
- Use WebSocket to push completion notification to client (don't poll)

**Cost management**:
- Gate behind premium tier (this is a paid feature)
- Queue requests, limit concurrent generations
- Cache results keyed by floor plan state hash — same layout = same video, no regen
- Offer 720p default, 1080p for premium

### Fallback for Development

During development without GPU access, mock the Cosmos API:
- Return a pre-recorded sample video after a fake delay
- Build the entire UI and pipeline against the mock
- Swap in real NIM endpoint via environment variable when GPU infra is available

---

## T10: Omniverse Kit Streaming for RTX-Quality Interactive 3D

**Why this is impressive**: Server-side RTX rendering streamed to the browser via
WebRTC. The user gets ray-traced reflections, global illumination, and physically
accurate materials — things that are literally impossible in client-side WebGL/WebGPU.
This is what NVIDIA demos at GTC to standing ovations.

### What to Build

A "Cinematic View" mode: toggle from the standard Three.js view to a server-rendered
RTX viewport streamed in real-time. The user can orbit, zoom, and walk through the
venue with photorealistic rendering — real-time reflections on polished floors,
accurate soft shadows, realistic material response.

### Implementation

```
apps/
  web/
    src/
      components/
        omniverse/
          OmniverseStream.tsx     — Wrapper around AppStreamer React component
          StreamControls.tsx      — Quality selector, connection status, latency display
          SceneSyncBridge.ts      — Bidirectional sync: local state ↔ Omniverse scene
                                    (send furniture positions via JSON messaging,
                                     receive camera state back)

infra/
  omniverse/
    Dockerfile           — Kit App Streaming container config
    app.kit              — Omniverse Kit application configuration
    extensions/
      venue-scene/       — Custom Kit extension that loads USD scenes from our data
    helm/
      values.yaml        — Kubernetes Helm chart for auto-scaling GPU instances
```

### How It Works

1. User clicks "Cinematic View"
2. Client requests a streaming session from the orchestrator
3. Orchestrator spins up (or assigns from pool) a Kit App Streaming instance on an
   RTX GPU
4. `@nvidia/omniverse-webrtc-streaming-library` AppStreamer component connects via
   WebRTC
5. Client sends the current floor plan state as JSON to the Kit instance
6. Kit extension loads/updates the USD scene to match
7. User sees RTX-rendered viewport, can orbit and navigate
8. Mouse/keyboard events forwarded to server, rendered frames streamed back
9. Any changes in the local 2D/3D editor are pushed to the Kit scene in real-time

### Infrastructure

- Each concurrent user in Cinematic View needs **one RTX GPU** (A10G minimum, A100/H100
  for best quality)
- Deploy on Kubernetes with GPU node pools (AWS g5/p4, Azure NC A100, GCP A2)
- Auto-scale: spin up GPU instances on demand, scale to zero when idle
- Session timeout: 15 minutes of inactivity → release GPU
- Target latency: < 100ms local network, < 150ms cloud

### USD Scene Pipeline

```
Floor plan JSON state
  → Convert to OpenUSD scene (Python script using pxr library)
  → Store in object storage (S3/R2)
  → Kit extension loads USD via Omniverse Nucleus or direct URL
  → Apply materials from a curated USD material library
    (wood floors, fabric chairs, metal fixtures, glass, etc.)
```

### Fallback

When Omniverse infrastructure isn't available (most of development):
- "Cinematic View" button shows a tooltip: "RTX streaming available in production"
- Build all the UI, connection logic, and scene sync against a mock WebRTC stream
  (static image or pre-recorded video loop)
- The Three.js view remains the primary 3D experience

---

## T11: NVIDIA ACE Digital Human Venue Concierge

**Why this is impressive**: An AI-powered photorealistic digital human that answers
questions about your venue in real-time video conversation. This is science fiction
made real. Even if it's a demo/prototype, it shows you can integrate cutting-edge
multimodal AI pipelines.

### What to Build

A "Venue Concierge" — a photorealistic digital human avatar that:
- Appears in a video chat widget on the venue's public profile page
- Answers questions about the venue: capacity, pricing, availability, amenities
- Can describe floor plan options: "Our main hall seats 200 banquet-style or 300
  theater-style. Would you like me to show you the layouts?"
- Uses RAG over the venue's data (stored in your database) for accurate answers
- Speaks naturally with lip-synced animation

### Pipeline (NVIDIA ACE/Tokkio 5.0)

```
User speaks into microphone
  → WebRTC audio stream to ACE server
  → NVIDIA Riva ASR (speech-to-text)
  → LLM reasoning (Claude API or NVIDIA NeMo) with RAG context:
      - Venue details, pricing, availability from your database
      - Floor plan descriptions serialized from layout data
      - FAQ content curated by venue owner
  → NVIDIA Riva TTS (text-to-speech, natural voice)
  → Audio2Face-3D (generate facial animation from audio)
  → Render animated avatar (Unreal Engine or Omniverse)
  → WebRTC video stream back to user's browser
```

### Implementation

```
apps/
  web/
    src/
      components/
        concierge/
          ConciergeWidget.tsx      — Floating video chat bubble on venue pages
          ConciergeStream.tsx      — WebRTC video/audio connection to ACE
          ConciergeTrigger.tsx     — "Ask our AI concierge" CTA button
      app/api/concierge/
        context/route.ts           — GET: assemble RAG context for a venue (query DB,
                                     serialize floor plans, format for LLM)
        session/route.ts           — POST: create ACE streaming session

packages/
  concierge-rag/
    src/
      context-builder.ts           — Build LLM context from venue data
      prompt-template.ts           — System prompt for the concierge persona
      availability-checker.ts      — Real-time calendar availability lookup
```

### Infrastructure

- ACE/Tokkio 5.0 deployment: 4 GPUs support 3 concurrent streams
- Deploy on DGX Cloud or self-managed K8s with GPU nodes
- Queue system for high traffic: "You're #3 in queue" with estimated wait
- Fallback to text chat (standard Claude API) when video is unavailable

### Practical Reality Check

This is the most infrastructure-heavy feature. Realistic approach:
1. **Phase A**: Build the text-chat version first (Claude API + RAG over venue data).
   This works locally, costs pennies, and is genuinely useful.
2. **Phase B**: Build the UI and WebRTC connection logic against a mock avatar stream.
3. **Phase C**: When GPU infra is available, connect to real ACE deployment.

Phase A alone is a great feature. Phases B and C are the "transcendent" version.

---

## Implementation Order

Do these AFTER the main command's Phase 6 is complete:

1. **T2 (Incremental framework)** — Foundation for everything else. Medium difficulty.
2. **T5 (Property-based testing)** — Add to existing engine tests first. Easy to start.
3. **T1 (Constraint solver)** — The crown jewel. Hard. Build in Rust WASM.
4. **T6 (Binary protocol)** — Before collaboration. Medium difficulty.
5. **T3 (Spatial CRDT)** — During Phase 5 collaboration work. Hard.
6. **T4 (GPU compute)** — After 3D visualization is solid. Hard.
7. **T7 (Time-travel debugger)** — After event sourcing is proven. Medium.
8. **T8 (Performance observatory)** — Measures everything else.
9. **T9 (Cosmos world model)** — Start with text-to-video mock pipeline. Medium.
10. **T10 (Omniverse RTX streaming)** — Build UI + sync logic against mock. Hard infra.
11. **T11 (ACE digital concierge)** — Build text chat first, video later. Medium → Hard.

---

## What to Highlight in Interviews

When discussing this project, lead with:

1. "I built a constraint satisfaction solver in Rust/WASM that generates optimal venue
   layouts — same class of algorithm as SAT solvers and compiler optimizers"
2. "I designed a custom CRDT for spatial data that preserves concurrent edit intent
   instead of last-writer-wins, with formal proofs of convergence via property-based
   testing"
3. "I built an incremental computation framework inspired by Jane Street's Incremental
   library to avoid redundant recomputation in a reactive system"
4. "I wrote WebGPU compute shaders for parallel collision detection and crowd flow
   simulation — real-time safety analysis running on the GPU"
5. "The entire event sourcing system has property-based tests proving commutativity,
   inverse correctness, and state convergence across thousands of random inputs"

These are the sentences that make interviewers lean forward.

6. "I integrated NVIDIA Cosmos world foundation models to generate photorealistic
   walkthrough videos from floor plan state — converting structured spatial data into
   video via depth maps and segmentation maps"
7. "The app offers an RTX-quality cinematic view via Omniverse Kit streaming over
   WebRTC, with bidirectional scene sync between the client-side editor and the
   server-side renderer"
8. "I built a RAG-powered AI venue concierge — text chat using Claude API with
   real-time database context, with an NVIDIA ACE avatar pipeline for video
   conversation"
