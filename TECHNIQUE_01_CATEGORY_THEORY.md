# TECHNIQUE_01_CATEGORY_THEORY.md — Category-Theoretic Architecture

> **Purpose**: Feed this to Claude Code AFTER Phase 6 of the main command is complete
> (or after the existing engine is stable). This restructures the core architecture
> using category theory principles — making the system compositional, provably correct,
> and demonstrating the kind of algebraic thinking Jane Street uses daily.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_01_CATEGORY_THEORY.md and implement
> it incrementally, starting from CT-1."

---

## Why This Matters

Category theory is the "mathematics of mathematics" — it studies structure-preserving
transformations. Jane Street's entire codebase is built on these principles (via OCaml's
type system). By restructuring the venue platform's core around categorical concepts,
every subsystem becomes composable, type-safe, and formally reasoned about. This isn't
decoration — it changes how data flows through the entire application.

---

## CT-1: The Venue Planning Category (Foundation)

### What to Build

Define the core domain as an explicit category in TypeScript:

```typescript
// packages/category/src/core.ts

/**
 * A Category consists of:
 * - Objects (types)
 * - Morphisms (structure-preserving maps between objects)
 * - Composition (morphisms compose associatively)
 * - Identity (every object has an identity morphism)
 */

// A morphism from A to B
type Morphism<A, B> = (a: A) => B;

// Category laws (enforced by tests, expressed by types)
// 1. Associativity: compose(f, compose(g, h)) === compose(compose(f, g), h)
// 2. Identity: compose(id, f) === f === compose(f, id)

// The core composition operator
function compose<A, B, C>(f: Morphism<A, B>, g: Morphism<B, C>): Morphism<A, C> {
  return (a: A) => g(f(a));
}

// Identity morphism
function id<A>(): Morphism<A, A> {
  return (a: A) => a;
}
```

### Domain Objects (objects in our category)

```
VenueSpec       — The specification of a venue (capacity, dimensions, amenities)
EventSpec       — The specification of an event (type, guest count, requirements)
FloorPlan       — A spatial arrangement of furniture in a venue
Constraint      — A rule that a layout must satisfy (fire code, capacity, spacing)
Assignment      — A mapping from event requirements to venue resources
Schedule        — A time-indexed collection of assignments
Proposal        — A priced assignment sent to a client
Configuration   — A complete event setup (venue + layout + services + schedule)
```

### Domain Morphisms (arrows between objects)

```
match:        EventSpec → VenueSpec → Compatibility Score
constrain:    FloorPlan → Constraint → ValidatedFloorPlan | Violation
assign:       EventSpec × VenueSpec → Assignment
price:        Assignment → PricedAssignment
schedule:     Assignment × TimeSlot → ScheduledAssignment
compose_svc:  Service × Service → ComposedService
validate:     Configuration → ValidatedConfiguration | Errors
```

The key insight: every operation in the system is a morphism, and morphisms compose.
`validate ∘ schedule ∘ price ∘ assign ∘ match` is a single pipeline from
`(EventSpec, VenueSpec)` to `ValidatedConfiguration`.

### Implementation

Create `packages/category/` with:

```
packages/category/
  src/
    core.ts           — Category, Morphism, compose, id
    objects.ts        — All domain object types (branded types, not just interfaces)
    morphisms.ts      — All domain morphisms as pure functions
    pipeline.ts       — Pipeline builder using composition
    laws.ts           — Category law checkers (for property-based tests)
  __tests__/
    laws.test.ts      — Property-based tests verifying category laws
    pipeline.test.ts  — Pipeline composition tests
```

### Property-Based Tests (Critical)

Use fast-check to verify the category laws hold for ALL morphisms:

```typescript
import fc from 'fast-check';

// Associativity: compose(f, compose(g, h)) === compose(compose(f, g), h)
test('morphism composition is associative', () => {
  fc.assert(fc.property(
    arbitraryEventSpec(),
    (event) => {
      const pipeline1 = compose(match, compose(price, validate));
      const pipeline2 = compose(compose(match, price), validate);
      expect(pipeline1(event)).toDeepEqual(pipeline2(event));
    }
  ));
});

// Identity: compose(id, f) === f
test('identity morphism is neutral', () => {
  fc.assert(fc.property(
    arbitraryFloorPlan(),
    (plan) => {
      const withId = compose(id<FloorPlan>(), validate);
      expect(withId(plan)).toDeepEqual(validate(plan));
    }
  ));
});
```

---

## CT-2: Functors for Data Transformation

### What to Build

A **functor** maps between categories while preserving structure. In practice, this
means: a type-safe transformation layer that converts data between subsystems while
guaranteeing no information is lost or corrupted.

### Concrete Functors in the Venue System

**EventSourceFunctor**: Maps from the Event Sourcing category to the Read Model category

```typescript
// The Event Sourcing category:
//   Objects: aggregate states
//   Morphisms: events (state transitions)

// The Read Model category:
//   Objects: projected views
//   Morphisms: view updates

// The functor F maps:
//   F(AggregateState) = ProjectedView
//   F(event) = viewUpdate
//   Preserving: F(event1 ∘ event2) = F(event1) ∘ F(event2)

interface Functor<F> {
  map<A, B>(fa: F<A>, f: (a: A) => B): F<B>;
}

// EventSourceFunctor preserves composition:
// Projecting event1 then event2 === Projecting (event1 ∘ event2)
// This is THE correctness guarantee for your read models.
```

**SpatialFunctor**: Maps between 2D and 3D representations

```typescript
// Maps 2D floor plan objects to 3D scene objects
// Preserving: spatial relationships, containment, adjacency
// F(move2D(obj, dx, dy)) = move3D(F(obj), dx, dy, 0)
// The functor law guarantees 2D↔3D sync is correct by construction

const SpatialFunctor = {
  // Object mapping: 2D shape → 3D mesh
  mapObject(obj2D: FloorPlanObject): SceneObject3D { ... },

  // Morphism mapping: 2D operation → 3D operation
  mapMorphism(op2D: FloorPlanOperation): SceneOperation3D { ... },

  // Functor law: mapMorphism(op1 ∘ op2) === mapMorphism(op1) ∘ mapMorphism(op2)
};
```

**SerializationFunctor**: Maps between in-memory domain objects and wire format

```typescript
// F(DomainObject) = WireFormat (Protobuf/JSON)
// F(domainOperation) = wireOperation
// Guarantees: deserialize(serialize(x)) === x (round-trip)
// Guarantees: serialize(f(x)) === wireF(serialize(x)) (commuting diagram)
```

### Implementation

Add to `packages/category/src/`:

```
functor.ts            — Functor interface and common functors
event-source-functor.ts — Event → Read Model projection functor
spatial-functor.ts    — 2D ↔ 3D mapping functor
serialization-functor.ts — Domain ↔ Wire format functor
```

### Tests

For each functor, verify the functor laws with property-based tests:

```typescript
// Functor Law 1 (Identity): F(id) === id
// Functor Law 2 (Composition): F(g ∘ f) === F(g) ∘ F(f)

test('SpatialFunctor preserves identity', () => {
  fc.assert(fc.property(arbitraryFloorPlanObject(), (obj) => {
    const mapped = SpatialFunctor.mapMorphism(id<FloorPlanObject>());
    expect(mapped).toEqual(id<SceneObject3D>());
  }));
});

test('SpatialFunctor preserves composition', () => {
  fc.assert(fc.property(
    arbitraryFloorPlanObject(),
    arbitraryFloorPlanOp(),
    arbitraryFloorPlanOp(),
    (obj, op1, op2) => {
      const composed2D = compose(op1, op2);
      const mappedComposed = SpatialFunctor.mapMorphism(composed2D);
      const composedMapped = compose(
        SpatialFunctor.mapMorphism(op1),
        SpatialFunctor.mapMorphism(op2)
      );
      expect(mappedComposed(SpatialFunctor.mapObject(obj)))
        .toDeepEqual(composedMapped(SpatialFunctor.mapObject(obj)));
    }
  ));
});
```

---

## CT-3: Natural Transformations for System Integration

### What to Build

A **natural transformation** is a way to convert between two functors while preserving
their structure. In practice: a principled way to swap out subsystem implementations
(e.g., switch from in-memory to database-backed read models) with a guarantee that
behavior is preserved.

### Concrete Natural Transformations

**Persistence strategy swap**:
```
η: InMemoryReadModel → PostgresReadModel
Where for every event e:
  η(InMemoryProjection(e)) === PostgresProjection(η(e))
```
This naturality condition guarantees that projecting-then-persisting gives the same
result as persisting-then-projecting. You can swap storage backends without changing
behavior.

**Rendering backend swap**:
```
η: WebGLRenderer → WebGPURenderer
Where for every scene operation op:
  η(WebGLRender(op)) === WebGPURender(η(op))
```
The WebGPU/WebGL fallback is a natural transformation — guaranteed to produce the
same visual result regardless of backend.

**Collaboration transport swap**:
```
η: WebSocketTransport → WebRTCTransport
Naturality guarantees message ordering and delivery semantics are preserved.
```

### Implementation

```
packages/category/src/
  natural-transformation.ts  — NaturalTransformation interface
  persistence-nt.ts          — InMemory ↔ Postgres swap
  renderer-nt.ts             — WebGL ↔ WebGPU swap
  transport-nt.ts            — WebSocket ↔ WebRTC swap
```

---

## CT-4: Monoidal Categories for Service Composition

### What to Build

A **monoidal category** has a "tensor product" (⊗) that combines objects. This models
how event services compose: `Catering ⊗ AV ⊗ Decoration = CompleteEventPackage`.

### The Service Composition Algebra

```typescript
// A monoid in the category of event services
interface ServiceMonoid {
  // The tensor product: combine two services
  tensor<A extends Service, B extends Service>(a: A, b: B): ComposedService<A, B>;

  // The unit: the "empty" service (no-op)
  unit: EmptyService;

  // Laws:
  // tensor(a, unit) === a                    (right identity)
  // tensor(unit, a) === a                    (left identity)
  // tensor(a, tensor(b, c)) === tensor(tensor(a, b), c)  (associativity)
}

// Concrete: composing venue services
const weddingPackage = ServiceMonoid.tensor(
  ServiceMonoid.tensor(catering, audioVisual),
  ServiceMonoid.tensor(floristry, photography)
);

// The tensor product handles:
// - Combining pricing (additive with bundle discounts)
// - Merging time requirements (union of setup windows)
// - Composing space requirements (non-overlapping allocation)
// - Joining staff requirements (with conflict detection)
```

### Operads for Complex Composition

An **operad** generalizes the monoidal structure to handle operations with multiple
inputs. Perfect for: "This event needs 1 venue + 2 caterers + 1 AV vendor + 3 decorators,
and their compositions have specific compatibility constraints."

```typescript
// An operad operation: n inputs → 1 output
interface OperadOp<Inputs extends any[], Output> {
  arity: number;  // how many inputs
  compose(...inputs: Inputs): Output;

  // Operad laws:
  // Associativity of nested composition
  // Unit (identity operation)
  // Equivariance (input reordering)
}

// Concrete: the "assemble event" operad
const assembleEvent: OperadOp<
  [VenueBooking, CateringContract, AVSetup, DecorationPlan, StaffSchedule],
  CompleteEventConfiguration
> = {
  arity: 5,
  compose(venue, catering, av, decor, staff) {
    // Verify all services are compatible (time, space, resources)
    // Verify no spatial conflicts (catering setup area vs AV placement)
    // Verify no temporal conflicts (setup windows don't overlap destructively)
    // Return the complete configuration or detailed conflict report
  }
};
```

### Implementation

```
packages/category/src/
  monoidal.ts        — MonoidalCategory, tensor, unit
  service-monoid.ts  — Service composition algebra
  operad.ts          — Operad interface for multi-input composition
  event-operad.ts    — Concrete event assembly operad
```

---

## CT-5: Kan Extensions for Venue-Event Matching

### What to Build

A **Kan extension** formalizes the "best approximation" problem. Given partial
information about what a client wants, the left Kan extension computes the best
possible venue match — this is the mathematically optimal way to do recommendation
when you have incomplete data.

### The Matching Problem as a Kan Extension

```
Client preferences (partial):
  "200 guests, outdoor, budget $10K, summer, near transit"

This defines a functor F: ClientPreferences → DesiredVenueProperties
But the client hasn't specified everything (AV needs, catering style, etc.)

The left Kan extension Lan_F(G) along the "available venues" functor G computes
the best approximation: the venue that satisfies the most specified preferences
while making optimal assumptions about unspecified ones.

Formally: Lan_F(G) is the "closest" venue functor to G that factors through F.
```

### Implementation

```typescript
// packages/category/src/kan-extension.ts

interface KanExtension<F, G> {
  // Left Kan extension: best approximation from below (optimistic match)
  leftKan(preferences: Partial<EventSpec>): RankedVenueMatches;

  // Right Kan extension: best approximation from above (conservative match)
  rightKan(preferences: Partial<EventSpec>): RankedVenueMatches;

  // The universal property: for any other matching h that factors through F,
  // there exists a unique natural transformation from Lan to h.
  // This means our matching is OPTIMAL — no other factorization does better.
}

// Practical implementation uses the coend formula:
// Lan_F(G)(c) = ∫^a Hom(F(a), c) ⊗ G(a)
//
// In venue matching terms:
// For each venue a, compute:
//   compatibility(preferences, venue_a) × quality(venue_a)
// Take the weighted colimit (weighted sum with compatibility as weights)
// This gives the optimal ranking.

function leftKanMatch(
  preferences: Partial<EventSpec>,
  venues: VenueSpec[],
  compatibilityMetric: (pref: Partial<EventSpec>, venue: VenueSpec) => number,
  qualityMetric: (venue: VenueSpec) => number,
): RankedVenueMatches {
  return venues
    .map(venue => ({
      venue,
      score: compatibilityMetric(preferences, venue) * qualityMetric(venue),
      // The coend formula in action
    }))
    .sort((a, b) => b.score - a.score);
}
```

---

## CT-6: Functorial Data Migration (Spivak's Framework)

### What to Build

David Spivak's work on **functorial data migration** replaces traditional ORMs with
category-theoretic database operations. Schema changes become functors between
categories of schemas, and data migration is a natural transformation.

### Application: Schema Evolution Without Data Loss

```typescript
// A database schema IS a category:
//   Objects = tables
//   Morphisms = foreign key relationships

// A schema migration IS a functor between schema categories:
//   F: OldSchema → NewSchema
//   Maps old tables to new tables
//   Maps old relationships to new relationships
//   PRESERVES composition (referential integrity by construction)

interface SchemaFunctor<OldSchema, NewSchema> {
  // Object mapping: which new table does each old table map to?
  mapTable<T extends keyof OldSchema>(table: T): keyof NewSchema;

  // Morphism mapping: how do foreign keys transform?
  mapRelation<A, B>(rel: Relation<OldSchema, A, B>): Relation<NewSchema, ...>;

  // Data migration operators (from Spivak):
  // Δ_F (pullback): migrate data from new schema back to old (backwards compatible reads)
  // Σ_F (left pushforward): migrate data forward (merge/union)
  // Π_F (right pushforward): migrate data forward (product/join)

  pullback<T>(newData: NewSchema[T]): OldSchema[...];
  pushforward<T>(oldData: OldSchema[T]): NewSchema[...];
}
```

This guarantees:
- Schema migrations preserve referential integrity by construction
- Forward and backward compatibility are formalized
- No data loss during migration (provable, not just tested)

### Implementation

```
packages/category/src/
  schema-category.ts       — Schema as category, tables as objects, FKs as morphisms
  schema-functor.ts        — Migration as functor with pullback/pushforward
  migration-builder.ts     — DSL for building type-safe migrations categorically
```

---

## CT-7: Adjunctions for Optimization

### What to Build

An **adjunction** is a pair of functors (F, G) where F is the "best free construction"
and G is the "forgetful" functor. This appears everywhere in optimization:
- Free functor: "generate all possible layouts"
- Forgetful functor: "extract just the positions"
- The adjunction: the generated layout is the BEST one for those positions

### Application: Layout Optimization as Adjunction

```
F: Constraints → Layouts (free functor: generate optimal layout from constraints)
G: Layouts → Constraints (forgetful functor: extract constraints a layout satisfies)

The adjunction F ⊣ G means:
  For any constraints C and layout L,
  Hom(F(C), L) ≅ Hom(C, G(L))

  "A morphism from the optimal layout to L"
  is the same as
  "A morphism from the constraints to the constraints L satisfies"

  This is a universal property: F(C) is the BEST layout satisfying C.
```

### Implementation

```
packages/category/src/
  adjunction.ts           — Adjunction interface (unit, counit, triangle identities)
  layout-adjunction.ts    — Constraint ⊣ Layout adjunction for optimization
  matching-adjunction.ts  — Preference ⊣ Venue adjunction for recommendation
```

---

## Integration with Existing Codebase

### Refactoring Strategy

Do NOT rewrite the existing engine. Instead:

1. **Wrap** existing functions as morphisms in the category
2. **Add** functor interfaces to existing data transformations
3. **Layer** the category theory package alongside existing code
4. **Gradually** migrate subsystems to use categorical composition
5. **Test** everything with property-based verification of laws

### File Structure

```
packages/
  category/
    src/
      core.ts                    — Category, Morphism, compose, id
      functor.ts                 — Functor interface
      natural-transformation.ts  — Natural transformation interface
      monoidal.ts                — Monoidal category, tensor, unit
      adjunction.ts              — Adjunction interface
      operad.ts                  — Operad for multi-input composition
      kan-extension.ts           — Kan extensions for matching

      // Domain-specific implementations
      objects.ts                 — Venue domain objects as categorical objects
      morphisms.ts               — Domain operations as morphisms
      pipeline.ts                — Compositional pipeline builder
      event-source-functor.ts    — Event sourcing as a functor
      spatial-functor.ts         — 2D ↔ 3D as a functor
      serialization-functor.ts   — Domain ↔ Wire as a functor
      service-monoid.ts          — Service composition algebra
      event-operad.ts            — Event assembly operad
      venue-matching.ts          — Kan extension matching
      schema-category.ts         — Functorial data migration
      layout-adjunction.ts       — Layout optimization adjunction

      laws.ts                    — Category/functor/monoidal law checkers
    __tests__/
      laws.test.ts               — Property-based law verification
      functor-laws.test.ts       — Functor law tests for all functors
      composition.test.ts        — Pipeline composition tests
      matching.test.ts           — Kan extension matching tests
      service-composition.test.ts — Monoidal service composition tests
    README.md                    — Mathematical documentation with diagrams
```

### Documentation Requirements

The README.md for this package MUST include:
- Commuting diagrams (ASCII or Mermaid) for every functor and natural transformation
- Mathematical definitions alongside the TypeScript code
- Explanation of which category theory concepts map to which domain concepts
- References to Spivak's "Category Theory for the Sciences" and Milewski's
  "Category Theory for Programmers"
- Explanation of why this architecture provides correctness guarantees that
  traditional OOP/FP patterns don't

---

## Session Management

This is a large body of work. Implement in order:

1. **CT-1** (core category + objects + morphisms + pipeline) — 1 session
2. **CT-2** (functors: event-source, spatial, serialization) — 1 session
3. **CT-3** (natural transformations) — 1 session
4. **CT-4** (monoidal categories + operads for services) — 1 session
5. **CT-5** (Kan extensions for matching) — 1 session
6. **CT-6** (functorial data migration) — 1 session
7. **CT-7** (adjunctions for optimization) — 1 session

Each session: implement, write property-based tests verifying all laws, update
PROGRESS.md. Commit after each sub-section.
