/**
 * CT-2: Event Source Functor
 *
 * Maps from the Event Sourcing category to the Read Model category:
 *   F(AggregateState) = ProjectedView
 *   F(event) = viewUpdate
 *
 * Functor law guarantees:
 *   F(event1 ∘ event2) = F(event1) ∘ F(event2)
 *   "Projecting event1 then event2 ≡ Projecting (event1 composed with event2)"
 *
 * This is THE correctness guarantee for read models.
 */

import type { Morphism } from './core'
import { compose } from './core'

// ─── Event Sourcing Category ────────────────────────────────────────────────

/** An event in the sourcing system — a morphism from State → State. */
export interface DomainEvent {
  readonly type: string
  readonly timestamp: number
  readonly payload: Record<string, unknown>
}

/** An aggregate state — an object in the event sourcing category. */
export interface AggregateState {
  readonly version: number
  readonly data: Record<string, unknown>
}

/** A projected view — an object in the read model category. */
export interface ProjectedView {
  readonly viewType: string
  readonly data: Record<string, unknown>
  readonly lastEventVersion: number
}

/** A view update — a morphism in the read model category. */
export interface ViewUpdate {
  readonly viewType: string
  readonly changes: Record<string, unknown>
  readonly fromVersion: number
  readonly toVersion: number
}

// ─── Event Source Functor ───────────────────────────────────────────────────

/**
 * The EventSourceFunctor maps:
 *   Objects: AggregateState → ProjectedView
 *   Morphisms: (event: State → State) → (update: View → View)
 *
 * The functor law F(e1 ∘ e2) = F(e1) ∘ F(e2) ensures that
 * projecting events individually vs as a batch gives the same result.
 */
export interface EventSourceFunctor<S extends AggregateState, V extends ProjectedView> {
  /** Object mapping: project aggregate state to view. */
  project(state: S): V

  /** Morphism mapping: convert event handler to view updater. */
  mapEvent(
    applyEvent: Morphism<S, S>,
  ): Morphism<V, V>
}

/**
 * Create an EventSourceFunctor from a projection and an event-to-update mapper.
 *
 * The projection function defines the object mapping.
 * The event mapper defines the morphism mapping.
 * Together, they must satisfy the functor laws.
 */
export function createEventSourceFunctor<
  S extends AggregateState,
  V extends ProjectedView,
>(
  project: Morphism<S, V>,
  mapEventToUpdate: (applyEvent: Morphism<S, S>) => Morphism<V, V>,
): EventSourceFunctor<S, V> {
  return {
    project,
    mapEvent: mapEventToUpdate,
  }
}

/**
 * Verify the functor composition law for a given pair of events:
 *   F(e2 ∘ e1) ≡ F(e2) ∘ F(e1)
 *
 * Returns true if the law holds for the given state.
 */
export function verifyEventFunctorComposition<
  S extends AggregateState,
  V extends ProjectedView,
>(
  functor: EventSourceFunctor<S, V>,
  applyEvent1: Morphism<S, S>,
  applyEvent2: Morphism<S, S>,
  state: S,
  viewEquals: (a: V, b: V) => boolean,
): boolean {
  // Path 1: compose events, then project
  const composedEvent = compose(applyEvent1, applyEvent2)
  const resultState = composedEvent(state)
  const path1 = functor.project(resultState)

  // Path 2: project, then apply view updates
  const view = functor.project(state)
  const update1 = functor.mapEvent(applyEvent1)
  const update2 = functor.mapEvent(applyEvent2)
  const composedUpdate = compose(update1, update2)
  const path2 = composedUpdate(view)

  return viewEquals(path1, path2)
}

// ─── Fold-based Event Projection ────────────────────────────────────────────

/**
 * Project a sequence of events to a view by folding.
 * This is the standard event-sourcing pattern expressed functorially.
 *
 * foldEvents(events, init, apply, project) ≡ project(events.reduce(apply, init))
 */
export function foldEvents<S extends AggregateState, V extends ProjectedView>(
  events: readonly Morphism<S, S>[],
  initialState: S,
  project: Morphism<S, V>,
): V {
  let state = initialState
  for (const applyEvent of events) {
    state = applyEvent(state)
  }
  return project(state)
}
