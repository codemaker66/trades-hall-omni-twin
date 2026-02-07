/**
 * @omni-twin/category — Category-Theoretic Architecture
 *
 * A compositional, provably correct architecture for the venue planning domain.
 *
 * CT-1: Core category primitives (Morphism, compose, id, Result, Kleisli, Pipeline)
 * CT-2: Functors (event-source, spatial, serialization)
 * CT-3: Natural transformations (persistence, renderer, transport strategy swaps)
 * CT-4: Monoidal categories + operads (service composition, event assembly)
 * CT-5: Kan extensions (venue-event matching)
 * CT-6: Functorial data migration (Spivak's framework)
 * CT-7: Adjunctions (layout optimization, matching)
 */

// ─── CT-1: Core ─────────────────────────────────────────────────────────────
export {
  type Morphism, compose, compose3, id,
  type Result, ok, err,
  type KleisliMorphism, composeK, idK, liftK,
  type Product, pair, fst, snd,
  type Coproduct, inl, inr, match,
  type Pipeline, type KleisliPipeline, pipeline, kleisliPipeline,
} from './core'

// ─── CT-1: Domain Objects ───────────────────────────────────────────────────
export {
  type ISODateTime, type Minutes, type Cents, isoDateTime, minutes, cents,
  type VenueSpec, type ExitSpec, type ObstacleSpec, type Amenity,
  type EventSpec, type EventType, type EventStyle,
  type Requirement, type CateringStyle, type AVNeed,
  type FloorPlan, type FloorPlanItem, type TableGrouping,
  type Constraint, type Violation, type ValidatedFloorPlan, markValidated,
  type Assignment, type Schedule, type ScheduleEntry,
  type Service, type ServiceType, type ServiceRequirement,
  type Proposal, type CostLineItem,
  type Configuration, type ConfigurationStatus,
  type CompatibilityScore, compatibilityScore,
  type RankedVenueMatch, type RankedVenueMatches,
} from './objects'

// ─── CT-1: Domain Morphisms ─────────────────────────────────────────────────
export {
  matchEventToVenue, matchVenue,
  constrainFloorPlan,
  assignEventToVenue,
  priceAssignment,
  scheduleAssignment,
  validateConfiguration,
} from './morphisms'

// ─── CT-1: Laws ─────────────────────────────────────────────────────────────
export {
  checkAssociativity, checkLeftIdentity, checkRightIdentity,
  checkKleisliAssociativity, checkKleisliLeftIdentity, checkKleisliRightIdentity,
  checkFunctorIdentity, checkFunctorComposition,
  checkNaturality,
  checkMonoidAssociativity, checkMonoidLeftIdentity, checkMonoidRightIdentity,
  checkTriangleF,
  structuralEquals,
} from './laws'

// ─── CT-2: Functors ─────────────────────────────────────────────────────────
export {
  type Functor, type FunctorValue,
  type ContravariantFunctor, type Bifunctor, type BifunctorValue,
  type Endofunctor,
  createFunctor, composeFunctors,
} from './functor'

export {
  type DomainEvent, type AggregateState, type ProjectedView, type ViewUpdate,
  type EventSourceFunctor,
  createEventSourceFunctor, verifyEventFunctorComposition, foldEvents,
} from './event-source-functor'

export {
  type Object2D, type FloorPlanScene, type Operation2D,
  type Object3D, type Scene3D, type Operation3D,
  SpatialFunctor, verifySpatialFunctorLaw,
} from './spatial-functor'

export {
  type WireValue, type WireEnvelope,
  type SerializationCodec, type SerializationFunctor,
  createCodec, createSerializationFunctor,
  composeCodecs, verifyRoundTrip, verifyCommutingDiagram,
  withEnvelope,
} from './serialization-functor'

// ─── CT-3: Natural Transformations ──────────────────────────────────────────
export {
  type NaturalTransformation,
  verticalCompose, identityNT,
  type StrategySwap, createStrategySwap, verifyStrategyRoundTrip,
  type NaturalIsomorphism, createNaturalIsomorphism,
} from './natural-transformation'

export {
  type InMemoryReadModel, type PostgresReadModel, type PostgresRow,
  createPersistenceSwap,
  updateInMemory, updatePostgres,
} from './persistence-nt'

export {
  type RenderCommand, type RenderFrame,
  type WebGLState, type WebGLCommand,
  type WebGPUState, type WebGPUCommand,
  compileToWebGL, compileToWebGPU, createRendererSwap,
} from './renderer-nt'

export {
  type TransportMessage, type DeliveryGuarantee,
  type WebSocketState, type WebSocketFrame,
  type WebRTCState, type WebRTCMessage,
  encodeWebSocket, decodeWebSocket,
  encodeWebRTC, decodeWebRTC,
  createTransportSwap,
  sendWebSocket, sendWebRTC,
} from './transport-nt'

// ─── CT-4: Monoidal Categories ──────────────────────────────────────────────
export {
  type Monoid, mconcat,
  additiveMonoid, multiplicativeMonoid, stringMonoid, arrayMonoid,
  maxMonoid, minMonoid,
  type MonoidalCategory, monoidToMonoidalCategory,
  type CommutativeMonoid, createCommutativeMonoid,
  freeMonoid, foldFree,
} from './monoidal'

export {
  type ComposedService,
  serviceMonoid, liftService, bundleServices, finalCost,
  type ServiceConflict, detectConflicts,
} from './service-monoid'

export {
  type OperadOp, type Operad,
  createOp, createOperad, parallel,
} from './operad'

export {
  type VenueBooking, type CateringContract, type AVSetup,
  type DecorationPlan, type StaffSchedule,
  type EventComponent, type AssembledEvent,
  assembleEvent, eventOperad,
  combineCatering, combineAV,
} from './event-operad'

// ─── CT-5: Kan Extensions ───────────────────────────────────────────────────
export {
  type KanExtension, type RankedResult, type ScoringFactor,
  type KanConfig, type KanDimension,
  buildKanExtension, coendFormula, endFormula,
} from './kan-extension'

export {
  type VenuePreferences,
  venueQuality,
  matchVenuesOptimistic, matchVenuesConservative, matchVenuesCoend,
} from './venue-matching'

// ─── CT-6: Functorial Data Migration ────────────────────────────────────────
export {
  type SchemaCategory, type TableSchema, type ColumnSchema, type ColumnType,
  type ForeignKey,
  SchemaCategoryBuilder, schema,
  col, stringCol, numberCol, boolCol, dateCol, uuidCol, jsonCol,
  validateSchema,
} from './schema-category'

export {
  type TableMapping, type ColumnMapping,
  type SchemaFunctor,
  SchemaFunctorBuilder, schemaFunctor,
  composeSchemaFunctors,
} from './schema-functor'

export {
  type MigrationStep,
  MigrationChain,
  addColumnMigration, renameColumnMigration,
} from './migration-builder'

// ─── CT-7: Adjunctions ──────────────────────────────────────────────────────
export {
  type Adjunction,
  createAdjunction,
  verifyLeftTriangle, verifyRightTriangle,
  transposeLeft, transposeRight,
  type MonadFromAdjunction, monadFromAdjunction,
} from './adjunction'

export {
  type LayoutConstraints, type OptimizedLayout, type LayoutItem,
  layoutAdjunction,
  optimizeLayout, satisfiedConstraints, constraintRoundTrip,
} from './layout-adjunction'

export {
  type MatchPreferences, type VenueSelection,
  matchingAdjunction,
  findOptimalVenue, venueCapabilities, preferencesRoundTrip,
  selectionSatisfies,
} from './matching-adjunction'
