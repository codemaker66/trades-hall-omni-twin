/**
 * CT-2: Serialization Functor
 *
 * Maps between in-memory domain objects and wire format (JSON):
 *   F: Domain → Wire
 *   G: Wire → Domain (inverse)
 *
 * Guarantees:
 *   deserialize(serialize(x)) ≡ x           (round-trip / retraction)
 *   serialize(f(x)) ≡ wireF(serialize(x))   (commuting diagram)
 */

import type { Morphism } from './core'

// ─── Wire Format Types ──────────────────────────────────────────────────────

/**
 * A JSON-safe wire format value.
 * This is the target category of our serialization functor.
 */
export type WireValue =
  | string | number | boolean | null
  | readonly WireValue[]
  | { readonly [key: string]: WireValue }

/**
 * A wire format envelope — what goes over the network.
 */
export interface WireEnvelope {
  readonly type: string
  readonly version: number
  readonly data: WireValue
  readonly timestamp: number
}

// ─── Serialization Functor Interface ────────────────────────────────────────

/**
 * A SerializationFunctor for a specific domain type.
 *
 * The pair (serialize, deserialize) must form a retraction:
 *   deserialize(serialize(x)) ≡ x  for all x
 *
 * This is slightly stronger than a functor — it's an adjunction
 * where the counit is an isomorphism.
 */
export interface SerializationCodec<Domain, Wire extends WireValue> {
  readonly typeName: string
  readonly version: number

  /** F: Domain → Wire (object mapping, forward direction). */
  serialize(domain: Domain): Wire

  /** G: Wire → Domain (object mapping, inverse direction). */
  deserialize(wire: Wire): Domain
}

/**
 * A morphism-aware serialization functor.
 * Also maps operations (morphisms) between categories.
 */
export interface SerializationFunctor<Domain, Wire extends WireValue> extends SerializationCodec<Domain, Wire> {
  /**
   * Morphism mapping: domain operation → wire operation.
   *
   * For any f: Domain → Domain,
   *   serialize(f(x)) ≡ mapMorphism(f)(serialize(x))
   */
  mapMorphism(
    domainOp: Morphism<Domain, Domain>,
  ): Morphism<Wire, Wire>
}

// ─── Codec Creation ─────────────────────────────────────────────────────────

/**
 * Create a simple SerializationCodec from serialize/deserialize pair.
 */
export function createCodec<Domain, Wire extends WireValue>(
  typeName: string,
  version: number,
  serialize: (d: Domain) => Wire,
  deserialize: (w: Wire) => Domain,
): SerializationCodec<Domain, Wire> {
  return { typeName, version, serialize, deserialize }
}

/**
 * Create a full SerializationFunctor that also maps morphisms.
 * The morphism mapping is derived from the codec: wrap domain ops with serialize/deserialize.
 */
export function createSerializationFunctor<Domain, Wire extends WireValue>(
  codec: SerializationCodec<Domain, Wire>,
): SerializationFunctor<Domain, Wire> {
  return {
    ...codec,
    mapMorphism(domainOp: Morphism<Domain, Domain>): Morphism<Wire, Wire> {
      return (wire: Wire) => {
        const domain = codec.deserialize(wire)
        const result = domainOp(domain)
        return codec.serialize(result)
      }
    },
  }
}

// ─── Compose Codecs ─────────────────────────────────────────────────────────

/**
 * Compose two codecs: Domain ↔ Intermediate ↔ Wire.
 * This is functor composition applied to serialization.
 */
export function composeCodecs<A, B extends WireValue, C extends WireValue>(
  inner: SerializationCodec<A, B>,
  outer: SerializationCodec<B, C>,
): SerializationCodec<A, C> {
  return createCodec(
    `${inner.typeName}→${outer.typeName}`,
    outer.version,
    (a: A) => outer.serialize(inner.serialize(a)),
    (c: C) => inner.deserialize(outer.deserialize(c)),
  )
}

// ─── Verification ───────────────────────────────────────────────────────────

/**
 * Verify the round-trip law: deserialize(serialize(x)) ≡ x
 */
export function verifyRoundTrip<Domain, Wire extends WireValue>(
  codec: SerializationCodec<Domain, Wire>,
  value: Domain,
  equals: (a: Domain, b: Domain) => boolean,
): boolean {
  const roundTripped = codec.deserialize(codec.serialize(value))
  return equals(roundTripped, value)
}

/**
 * Verify the commuting diagram: serialize(f(x)) ≡ mapMorphism(f)(serialize(x))
 */
export function verifyCommutingDiagram<Domain, Wire extends WireValue>(
  functor: SerializationFunctor<Domain, Wire>,
  op: Morphism<Domain, Domain>,
  value: Domain,
  wireEquals: (a: Wire, b: Wire) => boolean,
): boolean {
  // Path 1: apply domain op, then serialize
  const path1 = functor.serialize(op(value))

  // Path 2: serialize, then apply wire op
  const wireOp = functor.mapMorphism(op)
  const path2 = wireOp(functor.serialize(value))

  return wireEquals(path1, path2)
}

// ─── Envelope Codec ─────────────────────────────────────────────────────────

/**
 * Wrap a codec in an envelope for transport.
 * Returns a codec that produces a JSON-compatible wire object.
 */
export function withEnvelope<Domain, Wire extends WireValue>(
  codec: SerializationCodec<Domain, Wire>,
): {
  readonly typeName: string
  readonly version: number
  serialize(domain: Domain): WireEnvelope
  deserialize(wire: WireEnvelope): Domain
} {
  return {
    typeName: codec.typeName,
    version: codec.version,
    serialize(d: Domain): WireEnvelope {
      return {
        type: codec.typeName,
        version: codec.version,
        data: codec.serialize(d) as WireValue,
        timestamp: Date.now(),
      }
    },
    deserialize(env: WireEnvelope): Domain {
      return codec.deserialize(env.data as Wire)
    },
  }
}
