/**
 * Monoidal service composition tests (CT-4) and data migration tests (CT-6).
 */

import { describe, test, expect } from 'vitest'
import fc from 'fast-check'
import {
  additiveMonoid, multiplicativeMonoid, stringMonoid, arrayMonoid,
  mconcat, monoidToMonoidalCategory, freeMonoid, foldFree,
} from '../monoidal'
import type { Monoid } from '../monoidal'
import {
  checkMonoidAssociativity, checkMonoidLeftIdentity, checkMonoidRightIdentity,
} from '../laws'
import {
  serviceMonoid, liftService, bundleServices, finalCost, detectConflicts,
} from '../service-monoid'
import type { Service } from '../objects'
import { cents, minutes } from '../objects'
import { createOp, createOperad, parallel } from '../operad'
import {
  assembleEvent, combineCatering, combineAV,
} from '../event-operad'
import type { VenueBooking, CateringContract, AVSetup, DecorationPlan, StaffSchedule } from '../event-operad'
import {
  schema, stringCol, numberCol, boolCol, uuidCol, validateSchema,
} from '../schema-category'
import {
  schemaFunctor, composeSchemaFunctors,
} from '../schema-functor'
import {
  MigrationChain, addColumnMigration, renameColumnMigration,
} from '../migration-builder'

// ─── Monoid Laws ────────────────────────────────────────────────────────────

describe('CT-4: Monoid Laws', () => {
  test('additive monoid associativity', () => {
    fc.assert(fc.property(
      fc.integer(), fc.integer(), fc.integer(),
      (a, b, c) => checkMonoidAssociativity(
        additiveMonoid.combine, a, b, c, Object.is,
      ),
    ))
  })

  test('additive monoid identity', () => {
    fc.assert(fc.property(
      fc.integer(),
      (a) =>
        checkMonoidLeftIdentity(additiveMonoid.combine, 0, a, Object.is) &&
        checkMonoidRightIdentity(additiveMonoid.combine, 0, a, Object.is),
    ))
  })

  test('multiplicative monoid associativity', () => {
    fc.assert(fc.property(
      fc.integer({ min: -100, max: 100 }),
      fc.integer({ min: -100, max: 100 }),
      fc.integer({ min: -100, max: 100 }),
      (a, b, c) => checkMonoidAssociativity(
        multiplicativeMonoid.combine, a, b, c, Object.is,
      ),
    ))
  })

  test('string monoid laws', () => {
    fc.assert(fc.property(
      fc.string(), fc.string(), fc.string(),
      (a, b, c) =>
        checkMonoidAssociativity(stringMonoid.combine, a, b, c, Object.is) &&
        checkMonoidLeftIdentity(stringMonoid.combine, '', a, Object.is) &&
        checkMonoidRightIdentity(stringMonoid.combine, '', a, Object.is),
    ))
  })

  test('array monoid laws', () => {
    const arrM = arrayMonoid<number>()
    fc.assert(fc.property(
      fc.array(fc.integer()), fc.array(fc.integer()), fc.array(fc.integer()),
      (a, b, c) => {
        const eq = (x: readonly number[], y: readonly number[]) =>
          JSON.stringify(x) === JSON.stringify(y)
        return checkMonoidAssociativity(arrM.combine, a, b, c, eq)
      },
    ))
  })

  test('mconcat folds monoid', () => {
    expect(mconcat(additiveMonoid, [1, 2, 3, 4, 5])).toBe(15)
    expect(mconcat(multiplicativeMonoid, [1, 2, 3, 4])).toBe(24)
    expect(mconcat(stringMonoid, ['a', 'b', 'c'])).toBe('abc')
  })

  test('monoidal category from monoid', () => {
    const mc = monoidToMonoidalCategory(additiveMonoid)
    expect(mc.tensor(3, 4)).toBe(7)
    expect(mc.unit).toBe(0)
  })

  test('free monoid and foldFree', () => {
    const fm = freeMonoid<string>()
    expect(fm.combine(['a'], ['b', 'c'])).toEqual(['a', 'b', 'c'])

    const sumLengths = foldFree(additiveMonoid, (s: string) => s.length)
    expect(sumLengths(['hello', 'hi', 'hey'])).toBe(10)
  })
})

// ─── Service Monoid ─────────────────────────────────────────────────────────

describe('CT-4: Service Monoid', () => {
  const catering: Service = {
    id: 's1', name: 'Catering', type: 'catering',
    baseCost: cents(100000), setupTime: minutes(120), teardownTime: minutes(60),
    requirements: [{ kind: 'space', description: 'kitchen area', quantity: 20 }],
  }

  const av: Service = {
    id: 's2', name: 'AV', type: 'av',
    baseCost: cents(50000), setupTime: minutes(90), teardownTime: minutes(45),
    requirements: [{ kind: 'power', description: 'main power', quantity: 30 }],
  }

  const photography: Service = {
    id: 's3', name: 'Photography', type: 'photography',
    baseCost: cents(30000), setupTime: minutes(30), teardownTime: minutes(15),
    requirements: [],
  }

  test('empty service is identity', () => {
    const lifted = liftService(catering)
    const withEmpty = serviceMonoid.combine(serviceMonoid.empty, lifted)
    expect(withEmpty.components).toHaveLength(1)
    expect(withEmpty.totalBaseCost).toBe(catering.baseCost)
  })

  test('bundle services combines costs', () => {
    const bundle = bundleServices([catering, av, photography])
    expect(bundle.components).toHaveLength(3)
    expect(bundle.totalBaseCost).toBe(cents(180000))
    expect(bundle.discountRate).toBe(0.10)  // 3+ services = 10%
  })

  test('finalCost applies discount', () => {
    const bundle = bundleServices([catering, av, photography])
    const cost = finalCost(bundle)
    expect(cost).toBe(cents(162000))  // 180000 * 0.9
  })

  test('service monoid associativity', () => {
    const a = liftService(catering)
    const b = liftService(av)
    const c = liftService(photography)

    const lhs = serviceMonoid.combine(serviceMonoid.combine(a, b), c)
    const rhs = serviceMonoid.combine(a, serviceMonoid.combine(b, c))

    expect(lhs.components.map(s => s.id)).toEqual(rhs.components.map(s => s.id))
    expect(lhs.totalBaseCost).toBe(rhs.totalBaseCost)
  })

  test('requirement merging sums quantities', () => {
    const s1: Service = { ...catering, requirements: [{ kind: 'power', description: 'outlets', quantity: 10 }] }
    const s2: Service = { ...av, requirements: [{ kind: 'power', description: 'outlets', quantity: 20 }] }
    const bundle = bundleServices([s1, s2])
    const powerReq = bundle.allRequirements.find(r => r.description === 'outlets')
    expect(powerReq?.quantity).toBe(30)
  })

  test('conflict detection finds power issues', () => {
    const highPower: Service = {
      ...av,
      requirements: [{ kind: 'power', description: 'main', quantity: 60 }],
    }
    const highPower2: Service = {
      ...photography, id: 's4', name: 'Lighting',
      requirements: [{ kind: 'power', description: 'main', quantity: 50 }],
    }
    const bundle = bundleServices([highPower, highPower2])
    const conflicts = detectConflicts(bundle)
    expect(conflicts.length).toBeGreaterThan(0)
    expect(conflicts[0]!.reason).toContain('power')
  })
})

// ─── Operad Tests ───────────────────────────────────────────────────────────

describe('CT-4: Operads', () => {
  test('identity operation', () => {
    const operad = createOperad<number>('Test')
    expect(operad.identity.apply(42)).toBe(42)
    expect(operad.identity.arity).toBe(1)
  })

  test('operad composition', () => {
    const add = createOp('add', (a: number, b: number) => a + b)
    const double = createOp('double', (a: number) => a * 2)

    const operad = createOperad<number>('Arithmetic')
    const composed = operad.compose(add, double, double)
    // double(3) + double(4) = 6 + 8 = 14
    expect(composed.apply(3, 4)).toBe(14)
  })

  test('event assembly validates inputs', () => {
    const venue: VenueBooking = {
      kind: 'venue-booking', venueId: 'v1', date: '2025-06-15',
      startTime: '09:00', endTime: '17:00', cost: cents(200000),
    }
    const catering: CateringContract = {
      kind: 'catering-contract', vendorId: 'cat1', style: 'plated',
      guestCount: 100, setupMinutes: minutes(60), cost: cents(100000), spaceRequired: 30,
    }
    const avSetup: AVSetup = {
      kind: 'av-setup', vendorId: 'av1', equipment: ['projector', 'mic'],
      setupMinutes: minutes(45), powerDraw: 30, cost: cents(50000), spaceRequired: 10,
    }
    const decor: DecorationPlan = {
      kind: 'decoration-plan', vendorId: 'dec1', elements: ['flowers', 'lighting'],
      setupMinutes: minutes(90), cost: cents(30000),
    }
    const staff: StaffSchedule = {
      kind: 'staff-schedule', staffCount: 10, roles: ['server', 'bartender'],
      costPerHour: cents(5000), totalHours: 8,
    }

    const result = assembleEvent.apply(venue, catering, avSetup, decor, staff)
    expect(result.venue).toBe(venue)
    expect(result.catering).toBe(catering)
    expect((result.totalCost as number)).toBeGreaterThan(0)
    expect((result.maxSetupTime as number)).toBe(90)
  })

  test('combine catering', () => {
    const main: CateringContract = {
      kind: 'catering-contract', vendorId: 'main', style: 'plated',
      guestCount: 100, setupMinutes: minutes(60), cost: cents(80000), spaceRequired: 25,
    }
    const dessert: CateringContract = {
      kind: 'catering-contract', vendorId: 'dessert', style: 'buffet',
      guestCount: 100, setupMinutes: minutes(30), cost: cents(20000), spaceRequired: 10,
    }
    const combined = combineCatering.apply(main, dessert)
    expect(combined.vendorId).toBe('main+dessert')
    expect(combined.cost).toBe(cents(100000))
  })

  test('combine AV validates power', () => {
    const audio: AVSetup = {
      kind: 'av-setup', vendorId: 'audio', equipment: ['speakers'],
      setupMinutes: minutes(30), powerDraw: 60, cost: cents(20000), spaceRequired: 5,
    }
    const video: AVSetup = {
      kind: 'av-setup', vendorId: 'video', equipment: ['projector'],
      setupMinutes: minutes(45), powerDraw: 50, cost: cents(30000), spaceRequired: 5,
    }

    const validation = combineAV.validate(audio, video)
    expect(validation.ok).toBe(false)
    if (!validation.ok) expect(validation.error[0]).toContain('power')
  })
})

// ─── Schema Category + Migration Tests ──────────────────────────────────────

describe('CT-6: Schema Category', () => {
  test('build schema with tables and foreign keys', () => {
    const s = schema('venue_db', 1)
      .table('venues', 'id', {
        id: uuidCol(),
        name: stringCol(),
        capacity: numberCol(),
      })
      .table('events', 'id', {
        id: uuidCol(),
        venue_id: uuidCol(),
        name: stringCol(),
      })
      .foreignKey('fk_event_venue', 'events', 'venue_id', 'venues', 'id')
      .build()

    expect(s.tables.size).toBe(2)
    expect(s.relations).toHaveLength(1)
    expect(s.relations[0]!.sourceTable).toBe('events')
  })

  test('validate schema catches missing tables in FK', () => {
    const s = schema('bad_db', 1)
      .table('venues', 'id', { id: uuidCol() })
      .build()

    // Manually add a bad FK
    const badSchema = {
      ...s,
      relations: [{ name: 'bad_fk', sourceTable: 'events', sourceColumn: 'venue_id', targetTable: 'venues', targetColumn: 'id', onDelete: 'restrict' as const }],
    }
    const errors = validateSchema(badSchema)
    expect(errors.length).toBeGreaterThan(0)
    expect(errors[0]).toContain('events')
  })
})

describe('CT-6: Schema Functor (Migration)', () => {
  const v1Schema = schema('venue_db', 1)
    .table('venues', 'id', {
      id: uuidCol(),
      name: stringCol(),
      capacity: numberCol(),
    })
    .build()

  test('pushforward migrates a row forward', () => {
    const v2Schema = schema('venue_db', 2)
      .table('venues', 'id', {
        id: uuidCol(),
        venue_name: stringCol(),
        capacity: numberCol(),
      })
      .build()

    const functor = schemaFunctor('v1→v2', v1Schema, v2Schema)
      .mapTable('venues', 'venues', {
        id: { kind: 'rename', targetColumn: 'id' },
        name: { kind: 'rename', targetColumn: 'venue_name' },
        capacity: { kind: 'rename', targetColumn: 'capacity' },
      })
      .build()

    const row = { id: '123', name: 'Main Hall', capacity: 200 }
    const migrated = functor.pushforward('venues', row)
    expect(migrated).toEqual({ id: '123', venue_name: 'Main Hall', capacity: 200 })
  })

  test('pullback migrates a row backward', () => {
    const v2Schema = schema('venue_db', 2)
      .table('venues', 'id', {
        id: uuidCol(),
        venue_name: stringCol(),
        capacity: numberCol(),
      })
      .build()

    const functor = schemaFunctor('v1→v2', v1Schema, v2Schema)
      .mapTable('venues', 'venues', {
        id: { kind: 'rename', targetColumn: 'id' },
        name: { kind: 'rename', targetColumn: 'venue_name' },
        capacity: { kind: 'rename', targetColumn: 'capacity' },
      })
      .build()

    const row = { id: '123', venue_name: 'Main Hall', capacity: 200 }
    const pulled = functor.pullback('venues', row)
    expect(pulled).toEqual({ id: '123', name: 'Main Hall', capacity: 200 })
  })

  test('transform column mapping', () => {
    const v2Schema = schema('venue_db', 2)
      .table('venues', 'id', {
        id: uuidCol(),
        name: stringCol(),
        capacity_str: stringCol(),
      })
      .build()

    const functor = schemaFunctor('v1→v2', v1Schema, v2Schema)
      .mapTable('venues', 'venues', {
        id: { kind: 'rename', targetColumn: 'id' },
        name: { kind: 'rename', targetColumn: 'name' },
        capacity: { kind: 'transform', targetColumn: 'capacity_str', transform: (v) => String(v) },
      })
      .build()

    const row = { id: '1', name: 'Hall', capacity: 100 }
    const migrated = functor.pushforward('venues', row)
    expect(migrated).toEqual({ id: '1', name: 'Hall', capacity_str: '100' })
  })

  test('functor composition: v1→v2→v3', () => {
    const v2Schema = schema('venue_db', 2)
      .table('venues', 'id', {
        id: uuidCol(),
        venue_name: stringCol(),
        capacity: numberCol(),
      })
      .build()

    const v3Schema = schema('venue_db', 3)
      .table('venues', 'id', {
        id: uuidCol(),
        venue_name: stringCol(),
        max_capacity: numberCol(),
      })
      .build()

    const f1 = schemaFunctor('v1→v2', v1Schema, v2Schema)
      .mapTable('venues', 'venues', {
        id: { kind: 'rename', targetColumn: 'id' },
        name: { kind: 'rename', targetColumn: 'venue_name' },
        capacity: { kind: 'rename', targetColumn: 'capacity' },
      })
      .build()

    const f2 = schemaFunctor('v2→v3', v2Schema, v3Schema)
      .mapTable('venues', 'venues', {
        id: { kind: 'rename', targetColumn: 'id' },
        venue_name: { kind: 'rename', targetColumn: 'venue_name' },
        capacity: { kind: 'rename', targetColumn: 'max_capacity' },
      })
      .build()

    const composed = composeSchemaFunctors(f1, f2)
    const row = { id: '1', name: 'Hall', capacity: 200 }
    const migrated = composed.pushforward('venues', row)
    expect(migrated).toEqual({ id: '1', venue_name: 'Hall', max_capacity: 200 })
  })
})

describe('CT-6: Migration Builder', () => {
  test('add column migration', () => {
    const v1 = schema('db', 1)
      .table('venues', 'id', {
        id: uuidCol(),
        name: stringCol(),
      })
      .build()

    const step = addColumnMigration(v1, 'venues', 'active', 'boolean', true)
    expect(step.name).toBe('add-active-to-venues')

    const migrated = step.functor.pushforward('venues', { id: '1', name: 'Hall' })
    expect(migrated).toEqual({ id: '1', name: 'Hall', active: true })
  })

  test('rename column migration', () => {
    const v1 = schema('db', 1)
      .table('venues', 'id', {
        id: uuidCol(),
        name: stringCol(),
      })
      .build()

    const step = renameColumnMigration(v1, 'venues', 'name', 'venue_name')
    const migrated = step.functor.pushforward('venues', { id: '1', name: 'Hall' })
    expect(migrated).toEqual({ id: '1', venue_name: 'Hall' })
  })

  test('migration chain', () => {
    const v1 = schema('db', 1)
      .table('venues', 'id', {
        id: uuidCol(),
        name: stringCol(),
      })
      .build()

    const step1 = addColumnMigration(v1, 'venues', 'active', 'boolean', true)
    const step2 = renameColumnMigration(step1.functor.target, 'venues', 'name', 'venue_name')

    const chain = new MigrationChain(v1)
      .addStep(step1)
      .addStep(step2)

    const migrated = chain.migrateRow('venues', { id: '1', name: 'Hall' })
    expect(migrated).toEqual({ id: '1', venue_name: 'Hall', active: true })

    expect(chain.getSteps()).toHaveLength(2)
    expect(chain.finalSchema().version).toBe(3)
  })
})
