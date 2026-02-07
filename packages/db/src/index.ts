export { db, queryClient } from './client'
export * from './schema/index'
export {
  appendEvents,
  getEvents,
  getCurrentVersion,
  getSnapshot,
  saveSnapshot,
  toDomainEvent,
  ConcurrencyError,
  type StoredEvent,
  type StoredSnapshot,
} from './event-store'
