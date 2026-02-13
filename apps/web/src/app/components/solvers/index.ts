/**
 * Solver visualization components barrel export.
 *
 * Six components for real-time visualization of physics-inspired optimization:
 * - EnergyLandscape: 3D R3F surface plot with SA trajectory
 * - TemperatureViz: SVG energy traces + temperature ladder
 * - ParetoDashboard: SVG Pareto front scatter with objectives
 * - LayoutGallery: MCMC-sampled layout thumbnail grid
 * - ScheduleGantt: SVG room/time Gantt chart
 * - ConstraintPanel: Interactive weight sliders + energy breakdown
 */

export { EnergyLandscape } from './EnergyLandscape'
export type { EnergyLandscapeProps } from './EnergyLandscape'

export { TemperatureViz } from './TemperatureViz'
export type { TemperatureVizProps } from './TemperatureViz'

export { ParetoDashboard } from './ParetoDashboard'
export type { ParetoDashboardProps, ParetoSolution as ParetoPoint } from './ParetoDashboard'

export { LayoutGallery } from './LayoutGallery'
export type { LayoutGalleryProps, LayoutSample, LayoutItem } from './LayoutGallery'

export { ScheduleGantt } from './ScheduleGantt'
export type { ScheduleGanttProps, GanttEvent, GanttRoom } from './ScheduleGantt'

export { ConstraintPanel } from './ConstraintPanel'
export type { ConstraintPanelProps, ConstraintWeights, EnergyBreakdown } from './ConstraintPanel'
