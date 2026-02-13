import { FloorPlanEditor } from '../components/floor-plan-editor/FloorPlanEditor'

interface FloorPlanPageProps {
  searchParams: Promise<{ venueId?: string; planId?: string }>
}

export default async function FloorPlanPage({ searchParams }: FloorPlanPageProps) {
  const params = await searchParams
  return <FloorPlanEditor venueId={params.venueId} planId={params.planId} />
}
