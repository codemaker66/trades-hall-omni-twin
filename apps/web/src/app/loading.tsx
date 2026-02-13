export default function Loading() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950">
      <div className="flex flex-col items-center gap-4">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-700 border-t-slate-300" />
        <p className="text-sm text-slate-500">Loading...</p>
      </div>
    </div>
  )
}
