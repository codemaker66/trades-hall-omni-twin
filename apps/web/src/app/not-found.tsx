import Link from 'next/link'

export default function NotFound() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-950 px-4">
      <div className="w-full max-w-md space-y-6 text-center">
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-800">
          <span className="text-3xl font-bold text-slate-400">404</span>
        </div>
        <h1 className="text-2xl font-semibold text-slate-100">Page not found</h1>
        <p className="text-sm text-slate-400">
          The page you&apos;re looking for doesn&apos;t exist or has been moved.
        </p>
        <Link
          href="/"
          className="inline-block rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-slate-200 transition-colors hover:bg-slate-700"
        >
          Back to dashboard
        </Link>
      </div>
    </div>
  )
}
