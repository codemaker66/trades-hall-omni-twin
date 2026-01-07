export default function Home() {
  return (
    <div className="min-h-screen px-6 pb-16 pt-10 text-foreground">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-16">
        <header className="flex flex-wrap items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="h-11 w-11 rounded-2xl bg-gradient-to-br from-[#f2b76f] via-[#ed8a5b] to-[#a95d4f]" />
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.35em] text-neutral-600">
                Project
              </p>
              <p className="font-serif text-2xl">Omni-Twin</p>
            </div>
          </div>
          <nav className="flex flex-wrap items-center gap-6 text-sm text-neutral-700">
            <a className="hover:text-neutral-900" href="#modules">
              Modules
            </a>
            <a className="hover:text-neutral-900" href="#workflow">
              Workflow
            </a>
            <a className="hover:text-neutral-900" href="#exports">
              Export
            </a>
          </nav>
          <button className="rounded-full bg-accent px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:translate-y-[-1px] hover:shadow-md">
            Request pilot
          </button>
        </header>

        <main className="flex flex-col gap-16">
          <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr] lg:items-center">
            <div className="space-y-6 anim-rise">
              <p className="text-sm font-semibold uppercase tracking-[0.3em] text-accent-2">
                Web-first spatial planning MVP
              </p>
              <h1 className="font-serif text-4xl leading-tight sm:text-5xl lg:text-6xl">
                Design trade hall layouts in minutes, not days.
              </h1>
              <p className="max-w-xl text-lg text-neutral-700">
                Omni-Twin pairs a fast 2D/3D viewer with inventory-aware placement so
                teams can test seating plans, iterate scenarios, and export
                shareable outputs from one browser tab.
              </p>
              <div className="flex flex-wrap items-center gap-4">
                <button className="rounded-full bg-foreground px-6 py-3 text-sm font-semibold text-background shadow-lg shadow-black/10 transition hover:translate-y-[-1px]">
                  Launch MVP
                </button>
                <button className="rounded-full border border-foreground/20 bg-surface px-6 py-3 text-sm font-semibold text-foreground transition hover:border-foreground/40">
                  View V1 scope
                </button>
              </div>
              <div className="flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-[0.22em] text-neutral-600">
                <span className="rounded-full bg-surface px-3 py-2">Viewer</span>
                <span className="rounded-full bg-surface px-3 py-2">Inventory</span>
                <span className="rounded-full bg-surface px-3 py-2">Scenarios</span>
                <span className="rounded-full bg-surface px-3 py-2">PDF Export</span>
              </div>
            </div>
            <div className="rounded-3xl border border-black/10 bg-white/80 p-6 shadow-xl shadow-black/5 backdrop-blur anim-rise anim-delay-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold uppercase tracking-[0.3em] text-neutral-500">
                  Venue Snapshot
                </p>
                <span className="rounded-full bg-surface-muted px-3 py-1 text-xs font-semibold text-neutral-700">
                  Draft 01
                </span>
              </div>
              <div className="mt-6 grid gap-4 sm:grid-cols-2">
                {[
                  { label: "Active layout", value: "Main Hall" },
                  { label: "Tables placed", value: "48" },
                  { label: "Chairs placed", value: "384" },
                  { label: "Scenario count", value: "6" },
                ].map((item) => (
                  <div key={item.label} className="rounded-2xl bg-surface p-4">
                    <p className="text-xs uppercase tracking-[0.2em] text-neutral-500">
                      {item.label}
                    </p>
                    <p className="mt-2 text-xl font-semibold">{item.value}</p>
                  </div>
                ))}
              </div>
              <div className="mt-6 rounded-2xl border border-dashed border-accent/40 bg-surface-muted px-4 py-5 text-sm text-neutral-700">
                Drop assets into the viewer, snap to grid, and align to room
                boundaries before saving a scenario.
              </div>
            </div>
          </section>

          <section id="modules" className="space-y-6">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <h2 className="font-serif text-3xl">V1 modules</h2>
              <p className="text-sm text-neutral-600">
                Focused on planning clarity, fast iteration, and exportable results.
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {[
                {
                  title: "Interactive viewer",
                  detail:
                    "2D/3D toggle with zoom, pan, and grid overlays for fast placement.",
                },
                {
                  title: "Place tables + chairs",
                  detail:
                    "Snap to grid, rotate, and track seat counts in real time.",
                },
                {
                  title: "Inventory-aware catalog",
                  detail:
                    "Only show items on hand and flag overages immediately.",
                },
                {
                  title: "Save + load layouts",
                  detail:
                    "Persist venue states, restore drafts, and share links.",
                },
                {
                  title: "Scenario builder",
                  detail:
                    "Branch variations, compare metrics, and lock approved plans.",
                },
                {
                  title: "Export center",
                  detail:
                    "Generate PDF layout packs and inventory lists for vendors.",
                },
              ].map((module, index) => (
                <div
                  key={module.title}
                  className={`rounded-2xl border border-black/10 bg-surface p-6 shadow-sm anim-rise anim-delay-${index % 2 === 0 ? "1" : "2"}`}
                >
                  <h3 className="text-lg font-semibold">{module.title}</h3>
                  <p className="mt-3 text-sm text-neutral-600">{module.detail}</p>
                </div>
              ))}
            </div>
          </section>

          <section id="workflow" className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="space-y-5">
              <h2 className="font-serif text-3xl">Workflow</h2>
              <p className="text-sm text-neutral-700">
                Move from empty venue to export in a single browser session.
              </p>
              <div className="space-y-4">
                {[
                  "Load a venue shell or import a basic floor plan.",
                  "Drag tables and chairs from inventory into the scene.",
                  "Save a baseline layout, then branch scenarios.",
                  "Export PDF sheets and a summarized inventory list.",
                ].map((step, index) => (
                  <div
                    key={step}
                    className="flex gap-4 rounded-2xl border border-black/10 bg-white/70 p-4 anim-rise"
                    style={{ animationDelay: `${120 + index * 120}ms` }}
                  >
                    <div className="mt-1 h-8 w-8 rounded-full bg-accent text-center text-sm font-semibold text-white">
                      {index + 1}
                    </div>
                    <p className="text-sm text-neutral-700">{step}</p>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-3xl border border-black/10 bg-gradient-to-br from-white via-white to-[#f0efe9] p-8 shadow-lg shadow-black/5 anim-fade">
              <h3 className="font-serif text-2xl">Scenario comparator</h3>
              <p className="mt-3 text-sm text-neutral-700">
                Track capacity, aisle widths, and inventory usage across scenarios
                without leaving the main workspace.
              </p>
              <div className="mt-6 grid gap-3">
                {[
                  { name: "Baseline 450", count: "450 seats", status: "Approved" },
                  { name: "Workshop 320", count: "320 seats", status: "Review" },
                  { name: "Expo 220", count: "220 seats", status: "Draft" },
                ].map((scenario) => (
                  <div
                    key={scenario.name}
                    className="flex items-center justify-between rounded-2xl border border-black/10 bg-white/80 px-4 py-3"
                  >
                    <div>
                      <p className="text-sm font-semibold">{scenario.name}</p>
                      <p className="text-xs text-neutral-600">{scenario.count}</p>
                    </div>
                    <span className="rounded-full bg-surface-muted px-3 py-1 text-xs font-semibold text-neutral-700">
                      {scenario.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section id="exports" className="grid gap-6">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <h2 className="font-serif text-3xl">Exports and inventory</h2>
              <p className="text-sm text-neutral-600">
                Deliver clean PDFs and inventory summaries for operations teams.
              </p>
            </div>
            <div className="grid gap-4 md:grid-cols-2">
              {[
                {
                  title: "PDF layout pack",
                  detail:
                    "Multi-page export with layout thumbnails, measurements, and notes.",
                },
                {
                  title: "Inventory list",
                  detail:
                    "Counts by item type, grouped for vendor pickup and delivery.",
                },
              ].map((exportItem, index) => (
                <div
                  key={exportItem.title}
                  className={`rounded-2xl border border-black/10 bg-surface p-6 anim-rise anim-delay-${index === 0 ? "1" : "2"}`}
                >
                  <h3 className="text-lg font-semibold">{exportItem.title}</h3>
                  <p className="mt-3 text-sm text-neutral-600">
                    {exportItem.detail}
                  </p>
                </div>
              ))}
            </div>
            <div className="rounded-2xl border border-accent/40 bg-white/80 p-5 text-sm text-neutral-700">
              V1 stays focused on layout planning. Compliance checks, splat
              captures, and VR walkthroughs are intentionally out of scope.
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}
