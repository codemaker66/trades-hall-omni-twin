import { Scene } from './components/Scene'
import { SceneErrorBoundary } from './components/SceneErrorBoundary'
import { motion } from 'framer-motion'

function App() {
  return (
    <div className="relative w-full h-screen bg-deep-slate selection:bg-electric-indigo selection:text-white">
      {/* 3D Scene Layer */}
      <div className="absolute inset-0 z-0">
        <SceneErrorBoundary>
          <Scene />
        </SceneErrorBoundary>
      </div>

      {/* UI Overlay Layer */}
      <div className="absolute inset-0 z-10 pointer-events-none p-6 flex flex-col justify-between">

        {/* Header / HUD */}
        <header className="flex justify-between items-start">
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-panel p-4 pointer-events-auto"
          >
            <h1 className="text-xl font-bold tracking-tight text-white mb-1">
              VENUE<span className="text-electric-indigo">TWIN</span>
            </h1>
            <p className="text-xs text-gray-400 font-mono">TRADES HALL • LEVEL 1</p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-panel p-2 flex gap-2 pointer-events-auto"
          >
            <button className="px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors text-sm font-medium">
              Save View
            </button>
            <button className="px-4 py-2 rounded-lg bg-electric-indigo hover:bg-indigo-500 transition-colors text-white text-sm font-medium shadow-[0_0_15px_rgba(99,102,241,0.5)]">
              Publish
            </button>
          </motion.div>
        </header>

        {/* Bottom UI / Inventory */}
        <footer className="flex justify-center items-end pb-4">
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="glass-panel p-2 flex gap-3 pointer-events-auto"
          >
            {['🪑', '🛋️', '🪴', '💡', '🎨'].map((icon, i) => (
              <button
                key={i}
                className="w-12 h-12 flex items-center justify-center text-2xl rounded-xl bg-white/5 hover:bg-white/10 hover:scale-110 active:scale-95 transition-all cursor-pointer border border-white/5"
              >
                {icon}
              </button>
            ))}
          </motion.div>
        </footer>

        {/* Floating Settings Panel */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
          className="absolute top-24 right-6 glass-panel p-5 w-64 pointer-events-auto"
        >
          <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Properties</h2>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs text-gray-300">Opacity</label>
              <div className="h-1 bg-gray-700 rounded-full overflow-hidden">
                <div className="h-full w-2/3 bg-gradient-to-r from-electric-indigo to-sunset-coral"></div>
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs text-gray-300">Material</label>
              <div className="grid grid-cols-4 gap-2">
                {[1, 2, 3, 4].map(n => (
                  <div key={n} className="aspect-square rounded-md bg-white/10 hover:ring-2 ring-electric-indigo cursor-pointer transition-all"></div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

      </div>
    </div>
  )
}

export default App
