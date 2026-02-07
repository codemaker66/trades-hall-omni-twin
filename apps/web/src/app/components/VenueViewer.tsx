'use client'

import { Scene } from './Scene'
import { Header } from './venue-viewer/Header'
import { LeftSidebar } from './venue-viewer/LeftSidebar'
import { BottomToolbar } from './venue-viewer/BottomToolbar'
import { ChairPromptModal } from './venue-viewer/ChairPromptModal'
import { KeyboardShortcutsHelp } from './venue-viewer/KeyboardShortcutsHelp'
import { ToastContainer } from './ui/Toast'
import { useKeyboardShortcuts } from './venue-viewer/hooks/useKeyboardShortcuts'
import { useSharedProjectLoader } from './venue-viewer/hooks/useSharedProjectLoader'
import { FeatureFlagPanel } from './venue-viewer/FeatureFlagPanel'

export default function VenueViewer() {
    useKeyboardShortcuts()
    const { notice, setNotice, error, setError } = useSharedProjectLoader()

    return (
        <div className="relative w-full h-[100dvh] bg-surface-0 selection:bg-indigo-50 selection:text-white overflow-hidden">
            <div className="absolute inset-0 z-0">
                <Scene />
            </div>

            <div className="absolute inset-0 z-10 pointer-events-none p-6 flex flex-col justify-between font-serif">
                <Header />
                <LeftSidebar
                    projectNotice={notice}
                    setProjectNotice={setNotice}
                    projectError={error}
                    setProjectError={setError}
                />
                <BottomToolbar />
            </div>

            <ChairPromptModal />
            <KeyboardShortcutsHelp />
            <ToastContainer />
            <FeatureFlagPanel />
        </div>
    )
}
