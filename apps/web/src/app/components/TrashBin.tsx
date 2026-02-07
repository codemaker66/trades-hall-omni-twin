import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useVenueStore } from '../../store'

export const TrashBin = () => {
    const draggedItemType = useVenueStore((state) => state.draggedItemType)
    const setDraggedItem = useVenueStore((state) => state.setDraggedItem)
    const [isHovered, setIsHovered] = useState(false)

    return (
        <AnimatePresence>
            {draggedItemType && (
                <div className="flex justify-center w-full overflow-visible pointer-events-auto">
                    <motion.div
                        initial={{ height: 0, opacity: 0, y: 20 }}
                        animate={{ height: 'auto', opacity: 1, y: 0 }}
                        exit={{ height: 0, opacity: 0, y: 20 }}
                        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                        className="w-full flex justify-center py-2"
                        onPointerUp={(e) => {
                            e.stopPropagation()
                            setDraggedItem(null)
                            setIsHovered(false)
                        }}
                        onPointerEnter={() => setIsHovered(true)}
                        onPointerLeave={() => setIsHovered(false)}
                    >
                        <div
                            className={`
                                relative w-[400px] h-16
                                flex items-center justify-center gap-3
                                transition-all duration-200 ease-out
                                cursor-pointer group
                                ${isHovered ? 'scale-105' : 'scale-100'}
                            `}
                            role="button"
                            aria-label="Drop here to remove item"
                        >
                            <div className={`
                                absolute inset-0 rounded-lg
                                bg-gradient-to-b from-danger-20 to-danger-25
                                border-2 border-danger-40
                                shadow-[0_0_20px_rgba(139,0,0,0.5)] transition-all duration-300
                                ${isHovered ? 'bg-danger-30 border-danger-60 shadow-[0_0_40px_rgba(255,50,50,0.6)]' : ''}
                            `} />

                            <div className="absolute top-[-2px] left-[-2px] w-3 h-3 border-t-2 border-l-2 border-gold-40 rounded-tl-sm pointer-events-none" />
                            <div className="absolute top-[-2px] right-[-2px] w-3 h-3 border-t-2 border-r-2 border-gold-40 rounded-tr-sm pointer-events-none" />
                            <div className="absolute bottom-[-2px] left-[-2px] w-3 h-3 border-b-2 border-l-2 border-gold-40 rounded-bl-sm pointer-events-none" />
                            <div className="absolute bottom-[-2px] right-[-2px] w-3 h-3 border-b-2 border-r-2 border-gold-40 rounded-br-sm pointer-events-none" />

                            <svg
                                xmlns="http://www.w3.org/2000/svg"
                                className={`w-5 h-5 text-danger-80 relative z-10 transition-transform ${isHovered ? 'rotate-12' : ''}`}
                                fill="none"
                                viewBox="0 0 24 24"
                                stroke="currentColor"
                                aria-hidden="true"
                            >
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                            </svg>

                            <span className="text-sm font-bold tracking-widest uppercase text-danger-80 relative z-10 font-serif">
                                Remove
                            </span>
                        </div>
                    </motion.div>
                </div>
            )}
        </AnimatePresence>
    )
}
