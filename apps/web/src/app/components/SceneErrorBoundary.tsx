'use client'

import React from 'react'
import { Button } from './ui/Button'

interface State {
    hasError: boolean
    error: Error | null
    retryCount: number
}

const MAX_RETRIES = 3

function classifyError(error: Error): { title: string; guidance: string } {
    const msg = error.message.toLowerCase()

    if (msg.includes('webgl') || msg.includes('rendering context')) {
        return {
            title: 'WebGL Not Available',
            guidance: 'Your browser or GPU does not support WebGL. Try updating your graphics drivers or using a different browser (Chrome, Edge, or Firefox).',
        }
    }

    if (msg.includes('shader') || msg.includes('compile') || msg.includes('link')) {
        return {
            title: 'Shader Compilation Error',
            guidance: 'A GPU shader failed to compile. This is usually caused by outdated graphics drivers. Try updating your GPU drivers.',
        }
    }

    if (msg.includes('memory') || msg.includes('allocation') || msg.includes('oom')) {
        return {
            title: 'Out of Memory',
            guidance: 'The 3D scene ran out of GPU memory. Try closing other tabs or applications, or reducing the number of items in the scene.',
        }
    }

    return {
        title: '3D Renderer Error',
        guidance: 'The 3D scene encountered an unexpected error. This may be due to browser compatibility or GPU limitations.',
    }
}

export class SceneErrorBoundary extends React.Component<{ children: React.ReactNode }, State> {
    state: State = { hasError: false, error: null, retryCount: 0 }

    static getDerivedStateFromError(error: Error): Partial<State> {
        return { hasError: true, error }
    }

    private handleRetry = () => {
        this.setState((prev) => ({
            hasError: false,
            error: null,
            retryCount: prev.retryCount + 1,
        }))
    }

    render() {
        if (this.state.hasError) {
            const classified = this.state.error
                ? classifyError(this.state.error)
                : { title: '3D Renderer Error', guidance: 'An unknown error occurred.' }

            const canRetry = this.state.retryCount < MAX_RETRIES

            return (
                <div className="w-full h-full flex items-center justify-center bg-surface-0">
                    <div className="text-center max-w-md p-8 border border-surface-25 rounded-sm bg-surface-10">
                        <h2 className="text-surface-90 text-lg font-semibold mb-2">
                            {classified.title}
                        </h2>
                        <p className="text-surface-70 text-sm mb-4">
                            {classified.guidance}
                        </p>
                        {this.state.error && (
                            <pre className="text-danger-50 text-xs bg-surface-0 p-3 rounded-sm mb-4 overflow-auto max-h-24">
                                {this.state.error.message}
                            </pre>
                        )}
                        <div className="flex gap-2 justify-center">
                            {canRetry && (
                                <Button onClick={this.handleRetry}>
                                    Try Again ({MAX_RETRIES - this.state.retryCount} left)
                                </Button>
                            )}
                            {!canRetry && (
                                <p className="text-surface-60 text-xs">Maximum retry attempts reached. Please reload the page.</p>
                            )}
                        </div>
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}
