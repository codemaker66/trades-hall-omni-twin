import { Component, type ReactNode } from "react";

type SceneErrorBoundaryProps = {
  children: ReactNode;
};

type SceneErrorBoundaryState = {
  hasError: boolean;
};

export class SceneErrorBoundary extends Component<
  SceneErrorBoundaryProps,
  SceneErrorBoundaryState
> {
  state: SceneErrorBoundaryState = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.error("Scene failed to render.", error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex h-full w-full items-center justify-center bg-deep-slate text-gray-200">
          <div className="glass-panel p-4 text-center text-sm">
            <p className="font-mono">3D scene failed to load.</p>
            <p className="text-xs text-gray-400">
              Check the console for WebGL or asset errors.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
