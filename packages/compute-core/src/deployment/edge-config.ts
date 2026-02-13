// ---------------------------------------------------------------------------
// @omni-twin/compute-core — HPC-11: Edge Deployment Configuration
// ---------------------------------------------------------------------------
// Edge deployment configuration and validation for Cloudflare Workers,
// Deno Deploy, Vercel Edge Functions, and custom runtimes.
// ---------------------------------------------------------------------------

import type { EdgeConfig } from '../types.js';

// ---------------------------------------------------------------------------
// Provider defaults (bytes / milliseconds)
// ---------------------------------------------------------------------------

const MB = 1024 * 1024;

const PROVIDER_DEFAULTS: Record<
  EdgeConfig['provider'],
  Omit<EdgeConfig, 'provider'>
> = {
  cloudflare: {
    wasmSizeLimit: 10 * MB,
    executionTimeLimit: 30_000,
    memoryLimit: 128 * MB,
  },
  deno: {
    wasmSizeLimit: 20 * MB,
    executionTimeLimit: 50_000,
    memoryLimit: 512 * MB,
  },
  vercel: {
    wasmSizeLimit: 10 * MB,
    executionTimeLimit: 10_000,
    memoryLimit: 256 * MB,
  },
  custom: {
    wasmSizeLimit: 50 * MB,
    executionTimeLimit: 60_000,
    memoryLimit: 1024 * MB,
  },
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/**
 * Creates an EdgeConfig with sensible defaults for the given provider.
 *
 * Default limits per provider:
 * - **cloudflare**: 10 MB WASM, 30 s execution, 128 MB memory
 * - **deno**: 20 MB WASM, 50 s execution, 512 MB memory
 * - **vercel**: 10 MB WASM, 10 s execution, 256 MB memory
 * - **custom**: 50 MB WASM, 60 s execution, 1024 MB memory
 */
export function createEdgeConfig(provider: EdgeConfig['provider']): EdgeConfig {
  const defaults = PROVIDER_DEFAULTS[provider];
  return {
    provider,
    wasmSizeLimit: defaults.wasmSizeLimit,
    executionTimeLimit: defaults.executionTimeLimit,
    memoryLimit: defaults.memoryLimit,
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/**
 * Validates a deployment against edge provider constraints.
 * Returns a list of human-readable constraint violation strings
 * (empty array = all constraints met).
 */
export function validateEdgeDeployment(
  config: EdgeConfig,
  moduleSize: number,
  estimatedTimeMs: number,
  memoryMB: number,
): string[] {
  const violations: string[] = [];

  if (moduleSize > config.wasmSizeLimit) {
    violations.push(
      `WASM module size (${moduleSize} bytes) exceeds ${config.provider} limit ` +
      `(${config.wasmSizeLimit} bytes)`,
    );
  }

  if (estimatedTimeMs > config.executionTimeLimit) {
    violations.push(
      `Estimated execution time (${estimatedTimeMs} ms) exceeds ${config.provider} limit ` +
      `(${config.executionTimeLimit} ms)`,
    );
  }

  const memoryBytes = memoryMB * MB;
  if (memoryBytes > config.memoryLimit) {
    violations.push(
      `Memory requirement (${memoryMB} MB) exceeds ${config.provider} limit ` +
      `(${config.memoryLimit / MB} MB)`,
    );
  }

  return violations;
}

/**
 * Returns `true` if all edge deployment constraints are met for the
 * given module size, estimated execution time, and memory usage.
 */
export function fitsEdge(
  config: EdgeConfig,
  moduleSize: number,
  estimatedTimeMs: number,
  memoryMB: number,
): boolean {
  return validateEdgeDeployment(config, moduleSize, estimatedTimeMs, memoryMB).length === 0;
}

// ---------------------------------------------------------------------------
// Provider selection
// ---------------------------------------------------------------------------

/**
 * Selects the first edge provider whose constraints are satisfied by the
 * given module size, execution time, and memory usage.
 * Returns `null` if no provider fits.
 */
export function selectEdgeProvider(
  configs: EdgeConfig[],
  moduleSize: number,
  estimatedTimeMs: number,
  memoryMB: number,
): EdgeConfig | null {
  for (let i = 0; i < configs.length; i++) {
    const config = configs[i]!;
    if (fitsEdge(config, moduleSize, estimatedTimeMs, memoryMB)) {
      return config;
    }
  }
  return null;
}
