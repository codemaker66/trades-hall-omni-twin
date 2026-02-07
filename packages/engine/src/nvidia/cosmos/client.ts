/**
 * NVIDIA Cosmos API client.
 *
 * Production: calls NIM endpoint for video generation.
 * Development: mock client returning sample data after a simulated delay.
 */

import type { CosmosRequest, CosmosResult, JobResult } from '../types'

// ─── Client Interface ────────────────────────────────────────────────────────

export interface CosmosClient {
  /** Submit a video generation request. Returns a job ID. */
  submit(request: CosmosRequest): Promise<string>
  /** Check job status. */
  status(jobId: string): Promise<JobResult<CosmosResult>>
  /** Cancel a pending job. */
  cancel(jobId: string): Promise<void>
}

// ─── Mock Client ─────────────────────────────────────────────────────────────

/** Simulated generation time in ms. */
const MOCK_GENERATION_TIME = 5000

export class MockCosmosClient implements CosmosClient {
  private jobs = new Map<string, { request: CosmosRequest; createdAt: number }>()
  private jobCounter = 0

  async submit(request: CosmosRequest): Promise<string> {
    const jobId = `cosmos-job-${++this.jobCounter}`
    this.jobs.set(jobId, { request, createdAt: Date.now() })
    return jobId
  }

  async status(jobId: string): Promise<JobResult<CosmosResult>> {
    const job = this.jobs.get(jobId)
    if (!job) {
      return {
        jobId,
        status: 'failed',
        progress: 0,
        error: 'Job not found',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      }
    }

    const elapsed = Date.now() - job.createdAt
    const progress = Math.min(1, elapsed / MOCK_GENERATION_TIME)

    if (progress < 1) {
      return {
        jobId,
        status: 'processing',
        progress,
        createdAt: new Date(job.createdAt).toISOString(),
        updatedAt: new Date().toISOString(),
      }
    }

    return {
      jobId,
      status: 'complete',
      progress: 1,
      result: {
        videoUrl: `https://mock-cosmos.nvidia.com/videos/${jobId}.mp4`,
        thumbnailUrl: `https://mock-cosmos.nvidia.com/thumbs/${jobId}.jpg`,
        durationMs: job.request.duration * 1000,
        resolution: job.request.resolution,
      },
      createdAt: new Date(job.createdAt).toISOString(),
      updatedAt: new Date().toISOString(),
    }
  }

  async cancel(jobId: string): Promise<void> {
    this.jobs.delete(jobId)
  }
}

// ─── Production Client Stub ──────────────────────────────────────────────────

/**
 * Production Cosmos client.
 * Calls the NIM endpoint. Requires COSMOS_API_URL and COSMOS_API_KEY env vars.
 */
export class ProductionCosmosClient implements CosmosClient {
  constructor(
    private readonly apiUrl: string,
    private readonly apiKey: string,
  ) {}

  async submit(request: CosmosRequest): Promise<string> {
    const response = await fetch(`${this.apiUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        prompt: request.scene.description,
        resolution: request.resolution,
        duration: request.duration,
        model: request.model,
      }),
    })

    if (!response.ok) {
      throw new Error(`Cosmos API error: ${response.status} ${response.statusText}`)
    }

    const data = await response.json() as { jobId: string }
    return data.jobId
  }

  async status(jobId: string): Promise<JobResult<CosmosResult>> {
    const response = await fetch(`${this.apiUrl}/status/${jobId}`, {
      headers: { 'Authorization': `Bearer ${this.apiKey}` },
    })

    if (!response.ok) {
      throw new Error(`Cosmos API error: ${response.status}`)
    }

    return response.json() as Promise<JobResult<CosmosResult>>
  }

  async cancel(jobId: string): Promise<void> {
    await fetch(`${this.apiUrl}/cancel/${jobId}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${this.apiKey}` },
    })
  }
}

/** Create a Cosmos client based on environment. */
export function createCosmosClient(
  apiUrl?: string,
  apiKey?: string,
): CosmosClient {
  if (apiUrl && apiKey) {
    return new ProductionCosmosClient(apiUrl, apiKey)
  }
  return new MockCosmosClient()
}
