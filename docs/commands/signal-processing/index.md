# Signal Processing Command Track

Canonical command track: `docs/commands/signal-processing/`

Execution authority for agent commands is `docs/commands/**`.

## Commands

- `SIG-1` - Fourier Analysis for Demand Seasonality (depends_on: None)
- `SIG-2` - Wavelet Multi-Resolution Analysis (depends_on: `SIG-1`)
- `SIG-3` - Kalman Filtering for Real-Time Demand Estimation (depends_on: `SIG-2`)
- `SIG-4` - Digital Filter Preprocessing Pipeline (depends_on: `SIG-3`)
- `SIG-5` - Cross-Spectral Analysis and Venue Clustering (depends_on: `SIG-INT-1`)
- `SIG-6` - Anomaly Detection Ensemble (depends_on: `SIG-5`)
- `SIG-7` - Acoustic Simulation from 3D Venue Scans (depends_on: `SIG-6`)
- `SIG-8` - Occupancy Sensing and Crowd Flow (depends_on: `SIG-7`)
- `SIG-9` - Compressed Sensing for Sparse Venue Data (depends_on: `SIG-INT-2`)
- `SIG-10` - Advanced Time-Frequency Methods (depends_on: `SIG-9`)
- `SIG-11` - Streaming Signal Processing Architecture (depends_on: `SIG-10`)
- `SIG-12` - WASM and WebGPU Signal Processing (depends_on: `SIG-11`)
- `SIG-INT-1` - Signal Processing integration checkpoint 1 (depends_on: `SIG-1`, `SIG-2`, `SIG-3`, `SIG-4`)
- `SIG-INT-2` - Signal Processing integration checkpoint 2 (depends_on: `SIG-5`, `SIG-6`, `SIG-7`, `SIG-8`)
- `SIG-INT-3` - Signal Processing integration checkpoint 3 (depends_on: `SIG-9`, `SIG-10`, `SIG-11`, `SIG-12`)
