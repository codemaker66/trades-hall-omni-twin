# TECHNIQUE_08_SIGNAL_PROCESSING.md — Signal Processing for Venue Intelligence

> ## CRITICAL: NEVER OMIT, SIMPLIFY, OR WEAKEN
>
> Claude Code must NEVER:
> - Omit features, techniques, or subsystems to fit within a single session
> - Simplify implementations to make them "easier" or "more practical"
> - Replace a specified technique with a simpler alternative without explicit approval
> - Skip sections because they seem redundant or low-priority
> - Truncate implementations to keep diffs small — instead, split across commits
>
> If a task is too large for one session: split it into sub-tasks across multiple
> sessions. Document what remains in PROGRESS.md and pick it up next session.
> The project must contain the FULL depth of every technique specified.
> "Good enough" is not good enough. Every feature exists at maximum fidelity.

> **Purpose**: Feed this to Claude Code after the main phases and earlier techniques
> are complete. Implements the full signal processing pipeline: Fourier analysis,
> wavelets, Kalman filtering, digital filters, spectral methods, anomaly detection,
> acoustic simulation, occupancy sensing, compressed sensing, advanced time-frequency,
> streaming architectures, and WASM/WebGPU DSP.
>
> **How to use**: Tell Claude Code: "Read TECHNIQUE_08_SIGNAL_PROCESSING.md
> and implement incrementally, starting from SP-1."
>
> **This is 12 sub-domains. Implement ALL of them. Do not skip any.**

---

## Key Papers (Referenced Throughout)

- Cooley & Tukey (1965). FFT algorithm. *Mathematics of Computation*
- Thomson (1982). Multitaper spectral estimation
- Allen & Berkley (1979). Image Source Method for room acoustics
- Helbing & Molnar (1995). Social Force Model. *Physical Review E*
- Candès & Recht (2009). Matrix completion via nuclear norm
- Ren et al. (2019). Spectral Residual anomaly detection. KDD 2019 (Microsoft)
- Killick et al. (2012). PELT changepoint detection. *JASA*
- Adams & MacKay (2007). Bayesian Online Changepoint Detection
- Arts & van den Broek (2022). fCWT — 34× faster CWT. *Nature Computational Science*
- Wu et al. (2023). TimesNet — FFT for period discovery. ICLR 2023
- Xu et al. (2024). FITS — frequency-domain interpolation. ICLR 2024
- Yi et al. (2023). FreTS — frequency-domain MLPs. NeurIPS 2023
- Yi et al. (2024). FilterNet — forecasters as frequency filters. NeurIPS 2024
- Qosja et al. (2025). Kalman-RBF demand forecasting
- Kalman (1960). Linear Filtering and Prediction

---

## Architecture Overview

```
packages/
  signal-core/                        — TypeScript types + client-side DSP
    src/
      types.ts                        — SpectralAnalysis, DemandSignal types
      fft/
        fft_client.ts                 — fft.js wrapper for browser FFT
        goertzel.ts                   — Single-bin DFT for targeted tracking
        sliding_dft.ts                — O(N) recursive bin update
      streaming/
        stream_processor.ts           — WebSocket → spectral pipeline
        ring_buffer.ts                — Lock-free SPSC for AudioWorklet
      visualization/
        Spectrogram.tsx               — D3/Canvas time-frequency display
        DemandSpectrum.tsx            — Interactive PSD with peak labels
        AnomalyTimeline.tsx           — Multi-method anomaly overlay
        AcousticPreview.tsx           — RT60/STI display per layout
        OccupancyHeatmap.tsx          — Real-time zone density map

  compute-wasm/                       — Rust → WASM signal processing
    src/
      fft.rs                         — rustfft 6.4.1 with wasm_simd
      wavelets.rs                    — DWT/MODWT via custom or wavelet port
      kalman.rs                      — adskalman 0.17.0 state estimation
      filters.rs                     — Biquad IIR cascade, SG filter
      compressed_sensing.rs          — OMP for sparse recovery

apps/
  ml-api/
    src/
      signal/
        fourier/
          spectral_analysis.py        — FFT, Welch PSD, STFT, multitaper
          cross_spectral.py           — Coherence, phase, cepstrum
          spectral_clustering.py      — Venue clustering by PSD profile
        wavelets/
          decomposition.py            — DWT, MODWT, CWT, wavelet packets
          denoising.py                — Soft/hard threshold, BayesShrink
          multiscale_forecast.py      — Decompose → forecast each → reconstruct
        kalman/
          demand_tracker.py           — KF/EKF/UKF for demand state estimation
          multi_sensor_fusion.py      — Fuse website/inquiry/booking signals
          adaptive_kalman.py          — Innovation-based Q/R estimation
          rts_smoother.py             — Backward pass for retrospective
        filters/
          preprocessing.py            — Butterworth, SG, median pipeline
          bandpass_features.py        — Extract weekly/monthly/annual bands
        anomaly/
          spectral_residual.py        — Microsoft SR method
          matrix_profile.py           — STUMPY motifs + discords
          cusum.py                    — Mean shift detection
          changepoint.py              — PELT (ruptures), BOCPD
          stl_decomposition.py        — STL remainder anomalies
          ensemble_detector.py        — Majority vote across methods
        acoustic/
          room_simulation.py          — pyroomacoustics ISM + ray tracing
          rt60_calculator.py          — Sabine/Eyring from room geometry
          sti_estimator.py            — Speech Transmission Index
          material_database.py        — Absorption coefficients library
          impulse_response.py         — RIR generation for Web Audio
        occupancy/
          wifi_csi.py                 — Channel State Information processing
          co2_estimation.py           — Mass balance occupancy model
          crowd_flow.py               — Social Force Model simulation
          particle_filter.py          — Bayesian occupancy estimation
        compressed_sensing/
          sparse_recovery.py          — Basis Pursuit (cvxpy), OMP, FISTA
          matrix_completion.py        — Nuclear norm for venue-feature matrices
        time_frequency/
          hht.py                      — EMD/CEEMDAN + Hilbert spectrum
          vmd.py                      — Variational Mode Decomposition
          synchrosqueezing.py         — SST via ssqueezepy
          stockwell.py                — S-transform
      routes/
        signal.py                     — FastAPI endpoints
```

### Python Dependencies

```
scipy>=1.17.0                  # FFT, signal, Welch, STFT, filters
numpy>=2.0                     # Array operations
PyWavelets>=1.9.0              # DWT, MODWT, CWT, wavelet packets
filterpy>=1.4.5                # Kalman filter (unmaintained but functional)
simdkalman>=1.0.4              # 100× faster batch Kalman (vectorized)
stumpy>=1.13.0                 # Matrix Profile (Numba, GPU, streaming)
ruptures>=1.1.10               # PELT changepoint detection
statsmodels>=0.14.6            # STL decomposition
prophet>=1.3.0                 # Fourier seasonality + anomaly detection
pyroomacoustics>=0.8.6         # Room acoustic simulation (ISM + ray tracing)
PySocialForce>=1.1.2           # Social Force Model for crowd flow
EMD-signal>=1.9.0              # EMD/EEMD/CEEMDAN
ssqueezepy>=0.6.6              # Synchrosqueezing (GPU support)
cvxpy>=1.8.1                   # Basis pursuit (compressed sensing)
scikit-learn>=1.8.0            # OMP, IterativeImputer
```

### Rust Dependencies (compute-wasm)

```toml
[dependencies]
rustfft = { version = "6.4", features = ["wasm_simd"] }
adskalman = "0.17.0"
biquad = "0.4"
wasm-bindgen = "0.2"
```

---

## SP-1: Fourier Analysis for Demand Seasonality

### What to Build

Extract periodic components (weekly=7, monthly=30, annual=365) from booking
time series. The foundation of the entire signal processing pipeline.

### Python Implementation

```python
# apps/ml-api/src/signal/fourier/spectral_analysis.py

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import welch, stft, detrend, find_peaks, windows

class VenueSpectralAnalyzer:
    """
    DFT: X[k] = Σ_{n=0}^{N-1} x[n]·e^{-j2πkn/N}
    Frequency resolution: Δf = f_s/N
    For daily-sampled, 365 days: Δf ≈ 0.00274 cycles/day

    Windowing reduces spectral leakage:
    - Hann: -31.5 dB sidelobes, 4·Δf main lobe (good default)
    - Hamming: -42.7 dB sidelobes
    - Blackman: -58 dB sidelobes, 6·Δf main lobe
      (preferred when strong weekly signals risk leaking into monthly)

    Spectral estimation:
    - Periodogram: high-variance, variance does NOT decrease with N
    - Welch: K overlapping segments averaged, variance drops ~K
    - Multitaper (Thomson 1982): NW=4, K=7 Slepian tapers,
      superior bias-variance tradeoff
    """

    def __init__(self, fs: float = 1.0):
        self.fs = fs  # 1.0 = daily sampling

    def extract_seasonality(self, bookings: np.ndarray, window: str = "blackman"):
        N = len(bookings)
        bookings_dt = detrend(bookings, type="linear")
        win = getattr(windows, window)(N, sym=False)

        # FFT with zero-padding for frequency interpolation
        N_pad = 2 ** (int(np.ceil(np.log2(N))) + 1)
        X = fft(bookings_dt * win, n=N_pad)
        freqs = fftfreq(N_pad, d=1.0 / self.fs)
        pos = freqs > 0
        magnitude = 2.0 / N * np.abs(X[pos])
        periods = 1.0 / freqs[pos]

        # Identify dominant peaks
        peaks, props = find_peaks(
            magnitude, height=np.max(magnitude) * 0.05, distance=5)
        dominant = sorted(peaks, key=lambda i: magnitude[i], reverse=True)[:10]

        return {
            "freqs": freqs[pos],
            "magnitude": magnitude,
            "periods": periods,
            "dominant_periods": [(periods[i], magnitude[i]) for i in dominant],
        }

    def welch_psd(self, bookings: np.ndarray, nperseg: int = 256):
        """Welch's method: 50% overlap, K=8-16 segments, variance drops ~K."""
        bookings_dt = detrend(bookings, type="linear")
        f, Pxx = welch(bookings_dt, fs=self.fs, nperseg=nperseg, window="hann")
        return f, Pxx

    def time_varying_spectrum(self, bookings: np.ndarray,
                              nperseg: int = 90, noverlap: int = 83):
        """
        STFT for tracking seasonal evolution.
        90-day window (quarterly), 7-day hop.
        Time-frequency uncertainty: Δt·Δf ≥ 1/4π
        """
        bookings_dt = detrend(bookings, type="linear")
        f, t, Zxx = stft(bookings_dt, fs=self.fs,
                         nperseg=nperseg, noverlap=noverlap, window="hann")
        return f, t, np.abs(Zxx)

    def reconstruct_seasonal(self, bookings: np.ndarray,
                              target_periods: list = [365.25, 30.44, 7.0]):
        """
        Frequency-domain filtering: keep only target periodic components.
        Reconstruct via IFFT for clean seasonal signal.
        """
        N = len(bookings)
        bookings_dt = detrend(bookings, type="linear")
        N_pad = 2 ** (int(np.ceil(np.log2(N))) + 1)
        X = fft(bookings_dt, n=N_pad)
        freqs = fftfreq(N_pad, d=1.0 / self.fs)

        X_filt = np.zeros_like(X)
        for target_p in target_periods:
            idx = np.argmin(np.abs(freqs - 1.0 / target_p))
            bw = 3  # bandwidth in bins
            X_filt[idx - bw:idx + bw + 1] = X[idx - bw:idx + bw + 1]
            X_filt[N_pad - idx - bw:N_pad - idx + bw + 1] = \
                X[N_pad - idx - bw:N_pad - idx + bw + 1]

        return np.real(ifft(X_filt))[:N]

    def fft_convolve(self, signal: np.ndarray, kernel: np.ndarray):
        """FFT-based convolution: O(N log N) vs O(N²) direct."""
        n = len(signal) + len(kernel) - 1
        N = 2 ** int(np.ceil(np.log2(n)))
        return np.real(ifft(fft(signal, N) * fft(kernel, N)))[:n]
```

### Performance Benchmarks

```
| Size       | scipy (μs) | FFTW (μs) | rustfft (μs) | fft.js (μs) |
|------------|-----------|-----------|-------------|-------------|
| 2^10 (1K)  | ~15       | ~5        | ~8          | ~28         |
| 2^14 (16K) | ~180      | ~60       | ~100        | ~350        |
| 2^18 (256K)| ~4,000    | ~1,400    | ~2,200      | ~12,000     |
| 2^20 (1M)  | ~18,000   | ~7,000    | ~11,000     | ~60,000     |

For venue data (N ≤ few thousand): all complete in <1ms.
FFTW is 2-3× faster than scipy due to codelet optimization.
```

---

## SP-2: Wavelet Multi-Resolution Analysis

### What to Build

Localized time-frequency analysis — detect WHEN seasonal patterns
strengthen, weaken, or shift. Multi-scale forecasting pipeline.

```python
# apps/ml-api/src/signal/wavelets/decomposition.py

import pywt
import numpy as np

class VenueWaveletAnalyzer:
    """
    DWT via Mallat's pyramid: O(N) per level.
    cA_j[n] = Σ_k h[k-2n]·cA_{j-1}[k]  (low-pass + downsample)
    cD_j[n] = Σ_k g[k-2n]·cA_{j-1}[k]  (high-pass + downsample)

    Wavelet families for venue data:
    - db4 (Daubechies, 4 vanishing moments): smooth booking signals
    - Haar: step changes in pricing tiers
    - Morlet (complex): oscillatory demand, provides amplitude + phase
    - Mexican Hat: peak detection, booking spikes

    MODWT preferred over DWT for time series:
    - Shift-invariant (shifting input shifts coefficients by same amount)
    - Works for arbitrary sample sizes (DWT needs powers of 2)
    - Better temporal alignment
    - Cost: N coefficients per level (redundant) vs N/2^j for DWT
    """

    def modwt_decompose(self, signal: np.ndarray, wavelet: str = "db4",
                         level: int = None):
        """MODWT via PyWavelets MRA (multiresolution analysis)."""
        if level is None:
            level = pywt.dwt_max_level(len(signal), wavelet)
        # mra returns [D1, D2, ..., DJ, AJ] — detail + approx
        components = pywt.mra(signal, wavelet, level=level, transform="swt")
        return {
            "details": components[:-1],     # High-freq fluctuations per level
            "approximation": components[-1], # Low-freq trend
            "wavelet": wavelet,
            "level": level,
        }

    def denoise(self, signal: np.ndarray, wavelet: str = "db4",
                method: str = "soft", level: int = None):
        """
        Wavelet denoising via thresholding on detail coefficients.

        Hard: η_H(w,λ) = w if |w|>λ, else 0
        Soft: η_S(w,λ) = sign(w)·max(|w|-λ, 0)

        Universal threshold: λ = σ̂·√(2 ln n)
        σ̂ = MAD(cD₁)/0.6745  (noise from finest detail level)
        BayesShrink: adapts per level λ_j = σ̂²/σ̂_{x,j}
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        # Estimate noise from finest detail
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(signal)))

        denoised_coeffs = [coeffs[0]]  # Keep approximation
        for c in coeffs[1:]:
            if method == "soft":
                denoised_coeffs.append(
                    pywt.threshold(c, value=threshold, mode="soft"))
            else:
                denoised_coeffs.append(
                    pywt.threshold(c, value=threshold, mode="hard"))

        return pywt.waverec(denoised_coeffs, wavelet)
```

```python
# apps/ml-api/src/signal/wavelets/multiscale_forecast.py

"""
Multi-scale forecasting pipeline (dominant 2024-2025 approach):
1. MODWT decompose signal into levels
2. Forecast each component independently (ARIMA, LSTM, Transformer)
3. Reconstruct: x̂_{t+h} = Σ Ŵ_{j,t+h} + V̂_{J,t+h}

~14% accuracy improvement over monolithic approaches.

Cutting edge: WaveToken (ICML 2025) — wavelet-transformer hybrids.
VMD-Transformer also dominant in 2024-2025 benchmarks.
"""
```

### Libraries

```
PyWavelets 1.9.0: pip install PyWavelets==1.9.0
  - DWT, SWT, CWT, MODWT (via mra), wavelet packets
  - 1M-point DWT in ~8ms

discrete-wavelets 5.0.15: npm install discrete-wavelets (TypeScript)
dwt 0.5.2: cargo add dwt (Rust)
wasmlets (Flatiron Institute): WASM DWT/SWT/MODWT/CWT via wavelib
WASM wavelet transforms: ~70% of native speed
fCWT (Arts & van den Broek, Nature Comp Sci 2022): 34× faster CWT
```

---

## SP-3: Kalman Filtering for Real-Time Demand Estimation

### What to Build

Fuse multiple noisy signals (website visits, inquiries, bookings) into
a single authoritative demand estimate with uncertainty quantification.

```python
# apps/ml-api/src/signal/kalman/demand_tracker.py

import numpy as np
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

class VenueDemandTracker:
    """
    State vector: x = [demand_level, demand_velocity, seasonal_component]ᵀ

    Predict: x̂(k|k-1) = F·x̂(k-1|k-1) + B·u(k)
             P(k|k-1) = F·P(k-1|k-1)·Fᵀ + Q
    Update:  K(k) = P(k|k-1)·Hᵀ·(H·P(k|k-1)·Hᵀ + R)⁻¹
             x̂(k|k) = x̂(k|k-1) + K(k)·(z(k) - H·x̂(k|k-1))
             P(k|k) = (I - K·H)·P(k|k-1)

    Observation model H maps demand to multiple noisy signals:
      H = [[1, 0, 1],       — website visits (demand + seasonal)
           [0.5, 0, 0.3],   — inquiries (weaker signal)
           [0.2, 0, 0.1]]   — actual bookings (weakest but cleanest)

    R = diag([25.0, 4.0, 1.0]) — website noisiest, bookings most reliable
    Q = diag([1.0, 0.1, 0.5])  — process noise

    Performance: single step ~50-100μs Python, ~0.5-2μs Rust.
    simdkalman: 10,000 venues at hourly resolution in <1 second.
    """

    def __init__(self, dt: float = 1.0):
        self.kf = KalmanFilter(dim_x=3, dim_z=3)

        # State transition: constant velocity + seasonal
        self.kf.F = np.array([
            [1, dt, 0],
            [0, 1,  0],
            [0, 0,  1]])

        # Observation model
        self.kf.H = np.array([
            [1.0, 0, 1.0],    # website visits
            [0.5, 0, 0.3],    # inquiries
            [0.2, 0, 0.1]])   # bookings

        # Process noise
        self.kf.Q = np.diag([1.0, 0.1, 0.5])
        # Measurement noise
        self.kf.R = np.diag([25.0, 4.0, 1.0])
        # Initial state
        self.kf.x = np.array([100, 0, 10])
        self.kf.P = np.eye(3) * 100

    def update(self, website_visits, inquiries, bookings):
        self.kf.predict()
        z = np.array([website_visits, inquiries, bookings])
        self.kf.update(z)
        return {
            "demand_level": self.kf.x[0],
            "demand_velocity": self.kf.x[1],
            "seasonal": self.kf.x[2],
            "uncertainty": np.sqrt(self.kf.P[0, 0]),
        }
```

```python
# apps/ml-api/src/signal/kalman/adaptive_kalman.py

"""
Adaptive Kalman with innovation-based Q/R estimation:
  Ĉ(k) = α·Ĉ(k-1) + (1-α)·ỹ(k)·ỹ(k)ᵀ
  R̂ = Ĉ - H·P·Hᵀ

Exponential forgetting factor α ∈ [0.95, 0.99] — higher = more stable,
lower = faster adaptation to changing noise characteristics.

RTS Smoother for retrospective analysis:
  C(k) = P(k|k)·Fᵀ·P(k+1|k)⁻¹
  x̂ˢ(k) = x̂(k|k) + C(k)·(x̂ˢ(k+1) - F·x̂(k|k))
"""
```

```python
# Batch processing with simdkalman (100× faster than pykalman)

"""
import simdkalman

kf = simdkalman.KalmanFilter(
    state_transition=F,
    process_noise=Q,
    observation_model=H,
    observation_noise=R)

# Process 1000 venues simultaneously: ~5μs per venue
results = kf.compute(data_for_all_venues, observations=n_timesteps)
smoothed = results.smoothed  # RTS smoother included
"""
```

### UKF for Nonlinear Demand Models

```python
"""
UKF avoids Jacobians via 2n+1 sigma points:
  χ₀ = x̂
  χᵢ = x̂ ± √((n+λ)P)ᵢ

Merwe-Wan scaling: α ~1e-3 to 1 (spread), β=2 (Gaussian optimal), κ=0.

Use when demand model is nonlinear (e.g., capacity cliff effects,
threshold pricing, seasonal × event interactions).
"""
```

---

## SP-4: Digital Filter Preprocessing Pipeline

### What to Build

Clean raw booking data for ML consumption. Preserve peaks (booking spikes)
while removing noise.

```python
# apps/ml-api/src/signal/filters/preprocessing.py

from scipy.signal import savgol_filter, medfilt, butter, sosfiltfilt
import numpy as np

class BookingPreprocessor:
    """
    Recommended pipeline (in order):
    1. Median filter (kernel=5): outlier removal, 50% breakdown point
    2. Savitzky-Golay (window=15, poly=3): denoise preserving peaks
    3. Butterworth bandpass (sosfiltfilt): extract weekly/monthly/annual
    4. Low-pass Butterworth (cutoff=1/30): trend extraction
    5. SG with deriv=1,2: velocity/acceleration features

    Always use sosfiltfilt (zero-phase, second-order sections)
    over filtfilt with ba form — far better numerical stability.

    FIR vs IIR:
    - FIR: always stable, linear phase, higher order needed
    - IIR: 5-10× fewer ops for equivalent selectivity, nonlinear phase
    - 4th-order Chebyshev IIR ≈ 40-60 tap FIR

    Filter types:
    - Butterworth: maximally flat passband (default choice)
    - Chebyshev Type I: sharper transition, equiripple passband
    - Elliptic: sharpest transition for given order, equiripple both bands
    - Bessel: best shape preservation (maximally flat group delay)
    """

    def __init__(self, fs: float = 1.0):
        self.fs = fs

    def full_pipeline(self, bookings: np.ndarray) -> dict:
        # Step 1: Robust outlier removal
        step1 = medfilt(bookings, kernel_size=5)

        # Step 2: Smooth preserving peaks
        step2 = savgol_filter(step1, window_length=15, polyorder=3)

        # Step 3: Bandpass features
        weekly = self._bandpass(step2, low=1/8, high=1/6)      # ~7 day
        monthly = self._bandpass(step2, low=1/35, high=1/25)    # ~30 day
        annual = self._bandpass(step2, low=1/400, high=1/300)   # ~365 day

        # Step 4: Trend (low-pass)
        trend = self._lowpass(step2, cutoff=1/30)

        # Step 5: Derivative features
        velocity = savgol_filter(step2, window_length=15, polyorder=3, deriv=1)
        acceleration = savgol_filter(step2, window_length=15, polyorder=3, deriv=2)

        return {
            "cleaned": step2,
            "trend": trend,
            "weekly": weekly,
            "monthly": monthly,
            "annual": annual,
            "velocity": velocity,
            "acceleration": acceleration,
        }

    def _bandpass(self, signal, low, high, order=4):
        sos = butter(order, [low, high], btype="band",
                     fs=self.fs, output="sos")
        return sosfiltfilt(sos, signal)

    def _lowpass(self, signal, cutoff, order=4):
        sos = butter(order, cutoff, btype="low",
                     fs=self.fs, output="sos")
        return sosfiltfilt(sos, signal)
```

---

## SP-5: Cross-Spectral Analysis and Venue Clustering

```python
# apps/ml-api/src/signal/fourier/cross_spectral.py

"""
Magnitude-squared coherence:
  γ²(f) = |P_xy(f)|² / (P_xx(f)·P_yy(f))
  Range [0,1]. Near 1 = linear relationship at frequency f.

High coherence at f=1/7 between bookings and competitor pricing →
weekly pricing co-movement detected.

Phase spectrum reveals lead/lag:
  τ(f) = -phase(f)/(2πf) gives time delay in days.

Cepstral analysis for repeat patterns:
  c[n] = IFFT(log|FFT(x)|²)
  Peaks at quefrency q → periodicity of period q.
  Detects biweekly corporate events, monthly recurring bookings.
"""
```

```python
# apps/ml-api/src/signal/fourier/spectral_clustering.py

"""
Spectral clustering of venues by demand frequency profile:
1. Compute normalized PSD for each venue
2. Build similarity: W_ij = exp(-||PSD_i - PSD_j||²/2σ²)
3. Graph Laplacian eigenvectors
4. k-means on eigenvectors

Clusters: "weekend-dominant", "holiday-driven", "corporate-weekday", etc.
Venues in same cluster share optimal pricing strategies →
transfer learning across spectrally-similar venues.
"""
```

---

## SP-6: Anomaly Detection Ensemble

### What to Build

Three complementary methods in parallel with majority vote.

```python
# apps/ml-api/src/signal/anomaly/ensemble_detector.py

import stumpy
import ruptures
import numpy as np
from scipy.fft import fft, ifft

class VenueAnomalyEnsemble:
    """
    Runs STL + Matrix Profile + CUSUM in parallel.
    Flag points where ≥2 methods agree (majority vote).

    Catches:
    - Point anomalies (STL remainder exceedance)
    - Subsequence anomalies (Matrix Profile discords)
    - Mean shifts (CUSUM threshold crossing)
    """

    def detect(self, bookings: np.ndarray) -> dict:
        stl_flags = self._stl_anomalies(bookings)
        mp_flags = self._matrix_profile_discords(bookings)
        cusum_flags = self._cusum_shifts(bookings)

        # Majority vote
        combined = (stl_flags.astype(int) + mp_flags.astype(int)
                    + cusum_flags.astype(int))
        consensus = combined >= 2

        return {
            "anomalies": consensus,
            "stl": stl_flags,
            "matrix_profile": mp_flags,
            "cusum": cusum_flags,
        }

    def _matrix_profile_discords(self, signal, m=14):
        """
        STUMPY 1.13.0: P_i = min_j d(T_{i:i+m}, T_{j:j+m})
        Discords = max MP values (most unusual subsequences)
        Motifs = min MP values (most recurring patterns)
        GPU: stumpy.gpu_stump, Streaming: stumpy.stumpi (O(1) per update)
        """
        mp = stumpy.stump(signal, m=m)
        threshold = np.percentile(mp[:, 0], 99)
        flags = np.zeros(len(signal), dtype=bool)
        discord_indices = np.where(mp[:, 0] > threshold)[0]
        for idx in discord_indices:
            flags[idx:idx + m] = True
        return flags

    def _cusum_shifts(self, signal, k=0.5, h=5.0):
        """
        CUSUM: S⁺_t = max(0, S⁺_{t-1} + (x_t - μ₀ - k))
        Signal when S⁺_t > h.
        With k=0.5σ, h=5σ: ARL₀ ≈ 465 (false alarm every ~465 samples).
        """
        mu = np.mean(signal)
        sigma = np.std(signal)
        s_pos = np.zeros(len(signal))
        s_neg = np.zeros(len(signal))
        flags = np.zeros(len(signal), dtype=bool)
        for t in range(1, len(signal)):
            s_pos[t] = max(0, s_pos[t-1] + (signal[t] - mu - k * sigma))
            s_neg[t] = max(0, s_neg[t-1] + (mu - k * sigma - signal[t]))
            if s_pos[t] > h * sigma or s_neg[t] > h * sigma:
                flags[t] = True
        return flags

    def _spectral_residual(self, signal, q=3):
        """
        Microsoft SR (KDD 2019):
        L(f) = ln|FFT(x)| → AL(f) = h_q * L(f) → R(f) = L - AL
        S(t) = |IFFT(e^{R+jφ})|²
        36-69% F1 improvement over best baselines.
        """
        X = fft(signal)
        log_amp = np.log(np.abs(X) + 1e-10)
        kernel = np.ones(q) / q
        avg_log_amp = np.convolve(log_amp, kernel, mode="same")
        residual = log_amp - avg_log_amp
        saliency = np.abs(ifft(np.exp(residual + 1j * np.angle(X))))**2
        threshold = np.mean(saliency) + 3 * np.std(saliency)
        return saliency > threshold
```

### Changepoint Detection

```python
"""
PELT (Killick et al., JASA 2012):
  F(t) = min_{s<t} [F(s) + C(x_{s+1:t}) + β]
  O(n) expected time via dynamic programming with pruning.
  ruptures 1.1.10: rpt.Pelt(model="rbf", min_size=7).fit_predict(signal, pen=10)

BOCPD (Adams & MacKay 2007):
  Bayesian online changepoint detection.
  Maintains run-length distribution updated via message passing.
  Real-time capable.
"""
```

---

## SP-7: Acoustic Simulation from 3D Venue Scans

### What to Build

Predict how furniture layouts affect speech intelligibility and music quality.
Feed Matterport meshes into acoustic simulation.

```python
# apps/ml-api/src/signal/acoustic/room_simulation.py

import pyroomacoustics as pra
import numpy as np

class VenueAcousticSimulator:
    """
    Sabine: RT60 = 0.161V/A  (V=m³, A=Σ α_i·S_i sabins)
      Best for live rooms (ᾱ < 0.2)
    Eyring: RT60 = 0.161V/(-S·ln(1-ᾱ))
      More accurate for treated rooms (ᾱ > 0.2)

    STI targets:
      Conference: STI ≥ 0.60, RT60 < 0.6-0.8s, NC ≤ 30
      Party/music: RT60 = 1.0-2.0s

    Absorption coefficients at 1kHz:
      Carpet on foam: 0.69
      Heavy curtains: 0.72
      Occupied fabric seats: 0.96
      Glass: 0.03
      Concrete: 0.02

    Image Source Method (Allen & Berkley 1979):
    Order-17 generates ~42,875 images.
    pyroomacoustics 0.8.6: hybrid ISM + ray tracing, C++ backend.

    Pipeline: Matterport OBJ → segment surfaces → assign absorption
    → pyroomacoustics → RIR → Web Audio ConvolverNode (browser preview)
    """

    def simulate_shoebox(self, dimensions, materials, source_pos, mic_pos,
                          max_order=17):
        room = pra.ShoeBox(
            dimensions, fs=16000,
            materials=pra.Material(energy_absorption=materials),
            max_order=max_order,
            ray_tracing=True,
            air_absorption=True)
        room.add_source(source_pos)
        room.add_microphone(mic_pos)
        room.compute_rir()

        rt60_sabine = room.rt60_theory(formula="sabine")
        rt60_eyring = room.rt60_theory(formula="eyring")

        return {
            "rir": room.rir[0][0],        # Impulse response
            "rt60_sabine": rt60_sabine,
            "rt60_eyring": rt60_eyring,
            "sample_rate": 16000,
        }

    def estimate_sti(self, rir, fs=16000):
        """
        Speech Transmission Index from impulse response.
        Excellent >0.75, Good 0.60-0.75, Fair 0.45-0.60, Poor <0.45.
        """
        # Modulation Transfer Function → STI computation
        ...

    def layout_acoustic_impact(self, room_dims, furniture_items):
        """
        Compute RT60 delta when furniture is added/removed.
        Soft furnishings (sofas, curtains) absorb → lower RT60.
        Hard surfaces (glass tables) reflect → higher RT60.
        Audience is the LARGEST absorber — filled vs empty RT60 differs dramatically.
        """
        ...
```

### Browser-Side Acoustic Preview

```
Web Audio API ConvolverNode: convolve audio with pre-computed RIR.
Feasible for impulses up to ~5 seconds on modern desktops.
Layout changes → swap impulse buffers (~10-150ms latency per swap).
For interactive preview: precompute grid of RIRs, interpolate.

Advanced: Physics-Informed Neural Networks for 3D room acoustics
(Applied Sciences, 2025), differentiable acoustic rendering (DART, 2024),
Neural Acoustic Fields (NeurIPS 2022) → gradient-based layout optimization.
```

---

## SP-8: Occupancy Sensing and Crowd Flow

```python
# apps/ml-api/src/signal/occupancy/wifi_csi.py

"""
WiFi CSI (Channel State Information):
  87-90% occupancy accuracy with DNN classifiers.
  Captures amplitude + phase across OFDM subcarriers.
  Far superior to MAC-address counting (40-60% accuracy reduction
  from iOS/Android MAC randomization).
  ESP32 with esp-csi library for data collection.
  IEEE 802.11bf formalizes WiFi sensing.
  Inherently privacy-preserving: no device identification.
"""

# apps/ml-api/src/signal/occupancy/co2_estimation.py

"""
CO2 mass balance: N = Q·(C_indoor - C_outdoor)/G
  G ≈ 0.005 L/s per person (at rest)
  NDIR sensors (Sensirion SCD41, ±30ppm): T90 = 30-120s
  Suitable for event-level analysis, not real-time headcount.
"""

# apps/ml-api/src/signal/occupancy/crowd_flow.py

"""
Social Force Model (Helbing & Molnar, PRE 1995):
  m_i·(dv_i/dt) = F_desire + Σ F_social + Σ F_obstacle

PySocialForce 1.1.2 (NumPy), socialforce (PyTorch, differentiable).

LWR macroscopic model: ∂ρ/∂t + ∂(ρ·v(ρ))/∂x = 0
  Treats crowds as fluid.

Particle filters for real-time Bayesian occupancy estimation:
  Maintain weighted particles {(N_t^i, w_t^i)}
  Systematic resampling when N_eff < N_p/2.

Privacy: zone-level aggregation in 5-minute buckets
with differential privacy noise eliminates individual tracking.
"""
```

---

## SP-9: Compressed Sensing for Sparse Venue Data

```python
# apps/ml-api/src/signal/compressed_sensing/sparse_recovery.py

"""
For new venues with few bookings: recover full demand curve
from far fewer samples than Nyquist requires.

If demand is s-sparse in frequency domain:
  M = O(s·log(N/s)) measurements suffice
  N=365 days, s=20 components → ~100-150 observations recover the signal

Recovery algorithms:
- Basis Pursuit: min ||x||₁ s.t. Φx = y  (cvxpy 1.8.1)
- OMP: greedy O(s²MN)  (scikit-learn OrthogonalMatchingPursuit)
- FISTA: O(1/k²) convergence

Matrix completion for venue-feature matrices:
  Nuclear norm minimization: min ||X||_*
  Rank-r matrix: O(r·(n₁+n₂)·log²n) observations needed
  (Candès & Recht 2009)
"""

from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.fft import dct, idct
import numpy as np

def recover_demand_curve(observed_days, observed_values, total_days=365,
                          n_components=10):
    """
    Recover full demand curve from sparse observations.
    Uses DCT basis (demand is sparse in frequency domain).
    """
    N = total_days
    M = len(observed_days)

    # Measurement matrix (row selection) × DCT basis
    Phi = np.eye(N)[observed_days]
    Psi = dct(np.eye(N), norm="ortho", axis=0)
    A = Phi @ Psi.T

    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_components)
    omp.fit(A, observed_values)
    return idct(omp.coef_, norm="ortho")
```

---

## SP-10: Advanced Time-Frequency Methods

```python
# apps/ml-api/src/signal/time_frequency/hht.py

"""
Hilbert-Huang Transform via EMD:
  Adaptively decomposes into Intrinsic Mode Functions.
  No predefined basis needed — data-driven.
  CEEMDAN (Complete Ensemble EMD with Adaptive Noise):
  solves mode mixing with exact reconstruction.
  EMD-signal 1.9.0: pip install EMD-signal (18% perf improvement over prior)

VMD (Variational Mode Decomposition):
  Solves variational optimization for K band-limited modes.
  More robust to noise than EMD.
  VMD-LSTM and VMD-Transformer dominate 2024-2025 benchmarks.

Synchrosqueezing (SST):
  Reassigns CWT coefficients → sharper time-frequency representation
  beyond Heisenberg uncertainty limit for AM-FM signals.
  ssqueezepy 0.6.6: 160K-point signals in 86ms GPU vs 8.4s CPU.

Stockwell Transform:
  Combines STFT absolute phase with CWT multiresolution.
  stockwell 1.1.2: conda install -c conda-forge stockwell

Computation costs (single-threaded, N=10,000):
  STFT: ~5ms, CWT: ~200ms, SSQ CWT: ~250ms
  WVD: ~2-5s, EMD: ~100-500ms, EEMD (200 trials): ~20-100s
  VMD: ~1-3s
"""
```

---

## SP-11: Streaming Signal Processing Architecture

```typescript
// packages/signal-core/src/fft/sliding_dft.ts

/**
 * Sliding DFT: O(N) recursive bin update per new sample.
 * X_k[n] = e^{j2πk/N} · (X_k[n-1] + x[n] - x[n-N])
 *
 * vs O(N log N) for full FFT recomputation.
 *
 * Goertzel algorithm: single DFT bin via 2nd-order IIR.
 * More efficient than full FFT when computing < log₂(N) bins.
 * Track specific cycles: weekly (k=N/7), monthly (k=N/30).
 */

// packages/signal-core/src/streaming/stream_processor.ts

/**
 * Dashboard architecture:
 * WebSocket ingestion → signal preprocessor (EMA, resampling)
 *   → streaming spectral (SDFT per-bin + Goertzel targeted)
 *   → anomaly detection (KL divergence current vs baseline spectrum)
 *   → binary WebSocket delivery at ~10Hz
 *   → D3/Canvas spectrogram rendering
 *
 * Web Audio API AudioWorklet:
 * Dedicated real-time audio thread, 128-sample blocks.
 * Communication via MessagePort or SharedArrayBuffer (COOP/COEP).
 * ringbuf.js (Paul Adenot/Mozilla): wait-free SPSC ring buffers,
 * 325% performance improvement via loop unrolling in v0.4.0.
 *
 * TypeScript libraries:
 *   fft.js 4.0.4 — fastest pure JS radix-4 FFT
 *   als-fft 3.4.1 — STFT with configurable overlap/windowing
 *   fir-dsp 1.0.2 — WASM FIR convolution
 */
```

---

## SP-12: WASM and WebGPU Signal Processing

### What to Build

Browser-side FFT, wavelet, and acoustic processing — no server round-trips.

```rust
// compute-wasm/src/fft.rs

// Cargo.toml: rustfft = { version = "6.4", features = ["wasm_simd"] }
use wasm_bindgen::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};

#[wasm_bindgen]
pub fn magnitude_spectrum(real_input: &[f32], size: usize) -> Vec<f32> {
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let mut buffer: Vec<Complex<f32>> = real_input.iter()
        .map(|&r| Complex { re: r, im: 0.0 }).collect();
    fft.process(&mut buffer);
    buffer.iter().take(size / 2)
        .map(|c| (c.re * c.re + c.im * c.im).sqrt()).collect()
}

// Build: RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --target web --release
```

### WebGPU FFT

```
Stockham auto-sort algorithm (avoids explicit bit-reversal):
log₂(N) compute shader dispatches with butterfly operations.
WGSL twiddle factors computed per stage.

Existing implementations:
- BabylonJS (Popov72) — ocean simulation FFT
- zkSecurity NTT in WGSL — 5× speedup on polynomial evaluation

1M-point FFT performance:
  FFTW native (AVX):     ~2-5ms
  RustFFT WASM (SIMD):   ~5-15ms
  WebGPU compute:        ~1-5ms (excl. transfer)
  fft.js (pure JS):      ~50-100ms

Critical bottleneck: CPU→GPU data transfer (2-10ms), not computation.
WebGPU outperforms WebGL for large inputs and loop-driven algorithms
but underperforms CPU for small inputs due to setup overhead
(ACM IMC 2025: "From WebGL to WebGPU").

WASM vs native: raw WASM ~4× faster than pure JS;
with SIMD ~6×. WASM within 1.1-2× of native.
Rust→WASM 9% faster than C++→WASM (Dec 2025 benchmark).

Browser-side acoustic preview:
  Precompute RIRs via ray tracing in WebGPU compute shaders →
  FFT overlap-add convolution reverb → AudioWorklet streaming.
  Layout changes: swap impulse buffers (~10-150ms latency).
```

---

## Integration with Other Techniques

- **Stochastic Pricing** (SP-tech): Fourier-extracted seasonality feeds pricing models;
  Kalman-filtered demand state drives real-time price adjustments
- **Statistical Learning Theory** (SLT): Wavelet/VMD decomposed components fed to
  tree ensembles (SLT-6) or GPs (SLT-5); conformal prediction wraps the reconstructed
  forecast for guaranteed coverage
- **Physics Solvers** (PS): Acoustic simulation validates layout energy terms;
  crowd flow integrates with egress energy (PS-5)
- **Computer Vision** (CV): Matterport meshes feed acoustic simulation;
  material classification from scan textures assigns absorption coefficients
- **HPC** (HPC): FFT runs in WebGPU compute shaders (HPC-1);
  RustFFT compiled to WASM (HPC-2); streaming via Web Workers (HPC-3)

---

## Session Management

1. **SP-1** (Fourier: FFT, Welch, STFT, multitaper, frequency filtering) — 1 session
2. **SP-2** (Wavelets: DWT, MODWT, denoising, multi-scale forecast) — 1 session
3. **SP-3** (Kalman: KF/EKF/UKF, multi-sensor fusion, adaptive, RTS) — 1-2 sessions
4. **SP-4** (Digital filters: Butterworth, SG, median, bandpass pipeline) — 1 session
5. **SP-5** (Cross-spectral: coherence, cepstrum, spectral clustering) — 1 session
6. **SP-6** (Anomaly: SR, Matrix Profile, CUSUM, PELT, STL, ensemble) — 1-2 sessions
7. **SP-7** (Acoustic: pyroomacoustics, RT60, STI, material DB, RIR) — 1-2 sessions
8. **SP-8** (Occupancy: WiFi CSI, CO2, crowd flow, particle filter) — 1 session
9. **SP-9** (Compressed sensing: OMP, basis pursuit, matrix completion) — 1 session
10. **SP-10** (Time-frequency: HHT/EMD, VMD, SST, Stockwell) — 1 session
11. **SP-11** (Streaming: SDFT, Goertzel, AudioWorklet, dashboard) — 1 session
12. **SP-12** (WASM/WebGPU: rustfft WASM SIMD, WebGPU FFT, acoustic preview) — 1-2 sessions

Total: ~12-16 Claude Code sessions.
