import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import (
    firwin, filtfilt, resample_poly, periodogram, correlate, hilbert
)
from sklearn.preprocessing import StandardScaler
from fractions import Fraction

# ==== USER SETTINGS ====
wav_path     = r'put your wav file here'
fs_iq        = 20_48_000    # IQ sampling rate
rec_center   = 108.0e6     # RF center of recording
station_freq = 108.0e6     # VOR carrier tuned
audio_fs     = 50_000       # audio rate after decimation

# ==== FILTER TAP COUNTS ====
LP_TAPS      = 1021         # RF→audio low-pass (~12 kHz)
BPSUB_TAPS   = 501          # 9 960 ± 200 Hz band-pass
LP480_TAPS   = 501          # post-heterodyne low-pass (480 Hz)
BP30_TAPS    = 501          # 29–31 Hz band-pass

# ---- 1) LOAD IQ & MIX TO BASEBAND ----
sr, raw = wavfile.read(wav_path)
assert sr == fs_iq, f"Expected {fs_iq} Hz IQ, got {sr} Hz"
i = raw[:,0].astype(float)
q = raw[:,1].astype(float)
iq = i + 1j*q

t_iq = np.arange(len(iq)) / fs_iq
bb   = iq * np.exp(-2j*np.pi*(station_freq - rec_center)*t_iq)

# ---- 2) RF→AUDIO LP, DECIMATE & ENVELOPE ----
lp        = firwin(LP_TAPS, 12e3, fs=fs_iq)
audio_bb  = filtfilt(lp, 1, bb)
f         = Fraction(audio_fs, fs_iq).limit_denominator()
audio     = resample_poly(audio_bb, f.numerator, f.denominator)
fs_a      = audio_fs
t_a       = np.arange(len(audio)) / fs_a
env       = np.abs(audio)

# ---- 3) ESTIMATE SUBCARRIER CENTER FREQUENCY ----
# FFT of a 1 s slice, 0.5–1.5 s into the recording
slice_env     = env[int(0.5*fs_a):int(1.5*fs_a)]
f_axis, Pxx   = periodogram(slice_env, fs=fs_a, nfft=65536)
mask_sub      = (f_axis>9600)&(f_axis<10200)
sub_center    = f_axis[mask_sub][np.argmax(Pxx[mask_sub])]
print(f"Measured subcarrier center: {sub_center:.1f} Hz")

# ---- 4) DESIGN FILTERS ----
bp_sub   = firwin(BPSUB_TAPS, [sub_center-200, sub_center+200],
                   pass_zero=False, fs=fs_a)
lp_480   = firwin(LP480_TAPS, 480, fs=fs_a)
bp_30    = firwin(BP30_TAPS, [29,31], pass_zero=False, fs=fs_a)

# ---- 5) EXTRACT 30 Hz “VARIABLE” TONE (AM) ----
var_sig = filtfilt(bp_30, 1, env)

# ---- 6) EXTRACT 30 Hz “REFERENCE” TONE (FM-DEM) ----
#  a) isolate subcarrier
sub_cf = filtfilt(bp_sub, 1, env)
#  b) heterodyne exactly at measured sub_center
lo     = np.exp(-2j*np.pi*sub_center*t_a)
bb_lp  = filtfilt(lp_480, 1, sub_cf * lo)
#  c) instant. phase → instant. freq deviation
phi    = np.unwrap(np.angle(bb_lp))
inst_f = np.diff(phi)/(2*np.pi)*fs_a
inst_f = np.append(inst_f, inst_f[-1])  # pad last sample
#  d) band-pass to 30 Hz
ref_sig = filtfilt(bp_30, 1, inst_f)

# ---- 7) GATE STEADY-STATE ----
L        = min(len(var_sig), len(ref_sig), len(env))
t_sig    = t_a[:L]
var_sig  = var_sig[:L]
ref_sig  = ref_sig[:L]
env_sig  = env[:L]
inst_dev = inst_f[:L]

mask = (
    (t_sig>0.5) & (t_sig<t_sig[-1]-0.5) &        # discard startup/shutdown
    (env_sig>env_sig.mean()) &                  # only strong envelope
    (np.abs(inst_dev) < 50)                     # ±50 Hz FM deviation
)

# ---- 8) NORMALIZE & BUILD FEATURE MATRIX ----
# We’ll use ref & var samples as a 2-D feature for phase correlation
data = np.vstack([ref_sig[mask], var_sig[mask]]).T
scaler = StandardScaler().fit(data)
norm_data = scaler.transform(data)

# ---- 9) COMPUTE PHASE SHIFT VIA COVARIANCE ----
# covariance matrix → correlation → arccos → phase
xcov       = np.cov(norm_data.T)
corr_coef  = xcov[0,1]                   # off-diagonal
phase_rad  = np.arccos(np.clip(corr_coef, -1, 1))
phase_deg  = phase_rad * 180/np.pi
print(f"Decoded VOR bearing = {phase_deg:.1f}°")

# ==== 10) PLOTS ====

# Time-domain zoom (2.0–2.1 s)
t0, t1 = 2.0, 2.1
zm     = (t_sig>=t0) & (t_sig<=t1)
plt.figure(figsize=(8,3))
plt.plot(t_sig[zm], var_sig[zm]/np.std(var_sig[mask]), label='Variable (AM)')
plt.plot(t_sig[zm], ref_sig[zm]/np.std(ref_sig[mask]), label='Reference (FM)')
plt.title("Zoomed 30 Hz Signals")
plt.xlabel("Time [s]"); plt.ylabel("Normalized")
plt.legend(loc='upper right'); plt.tight_layout()

# Instantaneous phase-difference trace
plt.figure(figsize=(8,3))
pd_trace = np.angle(hilbert(var_sig)) - np.angle(hilbert(ref_sig))
pd_trace = np.mod(pd_trace * 180/np.pi, 360)
plt.plot(t_sig[zm], pd_trace[zm], '.', ms=2)
plt.axhline(phase_deg, color='r', ls='--', label=f'Bearing = {phase_deg:.1f}°')
plt.title("Instant. Phase Difference (Zoomed)")
plt.xlabel("Time [s]"); plt.ylabel("Phase (°)")
plt.legend(loc='upper right'); plt.tight_layout()

# Histogram of gated Δφ
plt.figure(figsize=(6,4))
hist, edges = np.histogram(pd_trace[mask], bins=360, range=(0,360))
centers      = (edges[:-1]+edges[1:])/2
plt.bar(centers, hist, width=1, alpha=0.7)
plt.axvline(phase_deg, color='r', ls='--', label=f'Bearing = {phase_deg:.1f}°')
plt.title("Phase-Difference Histogram")
plt.xlabel("Phase (°)"); plt.ylabel("Count")
plt.legend(loc='upper right'); plt.tight_layout()

plt.show()

