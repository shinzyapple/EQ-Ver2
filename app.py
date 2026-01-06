import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import get_window, lfilter
import io

# ===== 1/3オクターブ中心周波数 =====
OCTAVE_BANDS = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
    250, 315, 400, 500, 630, 800, 1000, 1250, 1600,
    2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
])

# ===== 音声処理 =====
def load_audio(file):
    data, sr = sf.read(file)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float64), sr

def compute_fft(signal, sr):
    window = get_window("hann", len(signal))
    fft = np.fft.rfft(signal * window)
    freq = np.fft.rfftfreq(len(signal), 1 / sr)
    mag = np.abs(fft)
    return freq, mag

def a_weighting(freq):
    f2 = freq ** 2
    ra = (12200**2 * f2**2) / (
        (f2 + 20.6**2)
        * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
        * (f2 + 12200**2)
    )
    return 20 * np.log10(ra + 1e-12) + 2.0

def band_energy(freq, mag, center_freq):
    f_low = center_freq / (2 ** (1/6))
    f_high = center_freq * (2 ** (1/6))
    idx = np.where((freq >= f_low) & (freq <= f_high))[0]
    if len(idx) == 0:
        return 0.0
    weights = 10 ** (a_weighting(freq[idx]) / 20)
    return np.mean(mag[idx] * weights)

def analyze(sig_a, sig_b, sr):
    freq_a, mag_a = compute_fft(sig_a, sr)
    freq_b, mag_b = compute_fft(sig_b, sr)

    eq = {}
    for band in OCTAVE_BANDS:
        ea = band_energy(freq_a, mag_a, band)
        eb = band_energy(freq_b, mag_b, band)
        diff = 20 * np.log10((eb + 1e-9) / (ea + 1e-9))
        eq[band] = float(np.clip(diff, -30, 30))
    return eq

# ===== RBJ EQ =====
def peaking_eq(signal, sr, fc, gain_db, Q):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1 / a0, a2 / a0])

    return lfilter(b, a, signal)

def apply_eq(signal, sr, eq):
    Q = 1 / (2 ** (1/6) - 2 ** (-1/6))
    out = signal.copy()
    for fc, gain in eq.items():
        if fc < sr / 2 and abs(gain) > 0.01:
            out = peaking_eq(out, sr, fc, gain, Q)
    return out

# ===== Streamlit UI =====
st.title("1/3 Octave EQ Analyzer")

file_a = st.file_uploader("音声ファイルA", type=["wav", "flac", "aiff"])
file_b = st.file_uploader("音声ファイルB", type=["wav", "flac", "aiff"])

if file_a and file_b:
    sig_a, sr_a = load_audio(file_a)
    sig_b, sr_b = load_audio(file_b)

    if sr_a != sr_b:
        st.error("サンプリングレートが一致していません")
    else:
        if st.button("解析"):
            eq = analyze(sig_a, sig_b, sr_a)

            st.subheader("EQ設定（dB）")
            st.code("\n".join([f"{k} Hz : {v:+.2f}" for k, v in eq.items()]))

            processed = apply_eq(sig_a, sr_a, eq)

            buf = io.BytesIO()
            sf.write(buf, processed, sr_a, format="WAV")
            st.download_button(
                "EQ適用後音声をダウンロード",
                buf.getvalue(),
                file_name="eq_result.wav",
                mime="audio/wav"
            )
