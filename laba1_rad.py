# app.py
# Streamlit-приложение для лабораторной: запись аудио (в браузере) + LPC анализ + генерация Word-отчёта
# ЗАМЕНА sounddevice -> streamlit-webrtc (работает в Streamlit, т.к. микрофон доступен только браузеру)

import streamlit as st
from scipy.io.wavfile import write, read
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import tempfile
import time
import pandas as pd
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from io import BytesIO

# === WebRTC audio capture ===
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av


# ==================== WORD REPORT ====================



def generate_word_report(template_path, context):
    """
    Генерация отчета в формате Word на основе шаблона с метками.
    Использует docxtpl для правильной замены меток и вставки изображений.
    """
    try:
        doc = DocxTemplate(template_path)
        render_context = context.copy()

        # matplotlib figures -> InlineImage
        for key, value in context.items():
            if key.startswith("graph_") and value is not None:
                buf = BytesIO()
                value.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                render_context[key] = InlineImage(doc, buf, width=Mm(150))
                plt.close(value)

        doc.render(render_context)

        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)
        return doc_bytes

    except Exception as e:
        st.error(f"Ошибка при генерации отчета: {str(e)}")
        raise


# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================

st.set_page_config(
    page_title="Лабораторная работа: LPC анализ речи",
    page_icon="🎤",
    layout="wide"
)


# ==================== SESSION STATE ====================

def init_session_state():
    defaults = {
        "variant": 13,
        "student_name": "Иванов И.И.",
        "recordings": {},          # {'8000': AudioRecorder, '11025': AudioRecorder}
        "lpc_results": None,
        "lpc_params": None,
        "current_page": "Запись аудио",
        "audio_files_exist": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# ==================== WEBRTC AUDIO ====================

class SimpleAudioCollector(AudioProcessorBase):
    """
    Собирает входящие аудиофреймы (av.AudioFrame) во время работы webrtc_streamer.
    """
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame


def _frames_to_mono_float(frames) -> tuple[np.ndarray, int]:
    """
    frames (list[av.AudioFrame]) -> (mono float64 array, sample_rate)
    """
    if not frames:
        return np.array([], dtype=np.float64), 0

    chunks = []
    sr = frames[0].sample_rate or 0

    for fr in frames:
        sr = fr.sample_rate or sr
        arr = fr.to_ndarray()  # (channels, samples) typically
        chunks.append(arr)

    x = np.concatenate(chunks, axis=1) if chunks else np.zeros((1, 0), dtype=np.float32)

    # mono
    if x.ndim == 2 and x.shape[0] > 1:
        x = np.mean(x, axis=0)
    elif x.ndim == 2:
        x = x[0]
    else:
        x = x.astype(np.float64)

    x = x.astype(np.float64)

    # if input is int, normalize
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x / maxv

    # normalize peak a bit
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 0:
        x = 0.9 * x / m

    return x, int(sr) if sr else 0


def resample_and_fix_duration(x: np.ndarray, in_sr: int, out_sr: int, duration_s: Optional[int]) -> np.ndarray:
    """
    Ресемпл + (опционально) обрезка/дополнение до duration_s секунд.
    """
    if x.size == 0:
        x = np.zeros(0, dtype=np.float64)

    fs = in_sr if in_sr else out_sr
    y = x

    if fs != out_sr:
        y = signal.resample_poly(y, out_sr, fs)
        fs = out_sr

    if duration_s is not None and duration_s > 0:
        target_len = int(out_sr * duration_s)
        if y.size > target_len:
            y = y[:target_len]
        elif y.size < target_len:
            y = np.pad(y, (0, target_len - y.size))

    # final normalize
    m = np.max(np.abs(y)) if y.size else 0.0
    if m > 0:
        y = 0.9 * y / m

    return y.astype(np.float64)


def float_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


# ==================== AUDIO CLASSES ====================

@dataclass
class AudioConfig:
    samplerate: int
    duration: int
    filename: str
    color: str = "blue"

    @property
    def total_samples(self) -> int:
        return self.duration * self.samplerate

    @property
    def size_kb(self) -> Optional[float]:
        if os.path.exists(self.filename):
            return os.path.getsize(self.filename) / 1024
        return None


class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.rate: Optional[int] = None
        self.data: Optional[np.ndarray] = None
        self.bit_depth: Optional[int] = None

    def save_array(self, fs: int, data: np.ndarray) -> "AudioRecorder":
        """
        Сохраняет wav на диск и подгружает параметры.
        """
        write(self.config.filename, fs, data)
        return self.load()

    def load(self) -> "AudioRecorder":
        if os.path.exists(self.config.filename):
            self.rate, self.data = read(self.config.filename)

            if np.issubdtype(self.data.dtype, np.integer):
                self.bit_depth = self.data.dtype.itemsize * 8
            else:
                self.bit_depth = self.data.dtype.itemsize * 8
        return self

    def get_info(self) -> dict:
        if self.data is None or self.rate is None:
            return {}
        return {
            "samplerate": self.rate,
            "samples": len(self.data),
            "duration": len(self.data) / self.rate if self.rate else 0.0,
            "size_kb": self.config.size_kb,
            "bit_depth": self.bit_depth,
            "dtype": str(self.data.dtype)
        }

    def get_fragment(self, start_sample: int, end_sample: int) -> np.ndarray:
        if self.data is None:
            return np.array([])
        return self.data[start_sample:end_sample]

    def file_exists(self) -> bool:
        return os.path.exists(self.config.filename)

    def get_audio_bytes(self):
        if os.path.exists(self.config.filename):
            with open(self.config.filename, "rb") as f:
                return f.read()
        return None


# ==================== LPC FUNCTIONS ====================

def make_window(name: str, N: int):
    name = name.lower()
    if name in ("hann", "hanning"):
        return signal.windows.hann(N, sym=False)
    if name in ("hamming",):
        return signal.windows.hamming(N, sym=False)
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(N)
    raise ValueError("Unknown window. Use: hann, hamming, rect")


def frame_signal(x: np.ndarray, frame_len: int, hop: int):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < frame_len:
        x = np.pad(x, (0, frame_len - n))
        n = len(x)
    num = 1 + int(np.ceil((n - frame_len) / hop))
    pad = (num - 1) * hop + frame_len - n
    if pad > 0:
        x = np.pad(x, (0, pad))
    frames = np.stack([x[i * hop:i * hop + frame_len] for i in range(num)], axis=0)
    return frames


def overlap_add(frames: np.ndarray, hop: int):
    num, frame_len = frames.shape
    out_len = (num - 1) * hop + frame_len
    y = np.zeros(out_len, dtype=np.float64)
    for i in range(num):
        y[i * hop:i * hop + frame_len] += frames[i]
    return y


def levinson_durbin(r: np.ndarray, order: int):
    r = np.asarray(r, dtype=np.float64)
    if len(r) < order + 1:
        raise ValueError("r must have length >= order+1")

    if r[0] <= 1e-12:
        return np.zeros(order), 0.0, np.zeros(order)

    a = np.zeros(order + 1, dtype=np.float64)
    e = r[0]
    a[0] = 1.0
    k = np.zeros(order, dtype=np.float64)

    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        ki = -(r[i] + acc) / e
        k[i - 1] = ki

        a_new = a.copy()
        for j in range(1, i):
            a_new[j] = a[j] + ki * a[i - j]
        a_new[i] = ki
        a = a_new

        e *= (1.0 - ki * ki)
        if e < 1e-12:
            e = 1e-12

    return a[1:], e, k


def autocorr(x: np.ndarray, order: int):
    x = np.asarray(x, dtype=np.float64)
    r_full = np.correlate(x, x, mode="full")
    mid = len(r_full) // 2
    r = r_full[mid:mid + order + 1]
    return r


def lpc_encode_frames(frames: np.ndarray, order: int):
    num, frame_len = frames.shape
    A = np.zeros((num, order), dtype=np.float64)
    E = np.zeros(num, dtype=np.float64)
    R_frames = np.zeros_like(frames)

    for i in range(num):
        x = frames[i]
        r = autocorr(x, order)
        a, e, _ = levinson_durbin(r, order)
        A[i] = a
        E[i] = e
        R_frames[i] = signal.lfilter(np.r_[1.0, A[i]], [1.0], x)

    return A, E, R_frames


def lpc_synthesize_frames(A: np.ndarray, E: np.ndarray, frame_len: int, excitation="noise"):
    num, order = A.shape
    frames_hat = np.zeros((num, frame_len), dtype=np.float64)

    for i in range(num):
        if excitation == "noise":
            src = np.random.randn(frame_len) * np.sqrt(max(E[i], 1e-12))
        else:
            raise ValueError("Only 'noise' excitation implemented")

        den = np.r_[1.0, A[i]]
        frames_hat[i] = signal.lfilter([1.0], den, src)

    return frames_hat


def run_lpc_codec(
    audio_data: np.ndarray,
    original_fs: int,
    target_fs=8000,
    frame_ms=30,
    overlap=0.5,
    window_name="hann",
    order=10
):
    x = audio_data.copy()

    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)

    if np.max(np.abs(x)) > 0:
        x = 0.9 * x / np.max(np.abs(x))

    fs = original_fs
    if target_fs is not None and fs != target_fs:
        x = signal.resample_poly(x, target_fs, fs)
        fs = target_fs

    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop = int(round(frame_len * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("Overlap too large -> hop <= 0")

    w = make_window(window_name, frame_len)

    frames = frame_signal(x, frame_len, hop)
    frames_w = frames * w[None, :]

    A, E, R_frames = lpc_encode_frames(frames_w, order)
    frames_hat = lpc_synthesize_frames(A, E, frame_len, excitation="noise")

    y = overlap_add(frames_hat, hop)
    r = overlap_add(R_frames, hop)

    y = y[:len(x)]
    r = r[:len(x)]

    t = np.arange(len(x)) / fs

    if np.max(np.abs(y)) > 0:
        y = 0.9 * y / np.max(np.abs(y))

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, fs, float_to_int16(y))

    return {
        "fs": fs,
        "x": x,
        "y": y,
        "r": r,
        "A": A,
        "E": E,
        "output_file": temp_file.name,
        "frame_len": frame_len,
        "hop": hop,
        "num_frames": A.shape[0]
    }


def plot_residual_vs_order(audio_data, original_fs, target_fs=8000, frame_ms=30, frame_number=10, max_order=20):
    x = audio_data.copy()

    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)

    fs = original_fs
    if fs != target_fs:
        x = signal.resample_poly(x, target_fs, fs)
        fs = target_fs

    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop = int(round(frame_len * 0.5))

    frames = frame_signal(x, frame_len, hop)
    w = make_window("hann", frame_len)
    frames_w = frames * w[None, :]

    if frame_number >= len(frames_w):
        frame_number = len(frames_w) - 1

    frame_data = frames_w[frame_number]

    orders = range(1, max_order + 1)
    residual_powers = []

    for order in orders:
        r = autocorr(frame_data, order)
        if r[0] > 1e-12:
            _, e, _ = levinson_durbin(r, order)
            residual_powers.append(e)
        else:
            residual_powers.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(orders), residual_powers, "b-o", linewidth=2, markersize=8)
    ax.set_xlabel("Порядок предсказания (p)")
    ax.set_ylabel("Мощность остатка предсказания")
    ax.set_title(f"Зависимость мощности остатка от порядка LPC\nФрейм #{frame_number}")
    ax.grid(True, alpha=0.3)

    return fig, list(orders), residual_powers


# ==================== PLOTS ====================

def create_signal_plot(data, samplerate, title, color="blue"):
    duration = len(data) / samplerate if samplerate else 0
    time_arr = np.linspace(0, duration, len(data)) if len(data) else np.array([])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_arr, data, color=color, linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Амплитуда")
    return fig


def create_waveform_plots(t, x_in, x_out, x_res, title_prefix=""):
    figs = []

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, x_in, "b-", linewidth=0.5)
    ax1.set_title(f"{title_prefix}Входной сигнал (осциллограмма)")
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")
    ax1.grid(True, alpha=0.3)
    figs.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t[:len(x_out)], x_out, "r-", linewidth=0.5)
    ax2.set_title(f"{title_prefix}Синтезированный сигнал (осциллограмма)")
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Амплитуда")
    ax2.grid(True, alpha=0.3)
    figs.append(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t[:len(x_res)], x_res, "g-", linewidth=0.5)
    ax3.set_title(f"{title_prefix}Остаток предсказания (осциллограмма)")
    ax3.set_xlabel("Время, с")
    ax3.set_ylabel("Амплитуда")
    ax3.grid(True, alpha=0.3)
    figs.append(fig3)

    return figs


def create_spectrogram(x, fs, title, nperseg=256, noverlap=192):
    f, tt, Sxx = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        scaling="spectrum", mode="magnitude"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(tt, f, 20 * np.log10(Sxx + 1e-12), shading="auto", cmap="viridis")
    ax.set_ylim(0, fs / 2)
    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    plt.colorbar(pcm, ax=ax, label="Уровень, dB")
    return fig


# ==================== STREAMLIT UI ====================

def run():
    st.sidebar.title("Навигация")
    pages = ["Запись аудио", "LPC анализ", "Отчет"]
    st.session_state.current_page = st.sidebar.radio("Перейти к:", pages)

    st.sidebar.title("Настройки")
    st.session_state.variant = st.sidebar.number_input(
        "Номер варианта", min_value=1, max_value=30, value=st.session_state.variant
    )
    st.session_state.student_name = st.sidebar.text_input("ФИО студента", value=st.session_state.student_name)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры LPC анализа")
    frame_ms = st.sidebar.slider("Размер сегмента (мс)", min_value=10, max_value=50, value=30, step=5)
    overlap = st.sidebar.slider("Перекрытие", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
    lpc_order = st.sidebar.slider("Порядок предсказания", min_value=2, max_value=20, value=10, step=1)

    if st.session_state.current_page == "Запись аудио":
        show_recording_page()
    elif st.session_state.current_page == "LPC анализ":
        show_lpc_analysis_page(frame_ms, overlap, lpc_order)
    elif st.session_state.current_page == "Отчет":
        show_report_page(frame_ms, overlap, lpc_order)


def show_recording_page():
    # картинки как в твоём коде (если их нет — просто пропустим)
    for img in ("prikol.jpg", "prikol1.jpg"):
        if os.path.exists(img):
            st.image(img)

    st.title("🎤 Запись аудиофайлов")
    st.markdown(
        "Данная лабораторная работа предполагает работу со файлом .wav (получение его значений параметров), "
        "работу с LPC-кодеком. Для этого необходимо записать два звуковых сигнала в разделе 'Запись аудио'. "
        "Далее — 'LPC-анализ' и запуск анализа. Потом — 'Отчет' и генерация Word.\n\n"
        "ВНИМАНИЕ: Вариантом, по сути, является лишь фамилия."
    )

    st.markdown("""
    ### Инструкция
    1. Нажмите **Start** в нужной колонке
    2. Дождитесь пока кнопка **Stop** станет с красным фоном
    3. Произнесите фамилию (примерно 5 секунд, можно чуть больше)
    4. Нажмите **Сохранить запись**
    5. Нажмите **Stop**
    """)

    col1, col2 = st.columns(2)

    def webrtc_record_block(title: str, key_prefix: str, target_sr: int, filename: str, color: str, state_key: str):
        st.subheader(title)

        # где храним кадры между перезапусками
        frames_key = f"{key_prefix}_frames"
        if frames_key not in st.session_state:
            st.session_state[frames_key] = []

        # кнопка "очистить запись"
        if st.button("🧹 Очистить запись", key=f"{key_prefix}_clear"):
            st.session_state[frames_key] = []
            st.success("Буфер записи очищен.")

        ctx = webrtc_streamer(
            key=f"{key_prefix}_webrtc",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={
                "audio": {
                    "echoCancellation": False,
                    "noiseSuppression": False,
                    "autoGainControl": False,
                    "channelCount": 1
                },
                "video": False
            },
            async_processing=True,
        )

        st.caption("Запись идёт в браузере: Start → говорите → Stop → Сохранить.")

        # ВАЖНО: пока идет запись, вытаскиваем фреймы из audio_receiver и копим в session_state
        if ctx and ctx.state.playing and ctx.audio_receiver:
            try:
                while True:
                    audio_frames = ctx.audio_receiver.get_frames(timeout=0.01)
                    if not audio_frames:
                        break
                    st.session_state[frames_key].extend(audio_frames)
            except Exception:
                # таймауты/пустые очереди — нормальная ситуация
                pass

        st.write(f"Кадров в буфере: {len(st.session_state[frames_key])}")

        if st.button(f"💾 Сохранить запись ({target_sr} Гц)", key=f"{key_prefix}_save"):
            frames = st.session_state[frames_key]
            if not frames:
                st.warning("Буфер пуст. Нажмите Start, поговорите 5 секунд, Stop — и попробуйте снова.")
                return

            # frames -> mono float -> ресемпл -> int16 wav
            x, in_sr = _frames_to_mono_float(frames)
            if x.size == 0:
                st.warning("Не удалось извлечь аудио. Попробуйте записать ещё раз.")
                return

            x_rs = resample_and_fix_duration(x, in_sr=in_sr, out_sr=target_sr, duration_s=5)
            x16 = float_to_int16(x_rs)
            x_rs = np.clip(x_rs, -1.0, 1.0)
            x32 = x_rs.astype(np.float32)

            config = AudioConfig(target_sr, 5, filename, color)
            recorder = AudioRecorder(config).save_array(target_sr, x32)
            st.session_state.recordings[state_key] = recorder

            st.success("Запись сохранена!")

        # отображение сохраненного
        if state_key in st.session_state.recordings:
            recorder = st.session_state.recordings[state_key]
            info = recorder.get_info()

            st.write("**Информация:**")
            st.write(f"- Длительность: {info.get('duration', 0):.2f} с")
            st.write(f"- Размер: {info.get('size_kb', 0):.2f} КБ")
            st.write(f"- Глубина квантования: {info.get('bit_depth', 'N/A')} бит")

            audio_bytes = recorder.get_audio_bytes()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.download_button(
                    label=f"📥 Скачать ({target_sr} Гц)",
                    data=audio_bytes,
                    file_name=os.path.basename(filename),
                    mime="audio/wav",
                    key=f"{key_prefix}_download"
                )

            fig = create_signal_plot(recorder.data, recorder.rate, f"Сигнал ({target_sr} Гц)", color)
            st.pyplot(fig)
            plt.close(fig)

    with col1:
        webrtc_record_block("Запись 1 (8000 Гц)", "rec8000", 8000, "output.wav", "blue", "8000")

    with col2:
        webrtc_record_block("Запись 2 (11025 Гц)", "rec11025", 11025, "output11025.wav", "green", "11025")

    # Фрагмент сигнала (как у тебя)
    st.markdown("---")
    st.subheader("Фрагмент сигнала")

    if "8000" in st.session_state.recordings:
        recorder = st.session_state.recordings["8000"]

        start_sample = 5000
        end_sample = 1000 * st.session_state.variant
        if start_sample >= end_sample:
            start_sample = 4000
            end_sample = 5000

        fragment = recorder.get_fragment(start_sample, end_sample)

        st.write("**Параметры фрагмента:**")
        st.write(f"- Начальный сэмпл: {start_sample}")
        st.write(f"- Конечный сэмпл: {end_sample}")
        st.write(f"- Количество сэмплов: {len(fragment)}")
        st.write(f"- Длительность: {len(fragment) / recorder.rate:.3f} с")

        fig = create_signal_plot(fragment, recorder.rate, f"Фрагмент сигнала (сэмплы {start_sample}-{end_sample})", "red")
        st.pyplot(fig)
        plt.close(fig)


def show_lpc_analysis_page(frame_ms, overlap, lpc_order):
    st.title("🔬 LPC анализ и синтез речи")

    if "11025" not in st.session_state.recordings:
        st.warning("Сначала выполните запись аудиофайлов на странице 'Запись аудио'")
        return

    recorder = st.session_state.recordings["11025"]

    st.write(f"**Анализируемый файл:** {recorder.config.filename}")
    st.write(f"**Частота дискретизации:** {recorder.rate} Гц")
    st.write("**Параметры анализа:**")
    st.write(f"- Размер сегмента: {frame_ms} мс")
    st.write(f"- Перекрытие: {overlap:.1%}")
    st.write(f"- Порядок предсказания: {lpc_order}")

    if st.button("🚀 Запустить LPC анализ", type="primary"):
        with st.spinner("Выполняется LPC анализ..."):
            results = run_lpc_codec(
                audio_data=recorder.data,
                original_fs=recorder.rate,
                target_fs=8000,
                frame_ms=frame_ms,
                overlap=overlap,
                window_name="hann",
                order=lpc_order
            )

            st.session_state.lpc_results = results
            st.session_state.lpc_params = {
                "frame_ms": frame_ms,
                "overlap": overlap,
                "lpc_order": lpc_order
            }

            st.success("Анализ завершен!")
            show_lpc_results(results)

    elif st.session_state.lpc_results is not None:
        show_lpc_results(st.session_state.lpc_results)

    st.markdown("---")
    st.subheader("Анализ зависимости остатка от порядка LPC")

    col1, col2 = st.columns(2)
    with col1:
        frame_number = st.number_input("Номер фрейма для анализа", min_value=0, value=10, step=1)
    with col2:
        max_order = st.number_input("Максимальный порядок", min_value=5, max_value=30, value=20, step=1)

    if st.button("📊 Построить график зависимости"):
        fig, orders, powers = plot_residual_vs_order(
            audio_data=recorder.data,
            original_fs=recorder.rate,
            target_fs=8000,
            frame_ms=frame_ms,
            frame_number=frame_number,
            max_order=max_order
        )

        st.pyplot(fig)
        plt.close(fig)

        # сохраняем для отчёта
        st.session_state.residual_plot = fig

        df = pd.DataFrame({
            "Порядок": orders,
            "Мощность остатка": powers,
            "Относительное уменьшение": [1 - p / powers[0] if powers[0] > 0 else 0 for p in powers]
        })
        st.dataframe(df)


def show_lpc_results(results):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Частота дискретизации", f"{results['fs']} Гц")
    with col2:
        st.metric("Количество фреймов", results["num_frames"])
    with col3:
        st.metric("Длина фрейма", f"{results['frame_len']} сэмплов")

    st.subheader("Осциллограммы")
    t = np.arange(len(results["x"])) / results["fs"]
    figs = create_waveform_plots(t, results["x"], results["y"], results["r"])

    tab1, tab2, tab3 = st.tabs(["Исходный", "Синтезированный", "Остаток"])
    with tab1:
        st.pyplot(figs[0])
    with tab2:
        st.pyplot(figs[1])
    with tab3:
        st.pyplot(figs[2])

    st.session_state.osc_plots = figs

    st.subheader("Спектрограммы")
    spec_figs = [
        create_spectrogram(results["x"], results["fs"], "Спектрограмма исходного сигнала"),
        create_spectrogram(results["y"], results["fs"], "Спектрограмма синтезированного сигнала"),
        create_spectrogram(results["r"], results["fs"], "Спектрограмма остатка"),
    ]

    tab1, tab2, tab3 = st.tabs(["Исходный", "Синтезированный", "Остаток"])
    with tab1:
        st.pyplot(spec_figs[0])
    with tab2:
        st.pyplot(spec_figs[1])
    with tab3:
        st.pyplot(spec_figs[2])

    st.session_state.spect_plots = spec_figs

    st.subheader("Синтезированный сигнал")
    if os.path.exists(results["output_file"]):
        with open(results["output_file"], "rb") as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/wav")
            st.download_button(
                label="📥 Скачать синтезированный сигнал",
                data=audio_bytes,
                file_name=f"synthesized_{st.session_state.lpc_params['lpc_order']}.wav",
                mime="audio/wav"
            )

    st.subheader("Статистика")
    total_params = results["A"].size + results["E"].size
    compression_ratio = len(results["x"]) / total_params if total_params > 0 else 0

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Параметров модели (A+E):** {total_params}")
        st.write(f"**Степень сжатия:** {compression_ratio:.2f} сэмплов/параметр")

    st.session_state.compression_ratio = compression_ratio
    st.session_state.total_params = total_params


def show_report_page(frame_ms, overlap, lpc_order):
    st.title("📄 Генерация отчета")

    required_data = [
        "8000" in st.session_state.recordings,
        "11025" in st.session_state.recordings,
        st.session_state.lpc_results is not None
    ]
    if not all(required_data):
        st.warning("Для генерации отчета необходимо выполнить запись обоих файлов и LPC анализ")
        return

    template_path = "pattern_rad_lab1.docx"
    if not os.path.exists(template_path):
        st.error(f"❌ Файл шаблона '{template_path}' не найден в текущей директории!")
        with st.expander("📁 Содержимое текущей директории"):
            for file in os.listdir("."):
                st.write(f"- {file}")
        return

    st.success("✅ Шаблон найден")

    if st.button("📥 Сгенерировать отчет Word", type="primary"):
        with st.spinner("🔄 Генерация отчета..."):
            try:
                recorder_8000 = st.session_state.recordings["8000"]
                recorder_11025 = st.session_state.recordings["11025"]
                lpc_results = st.session_state.lpc_results

                info_8000 = recorder_8000.get_info()
                info_11025 = recorder_11025.get_info()

                start_sample = 5000
                end_sample = 1000 * st.session_state.variant
                if start_sample >= end_sample:
                    start_sample = 4000
                    end_sample = 5000

                fragment = recorder_8000.get_fragment(start_sample, end_sample)

                st.info("🔄 Подготавливаю графики...")

                graph_8000 = create_signal_plot(recorder_8000.data, recorder_8000.rate, "Полный сигнал (8000 Гц)", "blue")
                graph_frag = create_signal_plot(fragment, recorder_8000.rate, "Фрагмент сигнала", "red")
                graph_11025 = create_signal_plot(recorder_11025.data, recorder_11025.rate, "Полный сигнал (11025 Гц)", "green")

                total_params = lpc_results["A"].size + lpc_results["E"].size
                compression_ratio = len(lpc_results["x"]) / total_params if total_params > 0 else 0

                context = {
                    "name": st.session_state.student_name,

                    "size_kb_8000": f"{info_8000.get('size_kb', 0):.2f}",
                    "bit_depth_8000": str(info_8000.get("bit_depth", "N/A")),
                    "graph_8000": graph_8000,

                    "start_sample": str(start_sample),
                    "end_sample": str(end_sample),
                    "len_fragment": str(len(fragment)),
                    "time_frag": f"{len(fragment) / recorder_8000.rate:.3f}",
                    "graph_frag": graph_frag,

                    "size_kb_11025": f"{info_11025.get('size_kb', 0):.2f}",
                    "bit_depth_11025": str(info_11025.get("bit_depth", "N/A")),
                    "graph_11025": graph_11025,

                    # LPC часть
                    "fs_lpc": lpc_results["fs"],
                    # иногда в шаблонах/списках встречается опечатка fs_lps — подстрахуемся:
                    "fs_lps": lpc_results["fs"],

                    "frame_ms": str(frame_ms),
                    "frame_sem": str(lpc_results["frame_len"]),
                    "overlap": f"{overlap:.1%}",
                    "full_frame": str(lpc_results["num_frames"]),
                    "full_sem": str(len(lpc_results["x"])),
                    "lpc_order": str(lpc_order),
                    "count_order": str(total_params),
                    "coeff": f"{compression_ratio:.2f}",
                }

                if hasattr(st.session_state, "osc_plots") and st.session_state.osc_plots and len(st.session_state.osc_plots) >= 3:
                    context["graph_osc_orig"] = st.session_state.osc_plots[0]
                    context["graph_osc_sint"] = st.session_state.osc_plots[1]
                    context["graph_osc_frag"] = st.session_state.osc_plots[2]

                if hasattr(st.session_state, "spect_plots") and st.session_state.spect_plots and len(st.session_state.spect_plots) >= 3:
                    context["graph_spect_orig"] = st.session_state.spect_plots[0]
                    context["graph_spect_sint"] = st.session_state.spect_plots[1]
                    context["graph_spect_frag"] = st.session_state.spect_plots[2]

                if hasattr(st.session_state, "residual_plot") and st.session_state.residual_plot:
                    context["graph_zavis_lpc"] = st.session_state.residual_plot

                st.info("📄 Формирую отчет...")
                doc_bytes = generate_word_report(template_path, context)

                st.download_button(
                    label="📥 Скачать отчет",
                    data=doc_bytes,
                    file_name=f"LPC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

                plt.close("all")
                st.success("✅ Отчет успешно сгенерирован!")

            except Exception as e:
                st.error(f"❌ Ошибка при генерации отчета: {str(e)}")
                import traceback
                with st.expander("Детали ошибки"):
                    st.code(traceback.format_exc())


