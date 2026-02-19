import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write, read
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from typing import Optional, List
import io
from datetime import datetime
import tempfile
import time
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import pandas as pd
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from io import BytesIO
import re


def generate_word_report(template_path, params):
    """
    Генерация отчета в формате Word на основе шаблона с метками
    Использует docxtpl для правильной замены меток и вставки изображений
    """
    try:
        # Загружаем шаблон
        doc = DocxTemplate(template_path)

        # Создаем копию параметров для контекста
        context = params.copy()

        # Конвертируем графики в InlineImage
        for key, value in params.items():
            if key.startswith('graph_') and value is not None:
                # Создаем буфер для графика
                buf = BytesIO()
                value.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)

                # Создаем InlineImage для вставки в документ
                # Ширина 150 мм (можно настроить под ваш шаблон)
                context[key] = InlineImage(doc, buf, width=Mm(150))

                # Закрываем фигуру matplotlib для освобождения памяти
                plt.close(value)

        # Рендерим документ с контекстом
        doc.render(context)

        # Сохраняем в байтовый поток
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)

        return doc_bytes

    except Exception as e:
        st.error(f"Ошибка при генерации отчета: {str(e)}")
        raise e

# ==================== НАСТРОЙКИ СТРАНИЦЫ ====================
st.set_page_config(
    page_title="Лабораторная работа: LPC анализ речи",
    page_icon="🎤",
    layout="wide"
)


# ==================== ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ СЕССИИ ====================
def init_session_state():
    """Инициализация переменных состояния сессии"""
    defaults = {
        'variant': 13,
        'student_name': 'Иванов И.И.',
        'recordings': {},
        'lpc_results': None,
        'current_page': 'Запись аудио',
        'audio_files_exist': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ==================== КЛАССЫ ДЛЯ РАБОТЫ С АУДИО ====================

@dataclass
class AudioConfig:
    """Конфигурация аудиозаписи"""
    samplerate: int
    duration: int
    filename: str
    color: str = 'blue'

    @property
    def total_samples(self) -> int:
        return self.duration * self.samplerate

    @property
    def size_kb(self) -> Optional[float]:
        if os.path.exists(self.filename):
            return os.path.getsize(self.filename) / 1024
        return None


class AudioRecorder:
    """Класс для работы с аудиозаписями"""

    def __init__(self, config: AudioConfig):
        self.config = config
        self.recording: Optional[np.ndarray] = None
        self.rate: Optional[int] = None
        self.data: Optional[np.ndarray] = None
        self.bit_depth: Optional[int] = None

    def record(self, progress_bar=None) -> 'AudioRecorder':
        """Запись аудио"""
        if progress_bar:
            progress_bar.progress(0.1)

        self.recording = sd.rec(
            self.config.total_samples,
            samplerate=self.config.samplerate,
            channels=1
        )

        # Имитация прогресса записи
        for i in range(10):
            time.sleep(self.config.duration / 10)
            if progress_bar:
                progress_bar.progress(0.1 + i * 0.09)

        sd.wait()
        if progress_bar:
            progress_bar.progress(1.0)

        return self

    def save(self) -> 'AudioRecorder':
        """Сохранение в файл"""
        if self.recording is not None:
            write(self.config.filename, self.config.samplerate, self.recording)
        return self

    def load(self) -> 'AudioRecorder':
        """Загрузка из файла"""
        if os.path.exists(self.config.filename):
            self.rate, self.data = read(self.config.filename)

            if np.issubdtype(self.data.dtype, np.integer):
                if self.data.dtype == np.int16:
                    self.bit_depth = 16
                elif self.data.dtype == np.int32:
                    self.bit_depth = 32
                elif self.data.dtype == np.int8:
                    self.bit_depth = 8
                else:
                    self.bit_depth = self.data.dtype.itemsize * 8
            else:
                self.bit_depth = self.data.dtype.itemsize * 8

        return self

    def get_info(self) -> dict:
        """Получение информации о файле"""
        if self.data is not None:
            return {
                'samplerate': self.rate,
                'samples': len(self.data),
                'duration': len(self.data) / self.rate,
                'size_kb': self.config.size_kb,
                'bit_depth': self.bit_depth,
                'dtype': str(self.data.dtype)
            }
        return {}

    def get_fragment(self, start_sample: int, end_sample: int) -> np.ndarray:
        """Получение фрагмента записи"""
        if self.data is not None:
            return self.data[start_sample:end_sample]
        return np.array([])

    def file_exists(self) -> bool:
        """Проверка существования файла"""
        return os.path.exists(self.config.filename)

    def get_audio_bytes(self):
        """Получение аудио в байтах для Streamlit"""
        if os.path.exists(self.config.filename):
            with open(self.config.filename, 'rb') as f:
                return f.read()
        return None


# ==================== ФУНКЦИИ ДЛЯ LPC АНАЛИЗА ====================

def make_window(name: str, N: int):
    """Создание оконной функции"""
    name = name.lower()
    if name in ("hann", "hanning"):
        return signal.windows.hann(N, sym=False)
    if name in ("hamming",):
        return signal.windows.hamming(N, sym=False)
    if name in ("rect", "rectangular", "boxcar"):
        return np.ones(N)
    raise ValueError("Unknown window. Use: hann, hamming, rect")


def frame_signal(x: np.ndarray, frame_len: int, hop: int):
    """Разбиение сигнала на фреймы"""
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
    """OLA реконструкция из фреймов"""
    num, frame_len = frames.shape
    out_len = (num - 1) * hop + frame_len
    y = np.zeros(out_len, dtype=np.float64)
    for i in range(num):
        y[i * hop:i * hop + frame_len] += frames[i]
    return y


def levinson_durbin(r: np.ndarray, order: int):
    """
    Решение системы Тёплица для LPC методом Левинсона-Дурбина
    """
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
    """Смещенная автокорреляция r[0..order]"""
    x = np.asarray(x, dtype=np.float64)
    r_full = np.correlate(x, x, mode="full")
    mid = len(r_full) // 2
    r = r_full[mid:mid + order + 1]
    return r


def lpc_encode_frames(frames: np.ndarray, order: int):
    """
    LPC кодирование покадрово
    """
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
    """
    LPC синтез из коэффициентов
    """
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
    """
    Запуск LPC кодека на переданных аудиоданных
    """
    x = audio_data.copy()

    # Конвертация в float если нужно
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)

    # Нормализация
    if np.max(np.abs(x)) > 0:
        x = 0.9 * x / np.max(np.abs(x))

    # Ресемплинг если нужно
    fs = original_fs
    if target_fs is not None and fs != target_fs:
        x = signal.resample_poly(x, target_fs, fs)
        fs = target_fs

    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop = int(round(frame_len * (1.0 - overlap)))
    if hop <= 0:
        raise ValueError("Overlap too large -> hop <= 0")

    w = make_window(window_name, frame_len)

    # Фреймирование с окном
    frames = frame_signal(x, frame_len, hop)
    frames_w = frames * w[None, :]

    # Кодирование
    A, E, R_frames = lpc_encode_frames(frames_w, order)

    # Декодирование
    frames_hat = lpc_synthesize_frames(A, E, frame_len, excitation="noise")

    # OLA обратно во временную область
    y = overlap_add(frames_hat, hop)

    # Остаток во временную область
    r = overlap_add(R_frames, hop)

    # Обрезка до исходной длины
    y = y[:len(x)]
    r = r[:len(x)]

    # Временная ось
    t = np.arange(len(x)) / fs

    # Нормализация синтезированного сигнала
    if np.max(np.abs(y)) > 0:
        y = 0.9 * y / np.max(np.abs(y))

    # Сохранение результата во временный файл
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    write(temp_file.name, fs, y)

    return {
        'fs': fs,
        'x': x,
        'y': y,
        'r': r,
        'A': A,
        'E': E,
        'output_file': temp_file.name,
        'frame_len': frame_len,
        'hop': hop,
        'num_frames': A.shape[0]
    }


def plot_residual_vs_order(audio_data, original_fs, target_fs=8000, frame_ms=30, frame_number=10, max_order=20):
    """
    Анализ зависимости мощности остатка от порядка предсказания
    """
    x = audio_data.copy()

    # Конвертация в float если нужно
    if np.issubdtype(x.dtype, np.integer):
        maxv = np.iinfo(x.dtype).max
        x = x.astype(np.float64) / maxv
    else:
        x = x.astype(np.float64)

    # Ресемплинг если нужно
    fs = original_fs
    if fs != target_fs:
        x = signal.resample_poly(x, target_fs, fs)
        fs = target_fs

    # Параметры фреймирования
    frame_len = int(round(frame_ms * 1e-3 * fs))
    hop = int(round(frame_len * 0.5))

    # Получаем фреймы
    frames = frame_signal(x, frame_len, hop)
    w = make_window("hann", frame_len)
    frames_w = frames * w[None, :]

    # Проверка номера фрейма
    if frame_number >= len(frames_w):
        frame_number = len(frames_w) - 1

    frame_data = frames_w[frame_number]

    # Исследование разных порядков
    orders = range(1, max_order + 1)
    residual_powers = []

    for order in orders:
        r = autocorr(frame_data, order)
        if r[0] > 1e-12:
            _, e, _ = levinson_durbin(r, order)
            residual_powers.append(e)
        else:
            residual_powers.append(0)

    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(orders, residual_powers, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Порядок предсказания (p)')
    ax.set_ylabel('Мощность остатка предсказания')
    ax.set_title(f'Зависимость мощности остатка от порядка LPC\nФрейм #{frame_number}')
    ax.grid(True, alpha=0.3)

    return fig, orders, residual_powers


# ==================== ФУНКЦИИ ДЛЯ СОЗДАНИЯ ГРАФИКОВ ====================

def create_signal_plot(data, samplerate, title, color='blue'):
    """Создание графика сигнала"""
    duration = len(data) / samplerate
    time_arr = np.linspace(0, duration, len(data))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_arr, data, color=color, linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel('Время (с)')
    ax.set_ylabel('Амплитуда')

    return fig


def create_waveform_plots(t, x_in, x_out, x_res, title_prefix=""):
    """Создание осциллограмм"""
    figs = []

    # Входной сигнал
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t, x_in, 'b-', linewidth=0.5)
    ax1.set_title(f"{title_prefix}Входной сигнал (осциллограмма)")
    ax1.set_xlabel("Время, с")
    ax1.set_ylabel("Амплитуда")
    ax1.grid(True, alpha=0.3)
    figs.append(fig1)

    # Синтезированный сигнал
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t[:len(x_out)], x_out, 'r-', linewidth=0.5)
    ax2.set_title(f"{title_prefix}Синтезированный сигнал (осциллограмма)")
    ax2.set_xlabel("Время, с")
    ax2.set_ylabel("Амплитуда")
    ax2.grid(True, alpha=0.3)
    figs.append(fig2)

    # Остаток предсказания
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(t[:len(x_res)], x_res, 'g-', linewidth=0.5)
    ax3.set_title(f"{title_prefix}Остаток предсказания (осциллограмма)")
    ax3.set_xlabel("Время, с")
    ax3.set_ylabel("Амплитуда")
    ax3.grid(True, alpha=0.3)
    figs.append(fig3)

    return figs


def create_spectrogram(x, fs, title, nperseg=256, noverlap=192):
    """Создание спектрограммы"""
    f, tt, Sxx = signal.spectrogram(
        x, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        scaling="spectrum", mode="magnitude"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    pcm = ax.pcolormesh(tt, f, 20 * np.log10(Sxx + 1e-12), shading="auto", cmap='viridis')
    ax.set_ylim(0, fs / 2)
    ax.set_title(title)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, Гц")
    plt.colorbar(pcm, ax=ax, label="Уровень, dB")

    return fig

def generate_word_report(template_path, context):
    """
    Генерация отчета в формате Word на основе шаблона с метками
    Использует docxtpl для правильной замены меток и вставки изображений
    """
    try:
        # Загружаем шаблон
        doc = DocxTemplate(template_path)

        # Создаем копию контекста для обработки
        render_context = context.copy()

        # Конвертируем графики в InlineImage
        for key, value in context.items():
            if key.startswith('graph_') and value is not None:
                # Создаем буфер для графика
                buf = BytesIO()
                value.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)

                # Создаем InlineImage для вставки в документ (ширина 150 мм)
                render_context[key] = InlineImage(doc, buf, width=Mm(150))

                # Закрываем фигуру matplotlib для освобождения памяти
                plt.close(value)

        # Рендерим документ с контекстом
        doc.render(render_context)

        # Сохраняем в байтовый поток
        doc_bytes = BytesIO()
        doc.save(doc_bytes)
        doc_bytes.seek(0)

        return doc_bytes

    except Exception as e:
        st.error(f"Ошибка при генерации отчета: {str(e)}")
        raise e

# ==================== ИНТЕРФЕЙС STREAMLIT ====================

def run():
    st.sidebar.title("Навигация")
    pages = ["Запись аудио", "LPC анализ", "Отчет"]
    st.session_state.current_page = st.sidebar.radio("Перейти к:", pages)

    # Боковая панель с настройками
    st.sidebar.title("Настройки")
    st.session_state.variant = st.sidebar.number_input("Номер варианта", min_value=1, max_value=30,
                                                       value=st.session_state.variant)
    st.session_state.student_name = st.sidebar.text_input("ФИО студента", value=st.session_state.student_name)

    # Настройки LPC (доступны на всех страницах)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Параметры LPC анализа")
    frame_ms = st.sidebar.slider("Размер сегмента (мс)", min_value=10, max_value=50, value=30, step=5)
    overlap = st.sidebar.slider("Перекрытие", min_value=0.0, max_value=0.9, value=0.5, step=0.1)
    lpc_order = st.sidebar.slider("Порядок предсказания", min_value=2, max_value=20, value=10, step=1)

    # Основной контент в зависимости от выбранной страницы
    if st.session_state.current_page == "Запись аудио":
        show_recording_page()
    elif st.session_state.current_page == "LPC анализ":
        show_lpc_analysis_page(frame_ms, overlap, lpc_order)
    elif st.session_state.current_page == "Отчет":
        show_report_page(frame_ms, overlap, lpc_order)


def show_recording_page():
    """Страница записи аудио"""

    image = os.path.join("prikol.jpg")
    st.image(image)
    image1 = os.path.join("prikol1.jpg")
    st.image(image1)

    st.title("🎤 Запись аудиофайлов")
    st.markdown("Данная лабораторная работа предполагает работу со файлом .wav (получение его значений параметров),"
                "работу с LPC-кодеком. Для этого необходимо записать два звуковых сигнала в разделе 'Запись аудио', вы в нем уже находитесь. "
                "Далее необхоимо зайти в раздел 'LPC-анализ' и нажать кнопку запуска анализа. Вы получите результат в виде значений и графиков. "
                "Уже после переходите в раздел 'Отчет' - сгенерируйте отчет, нажав на кнопку.\n"
                "ВНИМАНИЕ: Вариантом, по сути, является лишь фамилия, остальные параметры можно не трогать."
                "Параметры можно подергать и получить, немного, но отличную работу.")

    st.markdown("""
        ### СДАЧА РАБОТЫ
        Сдача происходит очно. Надо будет распечатать отчет и прийти к нему на защиту. 
        Защита может происходить бригадами до 3х человек.
        """)

    st.markdown("""
           ### МЕТОДИЧКИ
           Методы в мудле в разделе лабораторная 1. Сюда загрузить мимо - сервак не бесконечный
           """)

    st.markdown("""
    ### Инструкция
    1. Нажмите кнопку "Начать запись"
    2. Произнесите свою фамилию. Запись длится 5 секунд
    3. Прослушайте запись и при необходимости перезапишите
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Запись 1 (8000 Гц)")

        if st.button("🎙️ Записать (8000 Гц)", key="record_8000"):
            with st.spinner("Идет запись... Говорите!"):
                progress_bar = st.progress(0)

                config = AudioConfig(8000, 5, "output.wav", 'blue')
                recorder = AudioRecorder(config)
                recorder.record(progress_bar).save().load()

                st.session_state.recordings['8000'] = recorder
                st.success("Запись завершена!")

        if '8000' in st.session_state.recordings:
            recorder = st.session_state.recordings['8000']
            info = recorder.get_info()

            st.write(f"**Информация:**")
            st.write(f"- Длительность: {info.get('duration', 0):.2f} с")
            st.write(f"- Размер: {info.get('size_kb', 0):.2f} КБ")
            st.write(f"- Глубина квантования: {info.get('bit_depth', 'N/A')} бит")

            # Аудиоплеер
            audio_bytes = recorder.get_audio_bytes()
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')

                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать (8000 Гц)",
                    data=audio_bytes,
                    file_name="output.wav",
                    mime="audio/wav"
                )

            # График
            fig = create_signal_plot(recorder.data, recorder.rate, 'Сигнал (8000 Гц)', 'blue')
            st.pyplot(fig)
            plt.close(fig)

    with col2:
        st.subheader("Запись 2 (11025 Гц)")

        if st.button("🎙️ Записать (11025 Гц)", key="record_11025"):
            with st.spinner("Идет запись... Говорите!"):
                progress_bar = st.progress(0)

                config = AudioConfig(11025, 5, "output11025.wav", 'green')
                recorder = AudioRecorder(config)
                recorder.record(progress_bar).save().load()

                st.session_state.recordings['11025'] = recorder
                st.success("Запись завершена!")

        if '11025' in st.session_state.recordings:
            recorder = st.session_state.recordings['11025']
            info = recorder.get_info()

            st.write(f"**Информация:**")
            st.write(f"- Длительность: {info.get('duration', 0):.2f} с")
            st.write(f"- Размер: {info.get('size_kb', 0):.2f} КБ")
            st.write(f"- Глубина квантования: {info.get('bit_depth', 'N/A')} бит")

            # Аудиоплеер
            audio_bytes = recorder.get_audio_bytes()
            if audio_bytes:
                st.audio(audio_bytes, format='audio/wav')

                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать (11025 Гц)",
                    data=audio_bytes,
                    file_name="output11025.wav",
                    mime="audio/wav"
                )

            # График
            fig = create_signal_plot(recorder.data, recorder.rate, 'Сигнал (11025 Гц)', 'green')
            st.pyplot(fig)
            plt.close(fig)

    # Фрагмент сигнала
    st.markdown("---")
    st.subheader("Фрагмент сигнала")

    if '8000' in st.session_state.recordings:
        recorder = st.session_state.recordings['8000']

        # Расчет границ фрагмента
        start_sample = 5000
        end_sample = 1000 * st.session_state.variant

        # Проверка на равенство
        if start_sample >= end_sample:
            start_sample = 4000
            end_sample = 5000

        # Получение фрагмента
        fragment = recorder.get_fragment(start_sample, end_sample)

        st.write(f"**Параметры фрагмента:**")
        st.write(f"- Начальный сэмпл: {start_sample}")
        st.write(f"- Конечный сэмпл: {end_sample}")
        st.write(f"- Количество сэмплов: {len(fragment)}")
        st.write(f"- Длительность: {len(fragment) / recorder.rate:.3f} с")

        # График фрагмента
        fig = create_signal_plot(fragment, recorder.rate, f'Фрагмент сигнала (сэмплы {start_sample}-{end_sample})',
                                 'red')
        st.pyplot(fig)
        plt.close(fig)


def show_lpc_analysis_page(frame_ms, overlap, lpc_order):
    """Страница LPC анализа"""
    st.title("🔬 LPC анализ и синтез речи")

    # Проверка наличия записей
    if '11025' not in st.session_state.recordings:
        st.warning("Сначала выполните запись аудиофайлов на странице 'Запись аудио'")
        return

    recorder = st.session_state.recordings['11025']

    st.write(f"**Анализируемый файл:** {recorder.config.filename}")
    st.write(f"**Частота дискретизации:** {recorder.rate} Гц")
    st.write(f"**Параметры анализа:**")
    st.write(f"- Размер сегмента: {frame_ms} мс")
    st.write(f"- Перекрытие: {overlap:.1%}")
    st.write(f"- Порядок предсказания: {lpc_order}")

    if st.button("🚀 Запустить LPC анализ", type="primary"):
        with st.spinner("Выполняется LPC анализ..."):
            # Запуск LPC кодека
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

            # Сохранение параметров для отчета
            st.session_state.lpc_params = {
                'frame_ms': frame_ms,
                'overlap': overlap,
                'lpc_order': lpc_order
            }

            st.success("Анализ завершен!")

            # Отображение результатов
            show_lpc_results(results)

    # Если результаты уже есть в сессии, показываем их
    elif st.session_state.lpc_results is not None:
        show_lpc_results(st.session_state.lpc_results)

    # Анализ зависимости от порядка
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

        # Сохраняем график для отчета
        st.session_state.residual_plot = fig

        # Таблица с результатами
        df = pd.DataFrame({
            'Порядок': orders,
            'Мощность остатка': powers,
            'Относительное уменьшение': [1 - p / powers[0] if powers[0] > 0 else 0 for p in powers]
        })
        st.dataframe(df)


def show_lpc_results(results):
    """Отображение результатов LPC анализа"""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Частота дискретизации", f"{results['fs']} Гц")
    with col2:
        st.metric("Количество фреймов", results['num_frames'])
    with col3:
        st.metric("Длина фрейма", f"{results['frame_len']} сэмплов")

    # Осциллограммы
    st.subheader("Осциллограммы")
    t = np.arange(len(results['x'])) / results['fs']
    figs = create_waveform_plots(t, results['x'], results['y'], results['r'])

    tab1, tab2, tab3 = st.tabs(["Исходный", "Синтезированный", "Остаток"])

    with tab1:
        st.pyplot(figs[0])
    with tab2:
        st.pyplot(figs[1])
    with tab3:
        st.pyplot(figs[2])

    # Сохраняем для отчета
    st.session_state.osc_plots = figs

    # Спектрограммы
    st.subheader("Спектрограммы")

    spec_figs = []
    spec_figs.append(create_spectrogram(results['x'], results['fs'], 'Спектрограмма исходного сигнала'))
    spec_figs.append(create_spectrogram(results['y'], results['fs'], 'Спектрограмма синтезированного сигнала'))
    spec_figs.append(create_spectrogram(results['r'], results['fs'], 'Спектрограмма остатка'))

    tab1, tab2, tab3 = st.tabs(["Исходный", "Синтезированный", "Остаток"])

    with tab1:
        st.pyplot(spec_figs[0])
    with tab2:
        st.pyplot(spec_figs[1])
    with tab3:
        st.pyplot(spec_figs[2])

    # Сохраняем для отчета
    st.session_state.spect_plots = spec_figs

    # Синтезированное аудио
    st.subheader("Синтезированный сигнал")

    if os.path.exists(results['output_file']):
        with open(results['output_file'], 'rb') as f:
            audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/wav')

            st.download_button(
                label="📥 Скачать синтезированный сигнал",
                data=audio_bytes,
                file_name=f"synthesized_{st.session_state.lpc_params['lpc_order']}.wav",
                mime="audio/wav"
            )

    # Статистика
    st.subheader("Статистика")

    total_params = results['A'].size + results['E'].size
    compression_ratio = len(results['x']) / total_params

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Параметров модели (A+E):** {total_params}")
        st.write(f"**Степень сжатия:** {compression_ratio:.2f} сэмплов/параметр")

    # Сохраняем для отчета
    st.session_state.compression_ratio = compression_ratio
    st.session_state.total_params = total_params


def show_report_page(frame_ms, overlap, lpc_order):
    """Страница генерации отчета"""
    st.title("📄 Генерация отчета")

    # Проверка наличия всех необходимых данных
    required_data = ['8000' in st.session_state.recordings,
                     '11025' in st.session_state.recordings,
                     st.session_state.lpc_results is not None]

    if not all(required_data):
        st.warning("Для генерации отчета необходимо выполнить запись обоих файлов и LPC анализ")
        return

    # Проверка наличия шаблона
    template_path = "pattern_rad_lab1.docx"
    if not os.path.exists(template_path):
        st.error(f"❌ Файл шаблона '{template_path}' не найден в текущей директории!")

        # Показываем содержимое директории для отладки
        with st.expander("📁 Содержимое текущей директории"):
            files = os.listdir('.')
            for file in files:
                st.write(f"- {file}")
        return

    st.success("✅ Шаблон найден")

    # Кнопка генерации отчета
    if st.button("📥 Сгенерировать отчет Word", type="primary"):
        with st.spinner("🔄 Генерация отчета..."):
            try:
                # Получаем данные
                recorder_8000 = st.session_state.recordings['8000']
                recorder_11025 = st.session_state.recordings['11025']
                lpc_results = st.session_state.lpc_results

                # Получаем информацию о записях
                info_8000 = recorder_8000.get_info()
                info_11025 = recorder_11025.get_info()

                # Параметры фрагмента
                start_sample = 5000
                end_sample = 1000 * st.session_state.variant

                # Проверка на корректность границ
                if start_sample >= end_sample:
                    start_sample = 4000
                    end_sample = 5000

                fragment = recorder_8000.get_fragment(start_sample, end_sample)

                # Создание графиков для отчета
                st.info("🔄 Подготавливаю графики...")

                graph_8000 = create_signal_plot(recorder_8000.data, recorder_8000.rate, 'Полный сигнал (8000 Гц)',
                                                'blue')
                graph_frag = create_signal_plot(fragment, recorder_8000.rate, f'Фрагмент сигнала', 'red')
                graph_11025 = create_signal_plot(recorder_11025.data, recorder_11025.rate, 'Полный сигнал (11025 Гц)',
                                                 'green')

                # Расчет параметров для LPC части
                total_params = lpc_results['A'].size + lpc_results['E'].size
                compression_ratio = len(lpc_results['x']) / total_params if total_params > 0 else 0

                # Подготовка контекста для шаблона
                context = {
                    # Информация о студенте
                    'name': st.session_state.student_name,

                    # Данные для файла 8000 Гц
                    'size_kb_8000': f"{info_8000.get('size_kb', 0):.2f}",
                    'bit_depth_8000': str(info_8000.get('bit_depth', 'N/A')),
                    'graph_8000': graph_8000,

                    # Данные для фрагмента
                    'start_sample': str(start_sample),
                    'end_sample': str(end_sample),
                    'len_fragment': str(len(fragment)),
                    'time_frag': f"{len(fragment) / recorder_8000.rate:.3f}",
                    'graph_frag': graph_frag,

                    # Данные для файла 11025 Гц
                    'size_kb_11025': f"{info_11025.get('size_kb', 0):.2f}",
                    'bit_depth_11025': str(info_11025.get('bit_depth', 'N/A')),
                    'graph_11025': graph_11025,

                    # LPC параметры
                    'fs_lpc': lpc_results['fs'],
                    'frame_ms': str(frame_ms),
                    'frame_sem': str(lpc_results['frame_len']),
                    'overlap': f"{overlap:.1%}",
                    'full_frame': str(lpc_results['num_frames']),
                    'full_sem': str(len(lpc_results['x'])),
                    'lpc_order': str(lpc_order),
                    'count_order': str(total_params),
                    'coeff': f"{compression_ratio:.2f}"
                }

                # Добавляем графики LPC, если они есть
                if hasattr(st.session_state, 'osc_plots') and st.session_state.osc_plots and len(
                        st.session_state.osc_plots) >= 3:
                    context['graph_osc_orig'] = st.session_state.osc_plots[0]
                    context['graph_osc_sint'] = st.session_state.osc_plots[1]
                    context['graph_osc_frag'] = st.session_state.osc_plots[2]
                    st.success("✅ Осциллограммы добавлены")

                if hasattr(st.session_state, 'spect_plots') and st.session_state.spect_plots and len(
                        st.session_state.spect_plots) >= 3:
                    context['graph_spect_orig'] = st.session_state.spect_plots[0]
                    context['graph_spect_sint'] = st.session_state.spect_plots[1]
                    context['graph_spect_frag'] = st.session_state.spect_plots[2]
                    st.success("✅ Спектрограммы добавлены")

                if hasattr(st.session_state, 'residual_plot') and st.session_state.residual_plot:
                    context['graph_zavis_lpc'] = st.session_state.residual_plot
                    st.success("✅ График зависимости добавлен")

                # Проверяем, все ли метки будут заменены (просто информационно)
                st.info("🔍 Проверка наличия данных для меток...")

                # Список ожидаемых меток из твоего задания
                expected_placeholders = [
                    'name', 'size_kb_8000', 'bit_depth_8000', 'graph_8000',
                    'start_sample', 'end_sample', 'len_fragment', 'time_frag', 'graph_frag',
                    'size_kb_11025', 'bit_depth_11025', 'graph_11025',
                    'graph_osc_orig', 'graph_osc_sint', 'graph_osc_frag',
                    'graph_spect_orig', 'graph_spect_sint', 'graph_spect_frag',
                    'fs_lps', 'frame_ms', 'frame_sem', 'overlap', 'full_frame', 'full_sem',
                    'lpc_order', 'count_order', 'coeff', 'graph_zavis_lpc'
                ]

                missing = [p for p in expected_placeholders if p not in context]
                if missing:
                    st.warning(f"⚠️ Нет данных для меток: {', '.join(missing)}")
                else:
                    st.success("✅ Все данные для отчета подготовлены")

                # Генерация отчета по шаблону
                st.info("📄 Формирую отчет...")
                doc_bytes = generate_word_report(template_path, context)

                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать отчет",
                    data=doc_bytes,
                    file_name=f"LPC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

                # Закрываем все фигуры для освобождения памяти
                plt.close('all')

                st.success("✅ Отчет успешно сгенерирован!")

            except Exception as e:
                st.error(f"❌ Ошибка при генерации отчета: {str(e)}")
                import traceback
                with st.expander("Детали ошибки"):
                    st.code(traceback.format_exc())
