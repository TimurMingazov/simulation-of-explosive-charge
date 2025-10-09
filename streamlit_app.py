import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from docxtpl import DocxTemplate
import tempfile
import os

# ---------------------------
# Константы и дефолтные цели
# ---------------------------
AIR_DENSITY = 1.225  # кг/м³
SOUND_SPEED = 340.0  # м/с
ADIABATIC_INDEX = 1.4
B_COEFF = (ADIABATIC_INDEX + 1) / (2 * ADIABATIC_INDEX)
C_COEFF = (ADIABATIC_INDEX - 1) / ADIABATIC_INDEX
INITIAL_TEMPERATURE = 293.0  # К
ATMOSPHERIC_PRESSURE = 0.101  # МПа

DEFAULT_TARGETS = {
    "Кирпичные малоэтажные здания": {
        "Полное разрушение": 40.0,
        "Сильные повреждения": 30.0,
        "Средние повреждения": 20.0,
        "Слабые повреждения": 12.0
    },
    "Промышленные здания (металл, ж/б каркас)": {
        "Полное разрушение": 80.0,
        "Сильные повреждения": 60.0,
        "Средние повреждения": 40.0,
        "Слабые повреждения": 25.0
    },
    "Подземные сети коммунального хозяйства": {
        "Полное разрушение": 1500.0,
        "Сильные повреждения": 1250.0,
        "Средние повреждения": 800.0,
        "Слабые повреждения": 400.0
    },
    "Надводные корабли": {
        "Полное разрушение": 500.0,
        "Сильные повреждения": 120.0,
        "Средние повреждения": 42.0,
        "Слабые повреждения": 18.0
    },
    "Самолеты на аэродроме": {
        "Полное разрушение": 42.0,
        "Сильные повреждения": 17.0,
        "Средние повреждения": 10.0,
        "Слабые повреждения": 5.0
    }
}

# ---------------------------
# БАЗА ДАННЫХ ВЗРЫВЧАТЫХ ВЕЩЕСТВ
# ---------------------------
EXPLOSIVES_DB = {
    "Тротил (TNT)": {
        "tnt_equivalent": 1.0,
        "density": 1650,
        "heat_of_explosion": 4180,  # кДж/кг
        "detonation_velocity": 6900,  # м/с
        "color": "#ff0000"
    },
    "Гексоген (RDX)": {
        "tnt_equivalent": 1.3,
        "density": 1780,
        "heat_of_explosion": 5430,
        "detonation_velocity": 8750,
        "color": "#00ff00"
    },
    "Пентолит 50/50 (ТЭН/ТНТ)": {
        "tnt_equivalent": 1.13,
        "density": 1700,
        "heat_of_explosion": 4720,
        "detonation_velocity": 7460,
        "color": "#0000ff"
    },
    "ТЭН": {
        "tnt_equivalent": 1.33,
        "density": 1770,
        "heat_of_explosion": 5560,
        "detonation_velocity": 8300,
        "color": "#ff00ff"
    },
    "Аммонийная селитра": {
        "tnt_equivalent": 0.34,
        "density": 1725,
        "heat_of_explosion": 1420,
        "detonation_velocity": 2700,
        "color": "#ffff00"
    },
    "Гликольдинитрат": {
        "tnt_equivalent": 1.57,
        "density": 1760,
        "heat_of_explosion": 6560,
        "detonation_velocity": 9100,
        "color": "#000000"
    },
}
DEFAULT_EXPLOSIVE = "Тротил (TNT)"


# ---------------------------
# Базовые функции расчёта
# ---------------------------

def calculate_scaled_radius(mass, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    return 0.05 * (equivalent_mass ** (1 / 3))


def calculate_overpressure(mass, distance, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    distance = max(distance, 1e-6)
    return (1.4 * equivalent_mass / (distance ** 3)) + (0.43 * (equivalent_mass ** (2 / 3)) / (distance ** 2)) + (
            0.11 * (equivalent_mass ** (1 / 3)) / distance)


def calculate_compression_duration(mass, distance, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    distance = max(distance, 1e-6)
    return 1.5e-3 * (equivalent_mass ** (1 / 6)) * np.sqrt(distance)


def calculate_rarefaction_pressure(mass, distance, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    distance = max(distance, 1e-6)
    return -0.03 * (equivalent_mass ** (1 / 3)) / distance


def calculate_rarefaction_duration(mass, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    return 0.013 * (equivalent_mass ** (1 / 3))


def calculate_gas_velocity(overpressure):
    p_rel = overpressure
    return SOUND_SPEED * p_rel / (ADIABATIC_INDEX * np.sqrt(1 + p_rel * B_COEFF) + 1e-12)


def calculate_shock_wave_velocity(overpressure):
    p_rel = overpressure
    return SOUND_SPEED * np.sqrt(1 + p_rel * B_COEFF)


def calculate_gas_density(overpressure):
    p_rel = overpressure
    return AIR_DENSITY * (1 + p_rel * B_COEFF) / (1 + p_rel * C_COEFF)


def calculate_dynamic_pressure_from_state(overpressure):
    rho = calculate_gas_density(overpressure)
    v = calculate_gas_velocity(overpressure)
    return 0.5 * rho * v ** 2


def calculate_specific_impulse(overpressure, distance, mass, tnt_equivalent=1.0):
    tau = calculate_compression_duration(mass, distance, tnt_equivalent)
    return overpressure * 1e6 * tau


def pressure_time_history(mass, distance, t_array, tnt_equivalent=1.0):
    equivalent_mass = mass * tnt_equivalent
    p_peak = calculate_overpressure(mass, distance, tnt_equivalent)
    tau_p = calculate_compression_duration(mass, distance, tnt_equivalent)
    p_neg = calculate_rarefaction_pressure(mass, distance, tnt_equivalent)
    tau_n = calculate_rarefaction_duration(mass, tnt_equivalent)

    t0 = 0.1 * tau_p
    t1 = t0 + 0.8 * tau_p
    t2 = t1 + tau_n

    p_t = np.zeros_like(t_array)
    for i, t in enumerate(t_array):
        if t < 0:
            p_t[i] = 0.0
        elif t <= t0:
            p_t[i] = (t / t0) * p_peak
        elif t <= t1:
            p_t[i] = p_peak
        elif t <= t2:
            p_t[i] = p_peak + (p_neg - p_peak) * ((t - t1) / (t2 - t1))
        else:
            p_t[i] = p_neg * np.exp(- (t - t2) / (0.1 * t2 + 1e-6))
    return p_t


# ---------------------------
# Минимальная масса
# ---------------------------

def find_minimum_mass_for_pressure(target_distance, pressure_resistance_kpa, tnt_equivalent=1.0, mass_lower=0.01,
                                   mass_upper=5000.0, tol=1e-3):
    pr_mpa = pressure_resistance_kpa / 1000.0
    if calculate_overpressure(mass_upper, target_distance, tnt_equivalent) < pr_mpa:
        return None
    lo = mass_lower
    hi = mass_upper
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if calculate_overpressure(mid, target_distance, tnt_equivalent) >= pr_mpa:
            hi = mid
        else:
            lo = mid
    return round(hi, 3)


# ---------------------------
# Генерация графиков для отчета
# ---------------------------

def create_report_graphs(min_mass, target_distance, selected_value, distance_range, tnt_equivalent=1.0,
                         explosive_name="ВВ"):
    """Создает все графики для отчета"""
    figures = {}

    # 1) График радиуса поражения для минимальной массы
    fig1, ax = plt.subplots(figsize=(10, 6))
    overpressures = np.array([calculate_overpressure(min_mass, r, tnt_equivalent) for r in distance_range]) * 1000.0
    ax.plot(distance_range, overpressures, linewidth=2, color='blue')
    ax.axhline(y=selected_value, color='r', linestyle='--', label=f'Требуемое давление: {selected_value} кПа')
    ax.axvline(x=target_distance, color='g', linestyle='--', label=f'Расстояние до цели: {target_distance} м')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Δp, кПа')
    ax.set_title(f'Радиус поражения (масса ВВ = {min_mass:.3f} кг)')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures['damage_radius_graph_one'] = fig1

    # 2) Сравнительный график для масс ±20% от минимальной
    fig2, ax = plt.subplots(figsize=(10, 6))
    masses_comparison = [
        min_mass * 0.8,  # -20%
        min_mass,  # минимальная
        min_mass * 1.2  # +20%
    ]

    for m in masses_comparison:
        overpressures = np.array([calculate_overpressure(m, r, tnt_equivalent) for r in distance_range]) * 1000.0
        label = f'{m:.3f} кг'
        if m == min_mass:
            label += ' (мин. масса)'
        ax.plot(distance_range, overpressures, label=label, linewidth=2)

    ax.axhline(y=selected_value, color='r', linestyle='--', label=f'Требуемое давление: {selected_value} кPa')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Δp, кПа')
    ax.set_title('Радиусы поражения для различных масс')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures['damage_radius_graph_more'] = fig2

    # 3) Удельный импульс
    fig3, ax = plt.subplots(figsize=(10, 6))
    overpressures = np.array([calculate_overpressure(min_mass, r, tnt_equivalent) for r in distance_range])
    specific_impulses = np.array(
        [calculate_specific_impulse(p, r, min_mass, tnt_equivalent) for p, r in zip(overpressures, distance_range)])
    ax.plot(distance_range, specific_impulses, linewidth=2, color='purple')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Удельный импульс, Па·с')
    ax.set_title(f'Зависимость удельного импульса от расстояния')
    ax.grid(True)
    plt.tight_layout()
    figures['specific_impulse_graph'] = fig3

    # 4) Скоростной напор
    fig4, ax = plt.subplots(figsize=(10, 6))
    overpressures = np.array([calculate_overpressure(min_mass, r, tnt_equivalent) for r in distance_range])
    p_dyn = np.array([calculate_dynamic_pressure_from_state(p) for p in overpressures])
    ax.plot(distance_range, p_dyn / 1000.0, linewidth=2, color='orange')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('p_dyn, кПа')
    ax.set_title(f'Зависимость скоростного напора от расстояния')
    ax.grid(True)
    plt.tight_layout()
    figures['highspeed_pressure_graph'] = fig4

    # 5) Скорость распространения УВ
    fig5, ax = plt.subplots(figsize=(10, 6))
    overpressures = np.array([calculate_overpressure(min_mass, r, tnt_equivalent) for r in distance_range])
    D = np.array([calculate_shock_wave_velocity(p) for p in overpressures])
    ax.plot(distance_range, D, linewidth=2, color='brown')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Скорость ударной волны, м/с')
    ax.set_title(f'Скорость распространения УВ')
    ax.grid(True)
    plt.tight_layout()
    figures['graph_wave_spreading_rate'] = fig5

    # 6) Длительность фазы сжатия
    fig6, ax = plt.subplots(figsize=(10, 6))
    taus = np.array([calculate_compression_duration(min_mass, r, tnt_equivalent) for r in distance_range])
    ax.plot(distance_range, taus, linewidth=2, color='green')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('τ_+, с')
    ax.set_title(f'Длительность фазы сжатия')
    ax.grid(True)
    plt.tight_layout()
    figures['phase_duration_graph'] = fig6

    # 7) Изменение давления от времени
    fig7, ax = plt.subplots(figsize=(10, 6))
    max_tau = calculate_compression_duration(min_mass, target_distance,
                                             tnt_equivalent) + calculate_rarefaction_duration(min_mass, tnt_equivalent)
    t_stop = max(3 * max_tau, 0.05)
    t = np.linspace(0, t_stop, 1000)
    p_t = pressure_time_history(min_mass, target_distance, t, tnt_equivalent) * 1000.0
    ax.plot(t, p_t, linewidth=2, color='red')
    ax.set_xlabel('Время, с')
    ax.set_ylabel('Давление, кПа')
    ax.set_title(f'Изменение давления от времени (R={target_distance:.2f} м)')
    ax.grid(True)
    plt.tight_layout()
    figures['pressure_change_schedule'] = fig7

    return figures

def save_plot_to_buffer(fig):
    """Сохраняет график в буфер памяти"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)  # Закрываем фигуру чтобы освободить память
    return buf


# ---------------------------
# Основное приложение Streamlit
# ---------------------------

def main():
    st.set_page_config(page_title="Симулятор взрывных воздействий", layout="wide")

    st.title("💥 Симулятор взрывных воздействий")
    st.markdown("Расчет минимальной массы ВВ и генерация отчета")

    # Сайдбар с настройками
    # Сайдбар с настройками
    with st.sidebar:
        st.header("🎯 Параметры цели")

        # Выбор цели
        selected_target_name = st.selectbox(
            "Выберите цель",
            options=list(DEFAULT_TARGETS.keys()),
            index=1  # По умолчанию промышленные здания
        )

        # Выбор степени разрушения
        damage_levels = list(DEFAULT_TARGETS[selected_target_name].keys())
        selected_damage_level = st.selectbox(
            "Степень разрушения",
            options=damage_levels,
            index=0
        )

        # Получаем стандартное значение
        selected_value = DEFAULT_TARGETS[selected_target_name][selected_damage_level]

        # Настройка стойкости цели
        custom_resistance = st.number_input(
            "Точное значение стойкости цели (кПа)",
            value=float(selected_value),
            min_value=1.0,
            max_value=1500.0,
            step=1.0
        )

        # Расстояние до цели
        target_distance = st.slider(
            "Расстояние до цели (м)",
            min_value=1.0,
            max_value=10.0,
            value=10.0,
            step=0.1
        )

        st.header("💣 Параметры ВВ")

        # Выбор типа ВВ
        explosive_names = list(EXPLOSIVES_DB.keys())
        selected_explosive_name = st.selectbox(
            "Выберите тип ВВ:",
            options=explosive_names,
            index=explosive_names.index(DEFAULT_EXPLOSIVE)
        )
        selected_explosive_data = EXPLOSIVES_DB[selected_explosive_name]

        # Информация о ВВ
        st.info(f"""
        **{selected_explosive_name}**
        - Коэффициент приведения к тротилу: **{selected_explosive_data['tnt_equivalent']}**
        - Теплота взрыва: **{selected_explosive_data['heat_of_explosion']}** кДж/кг
        - Скорость детонации: **{selected_explosive_data['detonation_velocity']}** м/с
        - Плотность: **{selected_explosive_data['density']}** кг/м³
        """)

        st.header("📊 Настройки графиков")

        # Диапазон расстояний для графиков
        st.subheader("Диапазон расстояний на графиках")
        dist_min = st.number_input(
            "Минимальное расстояние (м)",
            min_value=0.1,
            max_value=100.0,
            value=1.0,
            step=0.1
        )

        dist_max = st.number_input(
            "Максимальное расстояние (м)",
            min_value=1.0,
            max_value=500.0,
            value=30.0,
            step=1.0
        )

        dist_points = st.slider(
            "Количество точек на графике",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )

        # Создаем диапазон расстояний
        distance_range = np.linspace(dist_min, dist_max, dist_points)

        # Данные студента
        st.header("👤 Данные студента")
        student_name = st.text_input("ФИО студента", value="Иванов И.И.")
        variant_number = st.number_input("Номер варианта", min_value=1, max_value=100, value=1)

        # Кнопка расчета
        calculate_btn = st.button("🚀 Рассчитать минимальную массу", type="primary")
    # Основная область
    # В основной области после расчета минимальной массы:
    if calculate_btn:
        # Расчет минимальной массы
        min_mass = find_minimum_mass_for_pressure(
            target_distance,
            custom_resistance,
            selected_explosive_data['tnt_equivalent']
        )

        if min_mass is None:
            st.error("❌ Не удалось достичь требуемого давления с разумными массами ВВ")
            return

        st.success(f"✅ Расчет выполнен! Минимальная масса: {min_mass:.3f} кг")

        # Показываем информацию о диапазоне
        st.info(f"📊 Диапазон расстояний на графиках: от {dist_min} м до {dist_max} м ({dist_points} точек)")

        # Показываем результаты
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Минимальная масса ВВ", f"{min_mass:.3f} кг")
        with col2:
            st.metric("Эквивалент TNT", f"{min_mass * selected_explosive_data['tnt_equivalent']:.3f} кг")
        with col3:
            st.metric("Требуемое давление", f"{custom_resistance} кПа")

            # Создаем графики для отчета с пользовательским диапазоном
        with st.spinner("Создание графиков для отчета..."):
            figures = create_report_graphs(
                min_mass,
                target_distance,
                custom_resistance,
                distance_range,
                selected_explosive_data['tnt_equivalent'],
                selected_explosive_name
            )


            # Сохраняем графики в буферы
            graph_buffers = {}
            for name, fig in figures.items():
                graph_buffers[name] = save_plot_to_buffer(fig)

        # Показываем превью графиков
        st.subheader("📊 Графики для отчета")

        cols = st.columns(2)
        graph_names = list(figures.keys())

        for i, graph_name in enumerate(graph_names):
            with cols[i % 2]:
                st.image(graph_buffers[graph_name], caption=graph_name, use_column_width=True)

        # Генерация отчета - ПРАВИЛЬНАЯ версия для Streamlit
        st.subheader("📄 Генерация отчета")

        # Создаем отчет в памяти
        try:
            # Проверяем шаблон
            if not os.path.exists("Shablon.docx"):
                st.error("❌ Файл шаблона 'Shablon.docx' не найден")
                return

            st.info("🔄 Подготавливаю данные для отчета...")

            # Подготавливаем данные - ОСОБЕННОСТЬ: для изображений используем специальный объект
            from docxtpl import InlineImage
            from docx.shared import Mm

            # Загружаем шаблон ДО создания контекста с изображениями
            doc = DocxTemplate("Shablon.docx")

            # Создаем контекст с InlineImage для графиков
            context = {
                # Текстовые данные
                'name': student_name,
                'var': variant_number,
                'VV': selected_explosive_name,
                'qv': selected_explosive_data['heat_of_explosion'],
                'd': selected_explosive_data['detonation_velocity'],
                'ro': selected_explosive_data['density'],
                'target': selected_target_name,
                'def': custom_resistance,
                'dist_target': target_distance,
                'degree_dest': selected_damage_level,
                'required_pressure': custom_resistance,
                'required_weight': min_mass,

                # Графики как InlineImage
                'damage_radius_graph_one': InlineImage(doc, graph_buffers['damage_radius_graph_one'], width=Mm(150)),
                'damage_radius_graph_more': InlineImage(doc, graph_buffers['damage_radius_graph_more'], width=Mm(150)),
                'specific_impulse_graph': InlineImage(doc, graph_buffers['specific_impulse_graph'], width=Mm(150)),
                'highspeed_pressure_graph': InlineImage(doc, graph_buffers['highspeed_pressure_graph'], width=Mm(150)),
                'graph_wave_spreading_rate': InlineImage(doc, graph_buffers['graph_wave_spreading_rate'],
                                                         width=Mm(150)),
                'phase_duration_graph': InlineImage(doc, graph_buffers['phase_duration_graph'], width=Mm(150)),
                'pressure_change_schedule': InlineImage(doc, graph_buffers['pressure_change_schedule'], width=Mm(150)),
            }

            st.info("🔄 Генерирую отчет...")

            # Рендерим документ
            doc.render(context)

            # Сохраняем в буфер
            output_buffer = BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)

            st.success("✅ Отчет готов!")

            # Кнопка скачивания
            st.download_button(
                label="📥 Скачать отчет DOCX",
                data=output_buffer,
                file_name=f"отчет_лабораторная_{student_name}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                type="primary"
            )

        except Exception as e:
            st.error(f"❌ Ошибка при генерации отчета: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
