import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

# ---------------------------
# Константы и дефолтные цели
# ---------------------------
DEFAULT_MASSES = [2.0, 3.8, 5.0]
MIN_MASS_ALLOWED = 0.01
MAX_MASSES = 5

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


# Добавим функцию для получения степеней разрушения
def get_damage_levels(target_name):
    """Возвращает доступные степени разрушения для цели"""
    return list(DEFAULT_TARGETS[target_name].keys())

# ---------------------------
# БАЗА ДАННЫХ ВЗРЫВЧАТЫХ ВЕЩЕСТВ
# ---------------------------
EXPLOSIVES_DB = {
    "Тротил (TNT)": {
        "tnt_equivalent": 1.0,
        "density": 1650,
        "color": "#ff0000"
    },
    "Гексоген (RDX)": {
        "tnt_equivalent": 1.3,
        "density": 1780,
        "color": "#00ff00"
    },
    "Пентолит 50/50 (ТЭН/ТНТ)": {
        "tnt_equivalent": 1.13,
        "density": 1700,
        "color": "#0000ff"
    },
    "ТЭН": {
        "tnt_equivalent": 1.33,
        "density": 1770,
        "color": "#ff00ff"
    },
    "Аммонийная селитра": {
        "tnt_equivalent": 0.34,
        "density": 1725,
        "color": "#ffff00"
    },
    "Гликольдинитрат": {
        "tnt_equivalent": 1.57,
        "density": 1760,
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


def calculate_scaled_distance(mass, distance, tnt_equivalent=1.0):
    R_s = calculate_scaled_radius(mass, tnt_equivalent)
    return distance / R_s


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


# ---------------------------
# Временные сигналы
# ---------------------------

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


def dynamic_pressure_time_history(mass, distance, t_array, tnt_equivalent=1.0):
    p_t = pressure_time_history(mass, distance, t_array, tnt_equivalent)
    return np.array([calculate_dynamic_pressure_from_state(p) for p in p_t])


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
# Парсеры ввода
# ---------------------------

def parse_mass_list(inp):
    if not inp:
        return DEFAULT_MASSES
    try:
        parts = [p.strip() for p in inp.split(',') if p.strip() != '']
        masses = [float(p) for p in parts]
        if len(masses) > MAX_MASSES:
            st.warning(f"Введено больше {MAX_MASSES} масс — будут использованы первые {MAX_MASSES}.")
            masses = masses[:MAX_MASSES]
        masses = [max(m, MIN_MASS_ALLOWED) for m in masses]
        return masses
    except Exception:
        st.error("Ошибка парсинга масс, использую значения по умолчанию")
        return DEFAULT_MASSES


def parse_distance_input(inp):
    if not inp:
        return np.linspace(1.0, 10.0, 100)
    inp = inp.strip()
    for sep in (':', '-'):
        if sep in inp:
            parts = inp.split(sep)
            try:
                if len(parts) == 2:
                    a = float(parts[0]);
                    b = float(parts[1])
                    if abs(a - b) < 1e-6:
                        return np.linspace(max(0.1, a * 0.9), a * 1.1 if a > 0 else 1.0, 100)
                    return np.linspace(min(a, b), max(a, b), 100)
                elif len(parts) == 3:
                    a = float(parts[0]);
                    b = float(parts[1]);
                    n = int(parts[2])
                    if abs(a - b) < 1e-6:
                        return np.linspace(max(0.1, a * 0.9), a * 1.1 if a > 0 else 1.0, max(2, n))
                    return np.linspace(min(a, b), max(a, b), max(2, n))
            except:
                continue
    try:
        val = float(inp)
        return np.linspace(max(0.1, val * 0.9), val * 1.1 if val > 0 else 1.0, 100)
    except:
        st.error("Неверный формат расстояния — использую диапазон 1..10 м")
        return np.linspace(1.0, 10.0, 100)


# ---------------------------
# Визуализация
# ---------------------------

def create_plots(masses, distance_range, selected_target_name, selected_target_value, target_distance,
                 tnt_equivalent=1.0, explosive_name="ВВ"):
    distance_range = np.array(distance_range)

    figures = []  # Список для всех графиков

    # 1) Δp - отдельный график
    fig1, ax = plt.subplots(figsize=(10, 6))
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r, tnt_equivalent) for r in distance_range]) * 1000.0
        ax.plot(distance_range, overpressures, label=f'{m} кг', linewidth=2)
    ax.axhline(y=selected_target_value, color='r', linestyle='--', label=f'Цель: {selected_target_name}')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Δp, кПа')
    ax.set_title(f'Избыточное давление Δp от расстояния ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig1)

    # 2) Удельный импульс - отдельный график
    fig2, ax = plt.subplots(figsize=(10, 6))
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r, tnt_equivalent) for r in distance_range])
        specific_impulses = np.array(
            [calculate_specific_impulse(p, r, m, tnt_equivalent) for p, r in zip(overpressures, distance_range)])
        ax.plot(distance_range, specific_impulses, label=f'{m} кг', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Удельный импульс, Па·с')
    ax.set_title(f'Удельный импульс I = Δp·τ_+ ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig2)

    # 3) Скорость ударной волны - отдельный график
    fig3, ax = plt.subplots(figsize=(10, 6))
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r, tnt_equivalent) for r in distance_range])
        D = np.array([calculate_shock_wave_velocity(p) for p in overpressures])
        ax.plot(distance_range, D, label=f'{m} кг', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Скорость ударной волны, м/с')
    ax.set_title(f'Скорость ударной волны D_φ(R) ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig3)

    # 4) τ_+ - отдельный график
    fig4, ax = plt.subplots(figsize=(10, 6))
    for m in masses:
        taus = np.array([calculate_compression_duration(m, r, tnt_equivalent) for r in distance_range])
        ax.plot(distance_range, taus, label=f'{m} кг', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('τ_+, с')
    ax.set_title(f'Длительность фазы сжатия τ_+ (R) ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig4)

    # 5) p_dyn(R) - отдельный график
    fig5, ax = plt.subplots(figsize=(10, 6))
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r, tnt_equivalent) for r in distance_range])
        p_dyn = np.array([calculate_dynamic_pressure_from_state(p) for p in overpressures])
        ax.plot(distance_range, p_dyn / 1000.0, label=f'{m} кг', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('p_dyn, кПа')
    ax.set_title(f'Скоростной напор p_φок(R) ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig5)

    # 6) Временные сигналы - отдельный график
    fig6, ax = plt.subplots(figsize=(10, 6))
    max_tau = max([calculate_compression_duration(m, target_distance, tnt_equivalent) +
                   calculate_rarefaction_duration(m, tnt_equivalent) for m in masses])
    t_stop = max(3 * max_tau, 0.05)
    t = np.linspace(0, t_stop, 1000)

    for m in masses:
        p_t = pressure_time_history(m, target_distance, t, tnt_equivalent) * 1000.0
        p_dyn_t = dynamic_pressure_time_history(m, target_distance, t, tnt_equivalent) / 1000.0
        ax.plot(t, p_t, label=f'Δp(t), m={m} кг', linewidth=2)
        ax.plot(t, p_dyn_t, linestyle='--', label=f'p_dyn(t), m={m} кг', linewidth=2)

    ax.set_xlabel('Время, с')
    ax.set_ylabel('Давление, кПа')
    ax.set_title(f'Временные зависимости в точке R={target_distance:.2f} м ({explosive_name})')
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    figures.append(fig6)

    return figures


# ---------------------------
# Основное приложение Streamlit
# ---------------------------

def main():
    st.set_page_config(page_title="Симулятор взрывных воздействий", layout="wide")

    st.title("💥 Симулятор взрывных воздействий")
    st.markdown("Расчет параметров ударной волны для различных масс ВВ и расстояний")

    # Сайдбар с настройками
    with st.sidebar:

        st.header("⚙️ Параметры расчета")

        # Выбор цели
        selected_target_name = st.selectbox(
            "Выберите цель",
            options=list(DEFAULT_TARGETS.keys()),
            index=0
        )

        # Выбор степени разрушения
        damage_levels = get_damage_levels(selected_target_name)
        selected_damage_level = st.selectbox(
            "Степень разрушения",
            options=damage_levels,
            index=0,
            help="Выберите требуемую степень повреждения цели"
        )

        # Получаем стандартное значение
        selected_value = DEFAULT_TARGETS[selected_target_name][selected_damage_level]

        # Настройка стойкости цели
        custom_resistance = st.number_input(
            "Или задайте точное значение стойкости (кПа)",
            value=float(selected_value),
            min_value=1.0,
            max_value=1500.0,
            step=1.0,
            help="Можно ввести точное значение для тонкой настройки"
        )

        # Используем кастомное значение если оно отличается от стандартного
        if custom_resistance != selected_value:
            selected_value = custom_resistance
            damage_display = f"пользовательское значение ({custom_resistance} кПа)"
        else:
            damage_display = f"{selected_damage_level} ({selected_value} кПа)"

        st.info(f"**Выбрано:** {selected_target_name}\n"
                f"**Стойкость:** {damage_display}")

        st.header("⚙️ Параметры взрывчатого вещества")

        # Выбор типа ВВ
        explosive_names = list(EXPLOSIVES_DB.keys())
        selected_explosive_name = st.selectbox(
            "Выберите тип ВВ:",
            options=explosive_names,
            index=explosive_names.index(DEFAULT_EXPLOSIVE)
        )
        selected_explosive_data = EXPLOSIVES_DB[selected_explosive_name]
        tnt_equiv = selected_explosive_data["tnt_equivalent"]

        st.info(f"**{selected_explosive_name}**\n\n"
                f"Коэффициент приведения к тротилу: **{tnt_equiv}**\n\n"
                f"*Эквивалентная масса тротила = Масса ВВ × {tnt_equiv}*")

        # Ввод масс
        masses_input = st.text_input(
            "Массы заряда через запятую (кг)",
            value="2.0, 3.8, 5.0",
            help="Введите массы через запятую, например: 2.0, 3.8, 5.0"
        )
        masses = parse_mass_list(masses_input)

        # Диапазон расстояний
        dist_input = st.text_input(
            "Диапазон расстояний",
            value="1-10",
            help="Формат: '1-10' или '1:10:100' или '5'"
        )
        distance_range = parse_distance_input(dist_input)

        # Target distance
        target_distance = st.slider(
            "Расстояние между взрывом и целью (м)",
            min_value=float(distance_range[0]),
            max_value=float(distance_range[-1]),
            value=float(np.mean(distance_range)),
            step=0.1
        )

        # Кнопка расчета
        calculate_btn = st.button("🚀 Запустить расчет", type="primary")

    # Основная область
    if calculate_btn:
        st.success("✅ Расчет выполнен!")

        # Показываем используемые параметры
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Тип ВВ", selected_explosive_name)
        with col2:
            st.metric("Коэффициент", f"{tnt_equiv}")
        with col3:
            st.metric("Массы заряда", ", ".join(map(str, masses)) + " кг")
        with col4:
            st.metric("Цель", f"{selected_target_name}\n({selected_damage_level}: {selected_value} кПа)")

        # Создаем и показываем графики
        with st.spinner("Создание графиков..."):
            figures = create_plots(masses, distance_range, selected_target_name, selected_value,
                                   target_distance, tnt_equiv, selected_explosive_name)

            # Отображаем каждый график отдельно с заголовком
            st.subheader("📈 Избыточное давление Δp от расстояния")
            st.pyplot(figures[0])
            st.markdown(f"""
            **Анализ:** График показывает зависимость избыточного давления от расстояния для {selected_explosive_name}.
            Красная пунктирная линия indicates целевое значение давления для выбранной цели.
            """)

            st.subheader("📊 Удельный импульс I = Δp·τ₊")
            st.pyplot(figures[1])
            st.markdown("""
            **Анализ:** Удельный импульс характеризует суммарное воздействие ударной волны на цель.
            """)

            st.subheader("🚀 Скорость ударной волны D_φ(R)")
            st.pyplot(figures[2])
            st.markdown("""
            **Анализ:** Скорость распространения ударной волны уменьшается с расстоянием.
            """)

            st.subheader("⏱️ Длительность фазы сжатия τ₊ (R)")
            st.pyplot(figures[3])
            st.markdown("""
            **Анализ:** Длительность положительной фазы давления увеличивается с расстоянием.
            """)

            st.subheader("💨 Скоростной напор p_φок(R)")
            st.pyplot(figures[4])
            st.markdown("""
            **Анализ:** Скоростной напор характеризует динамическое воздействие на препятствия.
            """)

            st.subheader("⏰ Временные зависимости давления")
            st.pyplot(figures[5])
            st.markdown(f"""
            **Анализ:** Временные зависимости давления в точке R={target_distance:.2f} м.
            Сплошные линии - избыточное давление, пунктирные - скоростной напор.
            """)

        # Расчет минимальной массы
        st.subheader("📋 Результаты расчета")

        mm = find_minimum_mass_for_pressure(target_distance, selected_value, tnt_equiv)
        equivalent_mass = mm * tnt_equiv if mm else None

        if mm:
            st.success(
                f"**Минимальная масса {selected_explosive_name}**: **{mm:.3f} кг**\n\n"
                f"*Эквивалентная масса тротила: {equivalent_mass:.3f} кг*\n\n"
                f"Для {selected_damage_level.lower()} ({DEFAULT_TARGETS[selected_target_name][selected_damage_level]} кПа) цели '{selected_target_name}' "
                f"на расстоянии {target_distance:.2f} м"
            )
        else:
            st.warning(
                f"Не удалось достичь требуемого давления {selected_value} кПа на расстоянии {target_distance:.2f} м"
            )

        # Таблица результатов
        results_data = []
        for m in masses:
            p = calculate_overpressure(m, target_distance, tnt_equiv) * 1000.0
            p_dyn = calculate_dynamic_pressure_from_state(
                calculate_overpressure(m, target_distance, tnt_equiv)) / 1000.0
            tau = calculate_compression_duration(m, target_distance, tnt_equiv)
            I = calculate_specific_impulse(calculate_overpressure(m, target_distance, tnt_equiv),
                                           target_distance, m, tnt_equiv)
            results_data.append({
                "Масса ВВ, кг": m,
                "Эквив. масса TNT, кг": round(m * tnt_equiv, 3),
                "Δp, кПа": round(p, 2),
                "p_dyn, кПа": round(p_dyn, 3),
                "τ_+, с": round(tau, 4),
                "I, Па·с": round(I, 3)
            })

        df = pd.DataFrame(results_data)
        st.dataframe(df.style.format({
            "Масса ВВ, кг": "{:.2f}",
            "Эквив. масса TNT, кг": "{:.3f}",
            "Δp, кПа": "{:.2f}",
            "p_dyn, кПа": "{:.3f}",
            "τ_+, с": "{:.4f}",
            "I, Па·с": "{:.3f}"
        }))

        # Кнопка скачивания результатов
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Скачать результаты (CSV)",
            data=csv,
            file_name="blast_simulation_results.csv",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()

