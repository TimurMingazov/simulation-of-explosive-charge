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
    "Кирпичные малоэтажные здания": 40.0,
    "Промышленные здания (металл, ж/б каркас)": 80.0,
    "Гусеница танка": 150.0,
    "Лёгкие автомобильные конструкции": 60.0,
    "Бетонные ограждения": 120.0,
}


# ---------------------------
# Базовые функции расчёта
# ---------------------------

def calculate_scaled_radius(mass):
    return 0.05 * (mass ** (1 / 3))


def calculate_scaled_distance(mass, distance):
    R_s = calculate_scaled_radius(mass)
    return distance / R_s


def calculate_overpressure(mass, distance):
    distance = max(distance, 1e-6)
    return (1.4 * mass / (distance ** 3)) + (0.43 * (mass ** (2 / 3)) / (distance ** 2)) + (
                0.11 * (mass ** (1 / 3)) / distance)


def calculate_compression_duration(mass, distance):
    distance = max(distance, 1e-6)
    return 1.5e-3 * (mass ** (1 / 6)) * np.sqrt(distance)


def calculate_rarefaction_pressure(mass, distance):
    distance = max(distance, 1e-6)
    return -0.03 * (mass ** (1 / 3)) / distance


def calculate_rarefaction_duration(mass):
    return 0.013 * (mass ** (1 / 3))


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


def calculate_specific_impulse(overpressure, distance, mass):
    tau = calculate_compression_duration(mass, distance)
    return overpressure * 1e6 * tau


# ---------------------------
# Временные сигналы (упрощённо)
# ---------------------------

def pressure_time_history(mass, distance, t_array):
    p_peak = calculate_overpressure(mass, distance)
    tau_p = calculate_compression_duration(mass, distance)
    p_neg = calculate_rarefaction_pressure(mass, distance)
    tau_n = calculate_rarefaction_duration(mass)

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


def dynamic_pressure_time_history(mass, distance, t_array):
    p_t = pressure_time_history(mass, distance, t_array)
    return np.array([calculate_dynamic_pressure_from_state(p) for p in p_t])


# ---------------------------
# Минимальная масса (бинарный поиск)
# ---------------------------

def find_minimum_mass_for_pressure(target_distance, pressure_resistance_kpa, mass_lower=0.01, mass_upper=5000.0,
                                   tol=1e-3):
    pr_mpa = pressure_resistance_kpa / 1000.0
    if calculate_overpressure(mass_upper, target_distance) < pr_mpa:
        return None
    lo = mass_lower
    hi = mass_upper
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if calculate_overpressure(mid, target_distance) >= pr_mpa:
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

def create_plots(masses, distance_range, selected_target_name, selected_target_value, target_distance):
    distance_range = np.array(distance_range)

    # Создаем фигуры
    fig1, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    # 1) Δp
    ax = axes[0]
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r) for r in distance_range]) * 1000.0
        ax.plot(distance_range, overpressures, label=f'{m} кг')
    ax.axhline(y=selected_target_value, color='r', linestyle='--', label=f'Цель: {selected_target_name}')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Δp, кПа')
    ax.set_title('Избыточное давление Δp от расстояния')
    ax.grid(True)
    ax.legend(fontsize='small')

    # 2) Удельный импульс
    ax = axes[1]
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r) for r in distance_range])
        specific_impulses = np.array(
            [calculate_specific_impulse(p, r, m) for p, r in zip(overpressures, distance_range)])
        ax.plot(distance_range, specific_impulses, label=f'{m} кг')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Удельный импульс, Па·с')
    ax.set_title('Удельный импульс I = Δp·τ_+')
    ax.grid(True)
    ax.legend(fontsize='small')

    # 3) Скорость ударной волны
    ax = axes[2]
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r) for r in distance_range])
        D = np.array([calculate_shock_wave_velocity(p) for p in overpressures])
        ax.plot(distance_range, D, label=f'{m} кг')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Скорость ударной волны, м/с')
    ax.set_title('Скорость ударной волны D_φ(R)')
    ax.grid(True)
    ax.legend(fontsize='small')

    # 4) τ_+
    ax = axes[3]
    for m in masses:
        taus = np.array([calculate_compression_duration(m, r) for r in distance_range])
        ax.plot(distance_range, taus, label=f'{m} кг')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('τ_+, с')
    ax.set_title('Длительность фазы сжатия τ_+ (R)')
    ax.grid(True)
    ax.legend(fontsize='small')

    # 5) p_dyn(R)
    ax = axes[4]
    for m in masses:
        overpressures = np.array([calculate_overpressure(m, r) for r in distance_range])
        p_dyn = np.array([calculate_dynamic_pressure_from_state(p) for p in overpressures])
        ax.plot(distance_range, p_dyn / 1000.0, label=f'{m} кг')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('p_dyn, кПа')
    ax.set_title('Скоростной напор p_φок(R)')
    ax.grid(True)
    ax.legend(fontsize='small')

    # 6) Контур p_dyn(q,R)
    ax = axes[5]
    mass_grid = np.linspace(min(masses), max(masses), 50)
    R_grid = np.linspace(np.min(distance_range), np.max(distance_range), 50)
    M, Rg = np.meshgrid(mass_grid, R_grid)
    p_dyn_grid = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            p_dyn_grid[i, j] = calculate_dynamic_pressure_from_state(calculate_overpressure(M[i, j], Rg[i, j])) / 1000.0
    cs = ax.contourf(M, Rg, p_dyn_grid, levels=30)
    ax.set_xlabel('Масса, кг')
    ax.set_ylabel('Расстояние, м')
    ax.set_title('Контур p_φок(q, R) (кПа)')
    plt.colorbar(cs, ax=ax, label='p_φок, кПа')

    plt.tight_layout()

    # Временные сигналы
    fig2, ax = plt.subplots(figsize=(10, 6))
    max_tau = max(
        [calculate_compression_duration(m, target_distance) + calculate_rarefaction_duration(m) for m in masses])
    t_stop = max(3 * max_tau, 0.05)
    t = np.linspace(0, t_stop, 1000)

    for m in masses:
        p_t = pressure_time_history(m, target_distance, t) * 1000.0
        p_dyn_t = dynamic_pressure_time_history(m, target_distance, t) / 1000.0
        ax.plot(t, p_t, label=f'Δp(t), m={m} кг')
        ax.plot(t, p_dyn_t, linestyle='--', label=f'p_dyn(t), m={m} кг')

    ax.set_xlabel('Время, с')
    ax.set_ylabel('Давление, кПа')
    ax.set_title(f'Временные зависимости в точке R={target_distance:.2f} м')
    ax.grid(True)
    ax.legend(fontsize='small')

    return fig1, fig2


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

        # Ввод масс
        masses_input = st.text_input(
            "Массы заряда через запятую (кг)",
            value="2.0, 3.8, 5.0",
            help="Введите массы через запятую, например: 2.0, 3.8, 5.0"
        )
        masses = parse_mass_list(masses_input)

        # Выбор цели
        selected_name = st.selectbox(
            "Выберите цель",
            options=list(DEFAULT_TARGETS.keys()),
            index=0
        )
        selected_value = DEFAULT_TARGETS[selected_name]

        # Настройка стойкости цели
        custom_resistance = st.number_input(
            "Стойкость цели (кПа)",
            value=float(selected_value),
            min_value=1.0,
            max_value=1000.0,
            step=1.0,
            help="Можно изменить значение стойкости выбранной цели"
        )
        selected_value = custom_resistance

        # Диапазон расстояний
        dist_input = st.text_input(
            "Диапазон расстояний",
            value="1-10",
            help="Формат: '1-10' или '1:10:100' или '5'"
        )
        distance_range = parse_distance_input(dist_input)

        # Target distance
        target_distance = st.slider(
            "Расстояние для детального анализа (м)",
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
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Массы заряда", ", ".join(map(str, masses)) + " кг")
        with col2:
            st.metric("Цель", f"{selected_name} ({selected_value} кПа)")
        with col3:
            st.metric("Диапазон расстояний", f"{distance_range[0]:.1f}-{distance_range[-1]:.1f} м")

        # Создаем и показываем графики
        with st.spinner("Создание графиков..."):
            fig1, fig2 = create_plots(masses, distance_range, selected_name, selected_value, target_distance)

            st.subheader("📊 Основные графики")
            st.pyplot(fig1)

            st.subheader("⏰ Временные зависимости")
            st.pyplot(fig2)

        # Расчет минимальной массы
        st.subheader("📋 Результаты расчета")

        mm = find_minimum_mass_for_pressure(target_distance, selected_value)
        if mm:
            st.success(
                f"**Минимальная масса** для поражения цели '{selected_name}' на расстоянии {target_distance:.2f} м: **{mm} кг**")
        else:
            st.warning(
                f"Не удалось достичь требуемого давления {selected_value} кПа на расстоянии {target_distance:.2f} м")

        # Таблица результатов
        results_data = []
        for m in masses:
            p = calculate_overpressure(m, target_distance) * 1000.0
            p_dyn = calculate_dynamic_pressure_from_state(calculate_overpressure(m, target_distance)) / 1000.0
            tau = calculate_compression_duration(m, target_distance)
            I = calculate_specific_impulse(calculate_overpressure(m, target_distance), target_distance, m)
            results_data.append({
                "Масса, кг": m,
                "Δp, кПа": round(p, 2),
                "p_dyn, кПа": round(p_dyn, 3),
                "τ_+, с": round(tau, 4),
                "I, Па·с": round(I, 3)
            })

        df = pd.DataFrame(results_data)
        st.dataframe(df.style.format({
            "Масса, кг": "{:.2f}",
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