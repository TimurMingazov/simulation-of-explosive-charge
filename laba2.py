import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import bisect
import os
import tempfile
from docxtpl import DocxTemplate, InlineImage
from docx.shared import Mm
from io import BytesIO
import base64


SOUND_SPEED = 340.0  # м/с
AIR_DENSITY = 1.225  # кг/м³
ADIABATIC_INDEX = 1.4
B_COEFF = (ADIABATIC_INDEX + 1) / (2 * ADIABATIC_INDEX)
C_COEFF = (ADIABATIC_INDEX - 1) / (2 * ADIABATIC_INDEX)

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

# Класс GPVSExplosion (остается без изменений)
class GPVSExplosion:
    def __init__(self, fuel_type, q, theta, target_type, target_size, protection_level):
        self.fuel_type = fuel_type
        self.q = q
        self.theta = theta
        self.target_type = target_type
        self.target_size = target_size
        self.protection_level = protection_level

        self.fuel_properties = {
            'Этан': {'μ_r': 30, 'C_st': 0.06, 'Q_v': 3.5e6, 'γ': 1.257, 'ρ_st': 1.25, 'D': 1800, 'q_m': 2.797,
                     'q_v': 3.496, 'c_sth': 5.66},
            'Этилен': {'μ_r': 28, 'C_st': 0.065, 'Q_v': 3.8e6, 'γ': 1.25, 'ρ_st': 1.15, 'D': 1880, 'q_m': 3.010,
                       'q_v': 3.869, 'c_sth': 6.54},
            'Пропилен': {'μ_r': 42, 'C_st': 0.05, 'Q_v': 3.6e6, 'γ': 1.22, 'ρ_st': 1.20, 'D': 1840, 'q_m': 2.922,
                         'q_v': 3.839, 'c_sth': 4.46},
            'Метан': {'μ_r': 16, 'C_st': 0.095, 'Q_v': 5.0e6, 'γ': 1.3, 'ρ_st': 0.72, 'D': 1800, 'q_m': 2.834,
                      'q_v': 3.696, 'c_sth': 12.3},
            'Пропан': {'μ_r': 44, 'C_st': 0.04, 'Q_v': 4.6e6, 'γ': 1.2, 'ρ_st': 1.52, 'D': 1850, 'q_m': 2.801,
                       'q_v': 3.676, 'c_sth': 4.03}
        }

        self.p0 = 101325
        self.Q_t = 4.184e6
        self.a = 340
        self.ρ_air = 1.225

        if fuel_type in self.fuel_properties:
            props = self.fuel_properties[fuel_type]
            self.μ_r = props['μ_r']
            self.C_st = props['C_st']
            self.Q_v = props['Q_v']
            self.γ = props['γ']
            self.ρ_st = props['ρ_st']
        else:
            self.μ_r = 30
            self.C_st = 0.06
            self.Q_v = 4.5e6
            self.γ = 1.2
            self.ρ_st = 1.25

    def calculate_detonation_velocity(self):
        term = 2 * (self.γ ** 2 - 1) * self.Q_v / (self.γ ** 2)
        D = np.sqrt(abs(term))
        return min(D, 2500)

    def calculate_cloud_volume_natural(self):
        V0 = (22.49 * self.q) / (self.μ_r * self.C_st)
        return V0

    def calculate_cloud_mass_natural(self):
        V0 = self.calculate_cloud_volume_natural()
        m = self.ρ_st * V0
        return m

    def calculate_tnt_equivalent(self):
        m = self.calculate_cloud_mass_natural()
        q_tnt = 2 * m * self.Q_v / self.Q_t
        return max(q_tnt, 0.1)

    def calculate_pressure_drop(self):
        D = self.calculate_detonation_velocity()
        Δp2 = (self.ρ_st * D ** 2) / (self.γ + 1)
        return max(Δp2 / 1e6, 0.1)

    def calculate_pressure_outside_cloud(self, R_relative):
        try:
            log_R = np.log10(max(R_relative, 0.1))
            p_relative = 0.65 - 2.18 * log_R + 0.52 * (log_R) ** 2
            Δp_m = self.p0 * (10 ** p_relative)
            return max(Δp_m / 1e6, 0.001)
        except:
            return 0.001

    def calculate_impulse_outside_cloud(self, R_relative):
        try:
            log_R = np.log10(max(R_relative, 0.1))
            I_relative = 2.11 - 0.97 * log_R + 0.44 * (log_R) ** 2
            q_tnt = self.calculate_tnt_equivalent()
            I = I_relative * (q_tnt ** (1 / 3))
            return max(I, 0.1)
        except:
            return 0.1

    def calculate_specific_impulse(self, R_relative):
        I = self.calculate_impulse_outside_cloud(R_relative)
        q_tnt = self.calculate_tnt_equivalent()
        I_specific = I / q_tnt
        return I_specific

    def calculate_shock_wave_velocity(self, R_relative):
        Δp = self.calculate_pressure_outside_cloud(R_relative) * 1e6
        U = self.a * np.sqrt(1 + (self.γ + 1) / (2 * self.γ) * (Δp / self.p0))
        return U

    def calculate_dynamic_pressure(self, R_relative):
        Δp = self.calculate_pressure_outside_cloud(R_relative) * 1e6
        U = self.calculate_shock_wave_velocity(R_relative)
        ρ_after_shock = self.ρ_air * (U / self.a)
        q_dynamic = 0.5 * ρ_after_shock * U ** 2
        return q_dynamic / 1e3

    def calculate_pressure_at_distance(self, distance):
        q_tnt = self.calculate_tnt_equivalent()
        R_relative = distance / (q_tnt ** (1 / 3))
        return self.calculate_pressure_outside_cloud(R_relative) * 1000


def calculate_overpressure_gvv(mass, distance, explosive_type='Тротил (TNT)'):
    """Расчет избыточного давления для ГВВ"""
    tnt_equivalent = EXPLOSIVES_DB[explosive_type]["tnt_equivalent"]
    equivalent_mass = mass * tnt_equivalent
    distance = max(distance, 1e-6)
    return (1.4 * equivalent_mass / (distance ** 3) +
           0.43 * (equivalent_mass ** (2 / 3)) / (distance ** 2) +
           0.11 * (equivalent_mass ** (1 / 3)) / distance)


def calculate_compression_duration(mass, distance, explosive_type='Тротил (TNT)'):
    """Длительность фазы сжатия"""
    tnt_equivalent = EXPLOSIVES_DB[explosive_type]["tnt_equivalent"]
    equivalent_mass = mass * tnt_equivalent
    distance = max(distance, 1e-6)
    return 1.5e-3 * (equivalent_mass ** (1 / 6)) * np.sqrt(distance)


def calculate_shock_wave_velocity_gvv(overpressure):
    """Скорость ударной волны для ГВВ"""
    p_rel = overpressure
    return SOUND_SPEED * np.sqrt(1 + p_rel * B_COEFF)


def calculate_dynamic_pressure_gvv(overpressure):
    """Скоростной напор для ГВВ"""
    p_rel = overpressure
    rho = AIR_DENSITY * (1 + p_rel * B_COEFF) / (1 + p_rel * C_COEFF)
    v = SOUND_SPEED * p_rel / (ADIABATIC_INDEX * np.sqrt(1 + p_rel * B_COEFF) + 1e-12)
    return 0.5 * rho * v ** 2


def calculate_specific_impulse_gvv(overpressure, distance, mass, explosive_type='Тротил'):
    """Удельный импульс для ГВВ"""
    tnt_equivalent = EXPLOSIVES_DB[explosive_type]["tnt_equivalent"]
    tau = calculate_compression_duration(mass, distance, explosive_type)
    return overpressure * 1e6 * tau


def find_minimum_mass_gvv(distance, target_pressure_kpa, explosive_type='Тротил', mass_range=(0.01, 1000), tol=0.01):
    """Поиск минимальной массы ГВВ для поражения цели"""
    target_pressure_mpa = target_pressure_kpa / 1000.0

    def pressure_difference(mass):
        pressure = calculate_overpressure_gvv(mass, distance, explosive_type)
        return pressure - target_pressure_mpa

    try:
        min_mass = bisect(pressure_difference, mass_range[0], mass_range[1], xtol=tol)
        return min_mass
    except ValueError:
        # Линейный поиск
        masses = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 1000)
        for mass in masses:
            pressure = calculate_overpressure_gvv(mass, distance, explosive_type)
            if pressure >= target_pressure_mpa:
                return mass
        return None



TARGET_DATABASE = {
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
    },
    "Танк": {
        "Полное разрушение": 170.0,
        "Сильные повреждения": 150.0,
        "Средние повреждения": 120.0,
        "Слабые повреждения": 100.0
    }
}

# Словарь для соответствия целей и их изображений
TARGET_IMAGES = {
    "Кирпичные малоэтажные здания": "small_zdanie.jpg",
    "Промышленные здания (металл, ж/б каркас)": "zdanie.png",
    "Подземные сети коммунального хозяйства": "komunalka.jpg",
    "Надводные корабли": "korabl.jpg",
    "Самолеты на аэродроме": "samolet.jpg",
    "Танк": "tank.png"
}

FUEL_TYPES = ['Этан', 'Этилен', 'Пропилен', 'Метан', 'Пропан']


def find_minimum_mass_gpvs(fuel_type, distance, target_pressure_kpa, mass_range=(0.01, 1000), tol=0.01):
    def pressure_difference(mass):
        explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, target_pressure_kpa)
        pressure = explosion.calculate_pressure_at_distance(distance)
        return pressure - target_pressure_kpa

    try:
        min_mass = bisect(pressure_difference, mass_range[0], mass_range[1], xtol=tol)
        return min_mass
    except ValueError:
        pressures = []
        masses = np.logspace(np.log10(mass_range[0]), np.log10(mass_range[1]), 100)
        for mass in masses:
            explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, target_pressure_kpa)
            pressure = explosion.calculate_pressure_at_distance(distance)
            pressures.append(pressure)
        for i, pressure in enumerate(pressures):
            if pressure >= target_pressure_kpa:
                return masses[i]
        return None


# Функции для построения графиков
def plot_pressure_vs_distance_gpvs(fuel_type, mass, distance_range=(1, 50)):
    explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, 70)
    distances = np.linspace(distance_range[0], distance_range[1], 100)
    pressures = [explosion.calculate_pressure_at_distance(dist) for dist in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, pressures, 'b-', linewidth=2, label=f'ГПВС ({fuel_type})')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Давление, кПа')
    ax.set_title(f'Давление от расстояния для ГПВС\nМасса: {mass:.3f} кг, Горючее: {fuel_type}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def plot_pressure_comparison_gpvs_gvv(fuel_type, mass_gpvs, mass_gvv, explosive_type, distance_range=(1, 50)):
    # График для ГПВС
    explosion_gpvs = GPVSExplosion(fuel_type, mass_gpvs, 1, 'цель', 0.1, 70)
    distances = np.linspace(distance_range[0], distance_range[1], 100)
    pressures_gpvs = [explosion_gpvs.calculate_pressure_at_distance(dist) for dist in distances]

    # График для ГВВ - исправленная строка
    pressures_gvv = [calculate_overpressure_gvv(mass_gvv, dist, explosive_type) * 1000 for dist in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, pressures_gpvs, 'b-', linewidth=2, label=f'ГПВС ({fuel_type}, {mass_gpvs:.3f} кг)')
    ax.plot(distances, pressures_gvv, 'r--', linewidth=2, label=f'ГВВ ({explosive_type}, {mass_gvv:.3f} кг)')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Давление, кПа')
    ax.set_title('Сравнение давления: ГПВС vs ГВВ')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def plot_mass_variation_analysis(fuel_type, min_mass, target_distance, target_pressure, distance_range=(1, 50)):
    mass_variations = [0.8, 0.9, 1.0, 1.1]
    masses = [min_mass * factor for factor in mass_variations]
    labels = [f'{factor * 100:.0f}% ({mass:.3f} кг)' for factor, mass in zip(mass_variations, masses)]
    distances = np.linspace(distance_range[0], distance_range[1], 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['red', 'orange', 'green', 'blue']

    for i, mass in enumerate(masses):
        explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, target_pressure)
        pressures = [explosion.calculate_pressure_at_distance(dist) for dist in distances]
        ax.plot(distances, pressures, color=colors[i], linewidth=2, label=labels[i])

    ax.axhline(y=target_pressure, color='black', linestyle=':', linewidth=2,
               label=f'Целевое давление: {target_pressure} кПа')
    ax.axvline(x=target_distance, color='gray', linestyle='--', linewidth=1,
               label=f'Целевое расстояние: {target_distance} м')

    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Давление, кПа')
    ax.set_title(f'Влияние массы заряда ГПВС на давление\nГорючее: {fuel_type}, Минимальная масса: {min_mass:.3f} кг')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def plot_dynamic_pressure_vs_distance(fuel_type, mass, distance_range=(1, 50)):
    explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, 70)
    distances = np.linspace(distance_range[0], distance_range[1], 100)
    q_tnt = explosion.calculate_tnt_equivalent()
    dynamic_pressures = [explosion.calculate_dynamic_pressure(dist / (q_tnt ** (1 / 3))) for dist in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, dynamic_pressures, 'g-', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Скоростной напор, кПа')
    ax.set_title(f'Скоростной напор от расстояния\nМасса: {mass:.3f} кг, Горючее: {fuel_type}')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    return fig


def plot_specific_impulse_vs_distance(fuel_type, mass, distance_range=(1, 50)):
    explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, 70)
    distances = np.linspace(distance_range[0], distance_range[1], 100)
    q_tnt = explosion.calculate_tnt_equivalent()
    specific_impulses = [explosion.calculate_specific_impulse(dist / (q_tnt ** (1 / 3))) for dist in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, specific_impulses, 'purple', linewidth=2)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Удельный импульс, Па·с/кг')
    ax.set_title(f'Удельный импульс от расстояния\nМасса: {mass:.3f} кг, Горючее: {fuel_type}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_shock_wave_velocity_vs_distance(fuel_type, mass, distance_range=(1, 50)):
    explosion = GPVSExplosion(fuel_type, mass, 1, 'цель', 0.1, 70)
    distances = np.linspace(distance_range[0], distance_range[1], 100)
    q_tnt = explosion.calculate_tnt_equivalent()
    velocities = [explosion.calculate_shock_wave_velocity(dist / (q_tnt ** (1 / 3))) for dist in distances]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(distances, velocities, 'red', linewidth=2, label='Скорость УВ')
    ax.axhline(y=explosion.a, color='gray', linestyle='--',
               label=f'Скорость звука ({explosion.a} м/с)')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('Скорость, м/с')
    ax.set_title(f'Скорость ударной волны от расстояния\nМасса: {mass:.3f} кг, Горючее: {fuel_type}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def run():
    """Основная функция Streamlit приложения"""
    st.set_page_config(page_title="Расчет параметров взрыва ГПВС", layout="wide")

    st.title("Программа расчета параметров взрыва газопаровоздушных смесей")
    st.markdown("---")




    # Сайдбар для ввода данных
    with st.sidebar:
        st.header("Параметры расчета")

        # Личные данные студента
        st.subheader("Данные студента")
        student_name = st.text_input("ФИО студента", "Иванов И.И.")
        student_group = st.text_input("Группа", "И912С")
        variant = st.number_input("Вариант", min_value=1, max_value=30, value=1)

        # Выбор цели и степени разрушения
        st.subheader("Параметры цели")
        target_options = list(TARGET_DATABASE.keys())
        selected_target = st.selectbox("Выберите цель", target_options, index=0)

        destruction_options = list(TARGET_DATABASE[selected_target].keys())
        selected_destruction = st.selectbox("Степень разрушения", destruction_options, index=2)
        protection = TARGET_DATABASE[selected_target][selected_destruction]

        distance = st.number_input("Расстояние до цели (м)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)

        # Выбор веществ
        st.subheader("Параметры веществ")
        fuel_type = st.selectbox("Тип ГПВС", FUEL_TYPES, index=0)
        explosive_type = st.selectbox("Тип ВВ", list(EXPLOSIVES_DB.keys()), index=0)

        # Кнопка расчета
        calculate_btn = st.button("🧨 Выполнить расчет", type="primary")

    # Основная область
    if calculate_btn:
        # Выполнение расчета
        with st.spinner("Выполняется расчет..."):
            # Расчет минимальных масс
            min_mass_gpvs = find_minimum_mass_gpvs(fuel_type, distance, protection)
            min_mass_gvv = find_minimum_mass_gvv(distance, protection, explosive_type)

            if min_mass_gpvs:
                # Создание графиков
                graphs = {}
                graphs['pressure_gpvs'] = plot_pressure_vs_distance_gpvs(fuel_type, min_mass_gpvs,
                                                                         (1, min(distance * 3, 50)))
                graphs['comparison'] = plot_pressure_comparison_gpvs_gvv(fuel_type, min_mass_gpvs, min_mass_gvv,
                                                                         explosive_type, (1, min(distance * 3, 50)))
                graphs['different_masses_gpvs'] = plot_mass_variation_analysis(fuel_type, min_mass_gpvs, distance,
                                                                               protection, (1, min(distance * 3, 50)))
                graphs['highspeed_pressure_gpvs'] = plot_dynamic_pressure_vs_distance(fuel_type, min_mass_gpvs,
                                                                                      (1, min(distance * 3, 50)))
                graphs['impuls_gpvs'] = plot_specific_impulse_vs_distance(fuel_type, min_mass_gpvs,
                                                                          (1, min(distance * 3, 50)))
                graphs['speed_wave_gpvs'] = plot_shock_wave_velocity_vs_distance(fuel_type, min_mass_gpvs,
                                                                                 (1, min(distance * 3, 50)))

                graphs['comparison'] = plot_pressure_comparison_gpvs_gvv(fuel_type, min_mass_gpvs, min_mass_gvv,
                                                                              explosive_type,
                                                                              (1, min(distance * 3, 50)))

                # Получение параметров вещества
                explosion = GPVSExplosion(fuel_type, min_mass_gpvs, 1, selected_target, 0.1, protection)
                fuel_props = explosion.fuel_properties[fuel_type]

                efficiency = min_mass_gvv / min_mass_gpvs if min_mass_gvv else 0
                efficiency_status = "ГПВС эффективнее ГВВ" if efficiency > 1 else "ГВВ эффективнее ГПВС"

        # Результаты расчета
        st.markdown("---")
        st.header("📊 Результаты расчета")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Минимальная масса ГПВС", f"{min_mass_gpvs:.3f} кг")
            st.metric("Тип горючего", fuel_type)
        with col2:
            st.metric("Минимальная масса ГВВ", f"{min_mass_gvv:.3f} кг")
            st.metric("Тип ВВ", explosive_type)
        with col3:
            st.metric("Коэффициент эффективности", f"{efficiency:.2f}")
            st.metric("Эффективность", efficiency_status)

        # Отображение графиков
        st.markdown("---")
        st.header("📈 Графики анализа")

        tabs = st.tabs(["Давление ГПВС", "Сравнение ГПВС/ГВВ", "Разные массы",
                        "Скоростной напор", "Удельный импульс", "Скорость УВ"])

        with tabs[0]:
            st.pyplot(graphs['pressure_gpvs'])
        with tabs[1]:
            st.pyplot(graphs['comparison'])
        with tabs[2]:
            st.pyplot(graphs['different_masses_gpvs'])
        with tabs[3]:
            st.pyplot(graphs['highspeed_pressure_gpvs'])
        with tabs[4]:
            st.pyplot(graphs['impuls_gpvs'])
        with tabs[5]:
            st.pyplot(graphs['speed_wave_gpvs'])

        # Генерация отчета
        st.markdown("---")
        st.header("📄 Генерация отчета")

        try:
            st.info("🔄 Начинаю генерацию отчета...")

            # Проверяем шаблон
            template_path = "pattern_laba2.docx"
            st.info(f"🔍 Проверяю наличие шаблона: {template_path}")

            if not os.path.exists(template_path):
                st.error(f"❌ Файл шаблона '{template_path}' не найден!")
                st.info("📁 Содержимое текущей директории:")
                for file in os.listdir('.'):
                    st.write(f" - {file}")
                return

            st.success("✅ Шаблон найден")

            # Создаем буферы для графиков
            st.info("🔄 Подготавливаю графики...")
            graph_buffers = {}

            for graph_name, fig in graphs.items():
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                graph_buffers[graph_name] = buf
                plt.close(fig)

            st.success(f"✅ Графики подготовлены: {len(graph_buffers)} шт.")

            # Загружаем шаблон
            st.info("📄 Загружаю шаблон...")
            doc = DocxTemplate(template_path)

            # Создаем контекст с правильными именами переменных для шаблона
            st.info("🔄 Создаю контекст данных...")

            # Получаем изображение цели
            target_image_name = TARGET_IMAGES.get(selected_target, "default_target.jpg")
            target_image_path = os.path.join("media2", target_image_name)

            # Создаем контекст с именами переменных как в шаблоне
            context = {
                # Основные данные
                'name': student_name,
                'var': variant,
                'gpvs': fuel_type,
                'target': selected_target,
                'protection': f"{protection:.1f}",
                'dist_before_target': f"{distance:.1f}",
                'min_mass': f"{min_mass_gpvs:.3f}",

                # Характеристики вещества (используем реальные данные из fuel_properties)
                'D': f"{fuel_props['D']:.0f}",
                'rho': f"{fuel_props['ρ_st']:.2f}",
                'Q_m': f"{fuel_props['q_m']:.3f}",
                'Q_v': f"{fuel_props['q_v']:.3f}",
                'y': f"{fuel_props['γ']:.3f}",
                'μ': f"{fuel_props['μ_r']:.0f}",
                'C': f"{fuel_props['C_st'] * 100:.1f}",
                'P': f"{explosion.calculate_pressure_drop():.2f}",
                'c_sth': f"{fuel_props['c_sth']:.2f}",

                # Дополнительные параметры
                'degree_destruction': get_destruction_description(selected_destruction),
                'efficiency_status': efficiency_status,
            }

            # Добавляем изображение цели в контекст
            st.info("🖼️ Добавляю изображение цели...")
            if os.path.exists(target_image_path):
                with open(target_image_path, "rb") as img_file:
                    target_image_buffer = BytesIO(img_file.read())
                context['target_pic'] = InlineImage(doc, target_image_buffer, width=Mm(120))
                st.success(f"✅ Изображение цели добавлено: {target_image_name}")
            else:
                st.warning(f"⚠️ Изображение цели не найдено: {target_image_path}")
                # Создаем пустой буфер для избежания ошибок
                context['target_pic'] = ""

            # Добавляем графики в контекст
            st.info("📊 Добавляю графики в отчет...")

            # Список графиков для отчета (имена должны совпадать с метками в шаблоне)
            graph_mapping = {
                'pressure_gpvs': 'pressure_gpvs',
                'comparison': 'comparison',
                'different_masses_gpvs': 'different_masses_gpvs',
                'highspeed_pressure_gpvs': 'highspeed_pressure_gpvs',
                'impuls_gpvs': 'impuls_gpvs',
                'speed_wave_gpvs': 'speed_wave_gpvs'
            }

            for template_var, graph_key in graph_mapping.items():
                try:
                    if graph_key in graph_buffers:
                        # Создаем InlineImage для каждого графика
                        context[template_var] = InlineImage(
                            doc,
                            graph_buffers[graph_key],
                            width=Mm(150)
                        )
                        st.success(f"✅ {template_var} добавлен")
                    else:
                        st.warning(f"⚠️ {template_var} не найден в результатах")
                        context[template_var] = ""
                except Exception as e:
                    st.error(f"❌ Ошибка при добавлении {template_var}: {str(e)}")
                    context[template_var] = ""

            # Рендерим документ
            st.info("🎨 Формирую отчет...")
            doc.render(context)

            # Сохраняем в буфер
            st.info("💾 Сохраняю отчет...")
            output_buffer = BytesIO()
            doc.save(output_buffer)
            output_buffer.seek(0)

            # Закрываем буферы
            for buf in graph_buffers.values():
                buf.close()
            if 'target_image_buffer' in locals():
                target_image_buffer.close()

            st.success("✅ Отчет готов!")

            # Кнопка скачивания
            st.download_button(
                label="📥 Скачать отчет DOCX",
                data=output_buffer,
                file_name=f"{student_group}_{student_name.replace(' ', '_')}_Лабораторная_2.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        except Exception as e:
            st.error(f"❌ Ошибка при генерации отчета: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    else:
        st.info("👈 Введите параметры в левой панели и нажмите 'Выполнить расчет'")
        st.info("Начальные данные должны быть, как и в первой работе. Если вы выбирали в качестве цели "
                "'Надводные корабли' 'Сильные повреждения', а ГВВ был 'Тротил', то тут берете также."
                "Остальные, как по кайфу.")
        st.image(os.path.join("media2", "mem2.jpg"))


def get_destruction_description(destruction_level):
    descriptions = {
        'Полное разрушение': 'полного разрушения цели',
        'Сильные повреждения': 'сильного разрушения цели',
        'Средние повреждения': 'среднего разрушения цели',
        'Слабые повреждения': 'слабого разрушения цели'
    }
    return descriptions.get(destruction_level, 'поражения цели')

