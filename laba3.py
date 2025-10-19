import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import streamlit as st
import io
import base64
from docxtpl import DocxTemplate
import tempfile
import os
from docxtpl import InlineImage
from docx.shared import Mm
from io import BytesIO


class Target:
    """Класс для описания целей"""

    def __init__(self, name, critical_pressure, velocity, armor_thickness, area, picture, armor_density=7800,
                 armor_strength=1e9):
        self.name = name
        self.critical_pressure = critical_pressure  # кПа
        self.velocity = velocity  # м/с
        self.armor_thickness = armor_thickness  # м
        self.area = area  # м²
        self.picture = picture # схема цели
        self.armor_density = armor_density  # кг/м³
        self.armor_strength = armor_strength  # Па


class FragmentationWarhead:
    def __init__(self):
        # Константы
        self.rho_steel = 7800  # плотность стали, кг/м³
        self.rho_tnt = 1600  # плотность тротила, кг/м³
        self.D_tnt = 7000  # скорость детонации тротила, м/с
        self.c_x = 1.24  # коэффициент лобового сопротивления

        # Параметры БЧ
        self.L_bch = 0.3  # длина БЧ, м
        self.D_max = 0.3  # максимальный диаметр БЧ, м
        self.fragment_size = 0.01  # размер осколка (куб 1 см)

        # Цели по умолчанию
        self.targets = {
            "Танк Abrams M1A2": Target("Танк Abrams M1A2", 200, 20, 0.03, 8.0, "tank.png"),
            "Вертолет Ми-8Т": Target("Вертолет Ми-8Т", 30, 50, 0.012, 28.9, "vertolet.jpg", 2800, 5e8),
            "БМП-2": Target("БМП-2", 100, 25, 0.015, 12.0, "bmp2.jpg", 7800, 8e8),
            "Самолет F-16": Target("Самолет F-16", 25, 300, 0.008, 15.0, "f16.png", 2800, 4e8),
            "Корабль (эсминец)": Target("Корабль (эсминец)", 500, 15, 0.02, 100.0, "esmines.jpg", 7800, 8e8),
            "Бункер": Target("Бункер", 1000, 0, 0.5, 50.0,"bunker.jpg", 2500, 3e7)
        }

        self.forma = [
        {'a': 0.66, 'b': 0.7, 'c': 0, 'name': 'Вариант 1'},
        {'a': 0.33, 'b': 0.8, 'c': 0, 'name': 'Вариант 2'},
        {'a': 0.0, 'b': 0.9, 'c': 0, 'name': 'Вариант 3'},
        {'a': 0.66, 'b': 0, 'c': 0, 'name': 'Вариант 4'},
        {'a': 0.33, 'b': 0.1, 'c': 0, 'name': 'Вариант 5'},
        {'a': 0.0, 'b': 0.2, 'c': 0, 'name': 'Вариант 6'},
        {'a': 0.66, 'b': 0.3, 'c': 0, 'name': 'Вариант 7'},
        {'a': 0.33, 'b': 0.4, 'c': 0, 'name': 'Вариант 8'},
        {'a': 0.0, 'b': 0.5, 'c': 0.5, 'name': 'Вариант 9'},
        ]

        # Текущая цель (по умолчанию танк)
        self.current_target = self.targets["Танк Abrams M1A2"]
        self.required_probability = 0.85  # требуемая вероятность поражения

        # Параметры встречи (по умолчанию)
        self.V_c = 600  # скорость ракеты, м/с
        self.V_target = self.current_target.velocity  # скорость цели, м/с
        self.approach_angle = 15  # угол подхода, градусы
        self.meeting_angle = 30  # угол встречи, градусы
        self.H = 0.001  # высота, км (1 м для наземной цели)

    def set_target(self, target_name):
        """Установка текущей цели"""
        if target_name in self.targets:
            self.current_target = self.targets[target_name]
            self.V_target = self.current_target.velocity
            return True
        return False

    def get_atmospheric_conditions(self, H_km):
        """Расчет атмосферных условий на заданной высоте"""
        if H_km <= 11:  # тропосфера
            T = 288.15 - 6.5 * H_km  # температура, K
            p = 101325 * (1 - 0.0065 * H_km / 288.15) ** 5.255  # давление, Па
        else:  # стратосфера
            T = 216.65  # постоянная температура
            p = 22632 * math.exp(-0.000157 * (H_km - 11))  # давление, Па

        # Плотность воздуха
        R = 287.05  # газовая постоянная воздуха
        rho_air = p / (R * T)

        return rho_air, T, p

    def calculate_warhead_geometry(self, a, b, c):
        """Расчет геометрии БЧ по параметрам образующей y = -ax² + bx + c"""
        num_belts = 30  # количество поясов
        belts = []

        for i in range(num_belts):
            x = i * self.L_bch / num_belts
            y = -a * x ** 2 + b * x + c
            belts.append({'x': x, 'y': max(y, 0.01)})  # минимальный радиус 1 см

        return belts

    def calculate_explosive_mass(self, belts):
        """Расчет массы ВВ в БЧ"""
        total_volume = 0
        for i in range(len(belts) - 1):
            # Объем усеченного конуса
            R1 = belts[i]['y']
            R2 = belts[i + 1]['y']
            h = belts[i + 1]['x'] - belts[i]['x']

            volume = (1 / 3) * math.pi * h * (R1 ** 2 + R1 * R2 + R2 ** 2)
            total_volume += volume

        # Вычитаем объем осколков (приближенно)
        fragment_volume = self.calculate_total_fragments_volume(belts)
        explosive_volume = total_volume - fragment_volume

        explosive_volume = max(explosive_volume, 0.001)

        return explosive_volume * self.rho_tnt

    def calculate_total_fragments_volume(self, belts):
        """Расчет общего объема осколков"""
        total_fragments = 0
        for belt in belts:
            # Окружность пояса
            circumference = 2 * math.pi * belt['y']
            # Количество осколков в поясе
            fragments_in_belt = circumference / self.fragment_size
            total_fragments += fragments_in_belt

        return total_fragments * (self.fragment_size ** 3)

    def calculate_fragment_distribution(self, belts):
        """Распределение осколков по поясам"""
        distribution = []
        for belt in belts:
            circumference = 2 * math.pi * belt['y']
            fragments_count = int(circumference / self.fragment_size)
            distribution.append({
                'radius': belt['y'],
                'fragments': fragments_count
            })
        return distribution

    def initial_fragment_velocity(self, m_explosive, m_fragment):
        """Начальная скорость осколков"""
        alpha = m_explosive / m_fragment
        phi = 4.0  # коэффициент для цилиндрической формы
        phi1 = 0.85  # коэффициент потерь

        v_p = 0.5 * self.D_tnt * math.sqrt((phi1 * alpha) / (2 + 4 * alpha / phi))
        return v_p

    def absolute_fragment_velocity(self, v_p, phi_angle):
        """Абсолютная скорость осколков с учетом скорости ракеты"""
        v_o = math.sqrt(self.V_c ** 2 + v_p ** 2 +
                        2 * self.V_c * v_p * math.cos(math.radians(phi_angle)))
        return v_o

    def fragment_velocity_at_distance(self, v_o, R, m_fragment, S_cp, H_km=None):
        """Скорость осколка на расстоянии R с учетом высоты"""
        if H_km is None:
            H_km = self.H

        # Получаем плотность воздуха на текущей высоте
        rho_air, T, p = self.get_atmospheric_conditions(H_km)

        exponent = -self.c_x * rho_air * S_cp * R / (2 * m_fragment)
        v_R = v_o * math.exp(exponent)
        return max(v_R, 0)  # скорость не может быть отрицательной

    def meeting_velocity(self, v_R, meeting_angle=None):
        """Скорость встречи с целью с учетом угла встречи"""
        if meeting_angle is None:
            meeting_angle = self.meeting_angle

        v_B = math.sqrt(self.V_target ** 2 + v_R ** 2 +
                        2 * self.V_target * v_R * math.cos(math.radians(meeting_angle)))
        return v_B

    def kinetic_energy(self, m_fragment, v_B, S_cp):
        """Кинетическая энергия осколка"""
        E = (m_fragment * v_B ** 2) / (2 * S_cp * 1e-4)  # Дж/см²
        return E

    def lethal_energy(self, armor_thickness):
        """Убойная удельная энергия"""
        return 27.5 * armor_thickness * 1000  # Дж/см² (толщина в мм)

    def penetration_thickness(self, v_B, fragment_density, armor_density, armor_strength):
        """Толщина пробиваемой преграды"""
        d1 = self.fragment_size  # диаметр осколка
        if v_B <= 0:
            return 0

        h_s = 0.138 * d1 * fragment_density * (v_B / math.sqrt((armor_strength * armor_density)))
        return h_s

    def fragment_density_at_distance(self, fragments_count, R, phi_i, phi_i1):
        """Плотность потока осколков на расстоянии R"""
        S_belt = 2 * math.pi * R ** 2 * (math.cos(math.radians(phi_i)) - math.cos(math.radians(phi_i1)))
        density = fragments_count / S_belt if S_belt > 0 else 0
        return density

    def target_destruction_probability(self, fragment_density, vulnerable_area):
        """Вероятность поражения цели"""
        P = 1 - math.exp(-fragment_density * vulnerable_area)
        return P

    def calculate_penetration_vs_distance(self, v_o, m_fragment, S_cp, R_values):
        """Расчет толщины пробиваемой преграды в зависимости от расстояния"""
        h_pen_values = []

        for R in R_values:
            v_R = self.fragment_velocity_at_distance(v_o, R, m_fragment, S_cp)
            v_B = self.meeting_velocity(v_R)

            h_pen = self.penetration_thickness(v_B, self.rho_steel,
                                               self.current_target.armor_density,
                                               self.current_target.armor_strength)
            h_pen_values.append(h_pen * 1000)  # в мм

        return h_pen_values

    def calculate_explosive_damage_radius(self, q, delta_p_critical):
        """Расчет радиуса поражения фугасным действием"""
        # Формула Садовского
        R_values = np.linspace(1, 20, 100)
        delta_p_values = []

        for R in R_values:
            delta_p = (1.4 * q / R ** 3 +
                       0.43 * (q ** (2 / 3)) / R ** 2 +
                       0.11 * (q ** (1 / 3)) / R)
            delta_p_values.append(delta_p * 1000)  # в кПа

        # Проверяем условия
        min_pressure = min(delta_p_values)
        max_pressure = max(delta_p_values)

        if min_pressure >= delta_p_critical:
            # Давление ВСЕГДА выше критического
            critical_radius = 20.0  # максимальный радиус расчета
        elif max_pressure < delta_p_critical:
            # Давление НИКОГДА не достигает критического
            critical_radius = 0.0
        else:
            # Ищем точку пересечения
            critical_radius = None
            for i in range(len(R_values) - 1):
                if delta_p_values[i] >= delta_p_critical and delta_p_values[i + 1] < delta_p_critical:
                    critical_radius = R_values[i]
                    break

        if critical_radius is None:
            critical_radius = 0.0

        return R_values, delta_p_values, critical_radius

    def run_comparative_analysis(self, a, b, c, base_height):
        """Сравнительный анализ при разных высотах и углах"""
        st.write(f"\n{'=' * 60}")
        st.write("СРАВНИТЕЛЬНЫЙ АНАЛИЗ ВЛИЯНИЯ ВЫСОТЫ И УГЛОВ ВСТРЕЧИ")
        st.write('=' * 60)

        # Расчет базовых параметров БЧ
        belts = self.calculate_warhead_geometry(a, b, c)
        m_explosive = self.calculate_explosive_mass(belts)
        distribution = self.calculate_fragment_distribution(belts)
        total_fragments = sum([d['fragments'] for d in distribution])
        m_fragment = self.rho_steel * (self.fragment_size ** 3)
        S_cp = 3 * (self.fragment_size ** 2) / 2

        v_p = self.initial_fragment_velocity(m_explosive / 30, m_fragment)
        v_o = self.absolute_fragment_velocity(v_p, self.approach_angle)

        R_values = np.linspace(1, 20, 50)

        # Сравнение разных углов при одинаковой высоте
        self.compare_different_angles(R_values, v_o, m_fragment, S_cp, base_height, total_fragments)

    def compare_different_heights(self, R_values, v_o, m_fragment, S_cp, meeting_angle, total_fragments):
        """Сравнение влияния разных высот"""
        heights = [0.001, 0.5, 5.0, 10.0]  # разные высоты
        height_names = ['1 м', '0.5 км', '5 км', '10 км']
        colors = ['red', 'blue', 'green', 'purple']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Влияние высоты на эффективность БЧ (Угол встречи: {meeting_angle}°)',
                     fontsize=16, fontweight='bold')

        for i, height in enumerate(heights):
            v_R_values = []
            E_values = []
            P_values = []
            density_values = []

            for R in R_values:
                v_R = self.fragment_velocity_at_distance(v_o, R, m_fragment, S_cp, height)
                v_B = self.meeting_velocity(v_R, meeting_angle)
                E = self.kinetic_energy(m_fragment, v_B, S_cp)

                fragment_density = total_fragments / (4 * math.pi * R ** 2)
                P = self.target_destruction_probability(fragment_density, self.current_target.area * 0.1)

                v_R_values.append(v_R)
                E_values.append(E)
                P_values.append(P)
                density_values.append(fragment_density)

            # График 1: Скорость осколков
            ax1.plot(R_values, v_R_values, color=colors[i], linewidth=2, label=f'{height_names[i]}')
            ax1.set_xlabel('Расстояние R, м')
            ax1.set_ylabel('Скорость осколков, м/с')
            ax1.set_title('Скорость осколков от расстояния')
            ax1.grid(True)
            ax1.legend()

            # График 2: Энергия осколков
            ax2.plot(R_values, E_values, color=colors[i], linewidth=2, label=f'{height_names[i]}')
            ax2.set_xlabel('Расстояние R, м')
            ax2.set_ylabel('Удельная энергия, Дж/см²')
            ax2.set_title('Энергия осколков от расстояния')
            ax2.grid(True)
            ax2.legend()

            # График 3: Плотность потока
            ax3.plot(R_values, density_values, color=colors[i], linewidth=2, label=f'{height_names[i]}')
            ax3.set_xlabel('Расстояние R, м')
            ax3.set_ylabel('Плотность потока, шт/м²')
            ax3.set_title('Плотность потока от расстояния')
            ax3.set_yscale('log')
            ax3.grid(True)
            ax3.legend()

            # График 4: Вероятность поражения
            ax4.plot(R_values, P_values, color=colors[i], linewidth=2, label=f'{height_names[i]}')
            ax4.axhline(y=self.required_probability, color='r', linestyle='--',
                        label=f'Требуемая P={self.required_probability}')
            ax4.set_xlabel('Расстояние R, м')
            ax4.set_ylabel('Вероятность поражения')
            ax4.set_title('Вероятность поражения от расстояния')
            ax4.grid(True)
            ax4.legend()

        st.pyplot(fig)

    def compare_different_angles(self, R_values, v_o, m_fragment, S_cp, height, total_fragments):
        """Сравнение влияния разных углов встречи"""
        angles = [0, 30, 60, 90]  # разные углы встречи
        angle_names = ['Лобовая 0°', 'Боковая 30°', 'Сбоку 60°', 'Сзади 90°']
        colors = ['red', 'blue', 'green', 'purple']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Влияние угла встречи на эффективность БЧ (Высота: {height} км)',
                     fontsize=16, fontweight='bold')

        for i, angle in enumerate(angles):
            v_R_values = []
            E_values = []
            P_values = []
            density_values = []

            for R in R_values:
                v_R = self.fragment_velocity_at_distance(v_o, R, m_fragment, S_cp, height)
                v_B = self.meeting_velocity(v_R, angle)
                E = self.kinetic_energy(m_fragment, v_B, S_cp)

                fragment_density = total_fragments / (4 * math.pi * R ** 2)
                P = self.target_destruction_probability(fragment_density, self.current_target.area * 0.1)

                v_R_values.append(v_R)
                E_values.append(E)
                P_values.append(P)
                density_values.append(fragment_density)

            # График 1: Скорость встречи
            ax1.plot(R_values, v_R_values, color=colors[i], linewidth=2, label=f'{angle_names[i]}')
            ax1.set_xlabel('Расстояние R, м')
            ax1.set_ylabel('Скорость осколков, м/с')
            ax1.set_title('Скорость осколков от расстояния')
            ax1.grid(True)
            ax1.legend()

            # График 2: Энергия осколков
            ax2.plot(R_values, E_values, color=colors[i], linewidth=2, label=f'{angle_names[i]}')
            ax2.set_xlabel('Расстояние R, м')
            ax2.set_ylabel('Удельная энергия, Дж/см²')
            ax2.set_title('Энергия осколков от расстояния')
            ax2.grid(True)
            ax2.legend()

            # График 3: Относительная скорость встречи
            meeting_velocities = [self.meeting_velocity(v_R, angle) for v_R in v_R_values]
            ax3.plot(R_values, meeting_velocities, color=colors[i], linewidth=2, label=f'{angle_names[i]}')
            ax3.set_xlabel('Расстояние R, м')
            ax3.set_ylabel('Скорость встречи, м/с')
            ax3.set_title('Скорость встречи от расстояния')
            ax3.grid(True)
            ax3.legend()

            # График 4: Вероятность поражения
            ax4.plot(R_values, P_values, color=colors[i], linewidth=2, label=f'{angle_names[i]}')
            ax4.axhline(y=self.required_probability, color='r', linestyle='--',
                        label=f'Требуемая P={self.required_probability}')
            ax4.set_xlabel('Расстояние R, м')
            ax4.set_ylabel('Вероятность поражения')
            ax4.set_title('Вероятность поражения от расстояния')
            ax4.grid(True)
            ax4.legend()

        st.pyplot(fig)

    def run_calculation(self, a=0.66, b=0.7, c=0, config_name="Базовая"):
        """Основной расчет"""
        st.write(f"\n{'=' * 60}")
        st.write(f"РАСЧЕТ ЭФФЕКТИВНОСТИ ОСКОЛОЧНОЙ БЧ - {config_name.upper()}")
        st.write(f"Цель: {self.current_target.name}")
        st.write(f"Параметры образующей: a={a}, b={b}, c={c}")
        st.write(f"Высота: {self.H} км, Угол встречи: {self.meeting_angle}°")
        st.write('=' * 60)

        # Атмосферные условия
        rho_air, T, p = self.get_atmospheric_conditions(self.H)
        st.write(f"Атмосферные условия на высоте {self.H} км:")
        st.write(f"Плотность воздуха: {rho_air:.4f} кг/м³")
        st.write(f"Температура: {T:.1f} K")
        st.write(f"Давление: {p / 1000:.1f} кПа")

        # 1. Расчет геометрии БЧ
        belts = self.calculate_warhead_geometry(a, b, c)
        st.write(f"Количество поясов: {len(belts)}")

        # 2. Расчет массы ВВ
        m_explosive = self.calculate_explosive_mass(belts)
        st.write(f"Масса ВВ: {m_explosive:.2f} кг")

        if m_explosive <= 0:
            st.error("ОШИБКА: Масса ВВ меньше или равна 0!")

            # Возвращаем пустые результаты чтобы избежать дальнейших ошибок
            return {
                'explosive_mass': 0,
                'total_fragments': 0,
                'critical_radius_explosive': 0,
                'critical_radius_fragments': None,
                'max_penetration': 0,
                'geometry_params': {'a': a, 'b': b, 'c': c},
                'v_o': 0,
                'v_p': 0,
                'R_values': [],
                'h_pen_values': [],
                'graphs': {},
                'results_text': "Расчет прерван: некорректная масса ВВ"
            }

        # 3. Распределение осколков
        distribution = self.calculate_fragment_distribution(belts)
        total_fragments = sum([d['fragments'] for d in distribution])
        st.write(f"Общее количество осколков: {total_fragments}")

        # 4. Параметры осколка
        m_fragment = self.rho_steel * (self.fragment_size ** 3)
        S_cp = 3 * (self.fragment_size ** 2) / 2  # средняя площадь

        # 5. Расчет скоростей
        v_p = self.initial_fragment_velocity(m_explosive / 30, m_fragment)  # для одного пояса
        v_o = self.absolute_fragment_velocity(v_p, self.approach_angle)

        st.write(f"Начальная скорость осколков: {v_p:.1f} м/с")
        st.write(f"Абсолютная скорость осколков: {v_o:.1f} м/с")

        # 6. Расчет характеристик на различных расстояниях
        R_values = np.linspace(1, 20, 50)
        v_R_values = []
        E_values = []
        P_values = []
        h_pen_values = []
        density_values = []

        for R in R_values:
            v_R = self.fragment_velocity_at_distance(v_o, R, m_fragment, S_cp)
            v_B = self.meeting_velocity(v_R)
            E = self.kinetic_energy(m_fragment, v_B, S_cp)

            # Плотность потока (усредненная)
            fragment_density = total_fragments / (4 * math.pi * R ** 2)
            density_values.append(fragment_density)
            P = self.target_destruction_probability(fragment_density, self.current_target.area * 0.1)

            # Толщина пробития
            h_pen = self.penetration_thickness(v_B, self.rho_steel,
                                               self.current_target.armor_density,
                                               self.current_target.armor_strength)

            v_R_values.append(v_R)
            E_values.append(E)
            P_values.append(P)
            h_pen_values.append(h_pen)  # в мм

        # 6.5 Расчет толщины пробития в зависимости от расстояния
        h_pen_values = self.calculate_penetration_vs_distance(v_o, m_fragment, S_cp, R_values)

        # 7. Расчет фугасного действия
        R_exp, delta_p, critical_radius = self.calculate_explosive_damage_radius(
            m_explosive, self.current_target.critical_pressure)

        # 8. Построение графиков
        graphs = self.plot_additional_graphs(R_values, v_R_values, E_values, P_values, density_values, R_exp, delta_p,
                                             config_name, h_pen_values)

        # 9. Анализ результатов
        results = self.analyze_results(R_values, P_values, critical_radius, h_pen_values, config_name)

        return {
            'explosive_mass': m_explosive,
            'total_fragments': total_fragments,
            'critical_radius_explosive': critical_radius,
            'critical_radius_fragments': self.find_critical_radius(R_values, P_values),
            'max_penetration': max(h_pen_values) if h_pen_values else 0,
            'geometry_params': {'a': a, 'b': b, 'c': c},
            'v_o': v_o,
            'v_p': v_p,
            'R_values': R_values,
            'h_pen_values': h_pen_values,
            'graphs': graphs,
            'results_text': results
        }

    def find_critical_radius(self, R_values, P_values):
        """Нахождение критического радиуса для осколков"""
        for i in range(len(R_values)):
            if P_values[i] >= self.required_probability:
                return R_values[i]
        return None

    def analyze_results(self, R_values, P_values, critical_radius_exp, h_pen_values, config_name):
        """Анализ результатов расчета"""
        critical_radius_frag = self.find_critical_radius(R_values, P_values)

        results_text = f"\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ({config_name.upper()}) ===\n"
        results_text += f"Цель: {self.current_target.name}\n"
        results_text += f"Высота: {self.H} км, Угол встречи: {self.meeting_angle}°\n"
        results_text += f"Радиус зоны поражения фугасным действием: {critical_radius_exp:.1f} м\n"

        if critical_radius_frag:
            results_text += f"Радиус поражения осколками (P≥{self.required_probability}): {critical_radius_frag:.1f} м\n"
        else:
            results_text += f"Требуемая вероятность поражения {self.required_probability} не достигается\n"

        # Анализ пробивной способности
        max_penetration = max(h_pen_values)
        results_text += f"Максимальная толщина пробития: {max_penetration:.1f} мм\n"

        if max_penetration < self.current_target.armor_thickness * 1000:
            results_text += "ВЫВОД: Осколки не способны пробить броню цели\n"
            results_text += "Рекомендация: Использование осколочной БЧ против данной цели неэффективно\n"
        else:
            results_text += "ВЫВОД: Осколки способны пробить броню\n"

        st.write(results_text)
        return results_text

    def plot_additional_graphs(self, R_values, v_R_values, E_values, P_values, density_values, R_exp, delta_p,
                               config_name, h_pen_values):
        """Построение дополнительных графиков"""
        graphs = {}

        # График 1: Избыточное давление от расстояния
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(R_exp, delta_p, 'red', linewidth=2)
        ax1.axhline(y=self.current_target.critical_pressure, color='r', linestyle='--',
                    label=f'Крит. давление {self.current_target.critical_pressure} кПа')
        ax1.set_xlabel('Расстояние R, м')
        ax1.set_ylabel('Избыточное давление, кПа')
        ax1.set_title('Избыточное давление от расстояния')
        ax1.grid(True)
        ax1.legend()
        graphs['pressure_graph'] = fig1
        st.pyplot(fig1)

        # График 2: Плотность потока от расстояния
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(R_values, density_values, 'orange', linewidth=2)
        ax2.set_xlabel('Расстояние R, м')
        ax2.set_ylabel('Плотность потока осколков, шт/м²')
        ax2.set_title('Плотность потока осколков от расстояния')
        ax2.set_yscale('log')
        ax2.grid(True)
        graphs['density_graph'] = fig2
        st.pyplot(fig2)

        # График 3: Скорость осколков от расстояния
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.plot(R_values, v_R_values, 'blue', linewidth=2)
        ax3.set_xlabel('Расстояние R, м')
        ax3.set_ylabel('Скорость осколков, м/с')
        ax3.set_title('Скорость осколков от расстояния')
        ax3.grid(True)
        graphs['fragment_velocity_graph'] = fig3
        st.pyplot(fig3)

        # График 4: Энергия осколков от расстояния
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.plot(R_values, E_values, 'green', linewidth=2)
        ax4.set_xlabel('Расстояние R, м')
        ax4.set_ylabel('Удельная энергия, Дж/см²')
        ax4.set_title('Энергия осколков от расстояния')
        ax4.grid(True)
        graphs['graph_energy_fragments'] = fig4
        st.pyplot(fig4)

        # График 5: Вероятность поражения
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        ax5.plot(R_values, P_values, 'purple', linewidth=2)
        ax5.axhline(y=self.required_probability, color='r', linestyle='--',
                    label=f'Требуемая вероятность P={self.required_probability}')
        ax5.set_xlabel('Расстояние R, м')
        ax5.set_ylabel('Вероятность поражения')
        ax5.set_title(
            f'Вероятность поражения от расстояния - {config_name}\nЦель: {self.current_target.name}\n(Высота: {self.H} км, Угол встречи: {self.meeting_angle}°)')
        ax5.grid(True)
        ax5.legend()
        graphs['probability_defeat_graph'] = fig5
        st.pyplot(fig5)

        # График 6: Толщина пробития
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.plot(R_values, h_pen_values, 'brown', linewidth=2, label='Толщина пробития')
        ax6.axhline(y=self.current_target.armor_thickness * 1000, color='r', linestyle='--',
                    label=f'Толщина брони цели: {self.current_target.armor_thickness * 1000:.1f} мм')
        ax6.set_xlabel('Расстояние R, м')
        ax6.set_ylabel('Толщина пробития, мм')
        ax6.set_title(f'Зависимость толщины пробития от расстояния\nЦель: {self.current_target.name}')
        ax6.grid(True)
        ax6.legend()
        graphs['penetration_graph'] = fig6
        st.pyplot(fig6)

        return graphs


def plot_to_base64(fig):
    """Конвертирует matplotlib figure в base64 строку"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str


def run():
    """Основная функция Streamlit приложения"""
    st.set_page_config(page_title="Расчет эффективности ОБЧ", layout="wide")
    image = os.path.join("media", "mem.jpg")
    st.image(image)
    gif = os.path.join("media", "mem_gif.gif")
    st.image(gif)

    st.title("🧨 Программа расчета эффективности осколочных боевых частей")
    st.markdown("---")

    # Создание объекта
    warhead = FragmentationWarhead()

    # Сайдбар для ввода данных
    with st.sidebar:
        st.header("Параметры расчета")

        # Личные данные студента
        st.subheader("Данные студента")
        student_name = st.text_input("ФИО студента", "Иванов И.И.")
        student_group = st.text_input("Группа", "И912С")
        variant = st.number_input("Вариант", min_value=1, max_value=30, value=1)

        # Выбор цели
        st.subheader("Выбор цели")
        target_options = list(warhead.targets.keys())
        selected_target = st.selectbox("Выберите цель", target_options, index=0)
        warhead.set_target(selected_target)

        # Параметры образующей
        st.subheader("Параметры образующей БЧ")
        forma = warhead.forma
        names = [item['name'] for item in forma]
        selected_name = st.selectbox("Выберите форму по варианту", names, index=0)
        selected_forma = next(item for item in forma if item['name'] == selected_name)
        col1, col2, col3 = st.columns(3)
        with col1:
            a = st.number_input("Параметр a", value=float(selected_forma['a']), step=0.01)
        with col2:
            b = st.number_input("Параметр b", value=float(selected_forma['b']), step=0.01)
        with col3:
            c = st.number_input("Параметр c", value=float(selected_forma['c']), step=0.01)

        # Параметры встречи
        st.subheader("Параметры встречи")
        H = st.selectbox("Высота, км", [0.001, 0.5, 5.0, 10.0], format_func=lambda x: f"{x} км")
        beta = st.number_input("Угол подхода β, град", value=15, min_value=0, max_value=90)
        Vc = st.number_input("Скорость ракеты Vc, м/с", value=600, min_value=100, max_value=1000)

        # Требуемая вероятность
        probability_defeat = st.slider("Требуемая вероятность поражения", 0.5, 1.0, 0.85, 0.05)

        # Кнопка расчета
        calculate_btn = st.button("🚀 Выполнить расчет", type="primary")

    # Основная область
    if calculate_btn:
        # Настройка параметров
        warhead.H = H
        warhead.V_c = Vc
        warhead.approach_angle = beta
        warhead.required_probability = probability_defeat

        # Выполнение расчета
        with st.spinner("Выполняется расчет..."):
            results = warhead.run_calculation(a=a, b=b, c=c, config_name=f"Вариант {variant}")

        # Сравнительный анализ
        st.markdown("---")
        st.header("📊 Сравнительный анализ")
        warhead.run_comparative_analysis(a, b, c, H)

        # Итоговые результаты
        st.markdown("---")
        st.header("🎯 Итоговые результаты")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Масса ВВ", f"{results['explosive_mass']:.2f} кг")
            st.metric("Количество осколков", f"{results['total_fragments']}")
        with col2:
            st.metric("Радиус фугасного поражения",
                      f"{results['critical_radius_explosive']:.1f} м" if results[
                          'critical_radius_explosive'] else "не достигается")
            st.metric("Макс. толщина пробития", f"{results['max_penetration']:.1f} мм")
        with col3:
            st.metric("Радиус осколочного поражения",
                      f"{results['critical_radius_fragments']:.1f} м" if results[
                          'critical_radius_fragments'] else "не достигается")
            armor_status = "способны" if results[
                                                'max_penetration'] >= warhead.current_target.armor_thickness * 1000 else "не способны"
            st.metric("Пробитие брони", armor_status)

            # Генерация отчета
            st.markdown("---")
            st.header("📄 Генерация отчета")

            try:
                st.info("🔄 Начинаю генерацию отчета...")

                # Проверяем шаблон
                template_path = "pattern_laba3.docx"
                st.info(f"🔍 Проверяю наличие шаблона: {template_path}")

                if not os.path.exists(template_path):
                    st.error(f"❌ Файл шаблона '{template_path}' не найден!")
                    st.info("📁 Содержимое текущей директории:")
                    for file in os.listdir('.'):
                        st.write(f" - {file}")


                st.success("✅ Шаблон найден")

                # Создаем буферы для графиков
                st.info("🔄 Подготавливаю графики...")
                graph_buffers = {}

                for graph_name, fig in results['graphs'].items():
                    buf = BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    graph_buffers[graph_name] = buf
                    plt.close(fig)

                st.success(f"✅ Графики подготовлены: {len(graph_buffers)} шт.")

                # Загружаем шаблон
                st.info("📄 Загружаю шаблон...")
                doc = DocxTemplate(template_path)

                # Создаем контекст
                st.info("🔄 Создаю контекст данных...")
                context = {
                    # Текстовые данные
                    'name': student_name,
                    'target': selected_target,
                    'probability_defeat': probability_defeat,
                    'var': variant,
                    'H': H,
                    'beta': beta,
                    'Vc': Vc,
                    'V_target': warhead.current_target.velocity,
                    'h': warhead.current_target.armor_thickness * 1000,
                    'S': warhead.current_target.area,
                    'critical_pressure': warhead.current_target.critical_pressure,
                    'a': a,
                    'b': b,
                    'c': c,
                    'armor_status': armor_status,
                    'm_explosive': round(results['explosive_mass'], 2),
                    'critical_radius_exp': round(results['critical_radius_explosive'], 2) if results[
                        'critical_radius_explosive'] else "20",
                    'distance_probability_detonation': round(results['critical_radius_fragments'], 2) if results[
                        'critical_radius_fragments'] else "не достигается",
                }

                # Добавляем графики в контекст
                st.info("📊 Добавляю графики в отчет...")

                # Список графиков для отчета
                graph_mapping = {
                    'pressure_graph': 'pressure_graph',
                    'density_graph': 'density_graph',
                    'fragment_velocity_graph': 'fragment_velocity_graph',
                    'graph_energy_fragments': 'graph_energy_fragments',
                    'probability_defeat_graph': 'probability_defeat_graph',
                    'penetration_graph': 'penetration_graph'
                }

                for template_var, graph_key in graph_mapping.items():
                    try:
                        if graph_key in graph_buffers:
                            # Создаем InlineImage для каждого графика
                            context[graph_key] = InlineImage(
                                doc,
                                graph_buffers[graph_key],
                                width=Mm(150)  # Ширина 150 мм
                            )
                            st.success(f"✅ {graph_key} добавлен")
                        else:
                            st.warning(f"⚠️ {graph_key} не найден в результатах")
                    except Exception as e:
                        st.error(f"❌ Ошибка при добавлении {graph_key}: {str(e)}")

                target_picture_name = warhead.current_target.picture
                target_picture_path = f"media/{target_picture_name}"

                with open(target_picture_path, "rb") as img_file:
                    picture_buffer = BytesIO(img_file.read())

                # Создаем InlineImage для картинки цели
                context['pic_target'] = InlineImage(doc, picture_buffer, width=Mm(120))

                # Рендерим документ
                st.info("🎨 Формирую отчет...")
                doc.render(context)
                if 'picture_buffer' in locals():
                    picture_buffer.close()

                # Сохраняем в буфер
                st.info("💾 Сохраняю отчет...")
                output_buffer = BytesIO()
                doc.save(output_buffer)
                output_buffer.seek(0)

                # Закрываем буферы графиков
                for buf in graph_buffers.values():
                    buf.close()

                st.success("✅ Отчет готов!")

                # Кнопка скачивания
                st.download_button(
                    label="📥 Скачать отчет DOCX",
                    data=output_buffer,
                    file_name=f"lab_report_{student_name.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )

            except Exception as e:
                st.error(f"❌ Ошибка при генерации отчета: {str(e)}")
                import traceback

                st.code(traceback.format_exc())
