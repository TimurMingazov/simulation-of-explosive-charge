import math
import streamlit as st
import os

# Таблица 1 (стр. 6): металлы
# ρ: 10^3 kg/m^3, k: W/mK, c: J/kgK, Tm, Tb: °C, Lm: 10^6 J/kg, Lv: 10^6 J/kg, R: -
METALS = {
    "Al": {"rho": 2.7e3, "k": 233, "c": 920, "Tm": 660,  "Tb": 2447, "Lm": 0.396e6, "Lv": 10.9e6, "R": 0.93},
    "Au": {"rho": 19.3e3, "k": 300, "c": 128, "Tm": 1063, "Tb": 2700, "Lm": 0.065e6, "Lv": 1.65e6, "R": 0.98},
    "Bi": {"rho": 9.7e3, "k": 8,   "c": 122, "Tm": 271,  "Tb": 1559, "Lm": 0.052e6, "Lv": 0.72e6, "R": 0.55},
    "Cr": {"rho": 7.1e3, "k": 45,  "c": 460, "Tm": 1903, "Tb": 2642, "Lm": 0.280e6, "Lv": 6.71e6, "R": 0.57},
    "Cu": {"rho": 8.2e3, "k": 373, "c": 386, "Tm": 1083, "Tb": 2595, "Lm": 0.205e6, "Lv": 4.78e6, "R": 0.91},
    "Fe": {"rho": 7.9e3, "k": 50,  "c": 450, "Tm": 1535, "Tb": 2900, "Lm": 0.278e6, "Lv": 7.0e6,  "R": 0.63},
    "Ni": {"rho": 8.7e3, "k": 68,  "c": 440, "Tm": 1453, "Tb": 2800, "Lm": 0.303e6, "Lv": 6.48e6, "R": 0.74},
    "V":  {"rho": 6.0e3, "k": 31,  "c": 514, "Tm": 1730, "Tb": 3400, "Lm": 0.344e6, "Lv": 8.98e6, "R": 0.55},
    "W":  {"rho": 19.3e3,"k": 190, "c": 105, "Tm": 3380, "Tb": 5530, "Lm": 0.191e6, "Lv": 4.35e6, "R": 0.68},
}

# Таблица 2 (стр. 6): неметаллы (для a=k/(ρc) нам хватает ρ,k,c и Tp + A)
# ρ: 10^3 kg/m^3, k: W/mK, c: 10^3 J/kgK, Tp: °C, A: -
NONMETALS = {
    "бумага":  {"rho": 0.7e3, "k": 0.25, "c": 1.5e3, "Tp": 600,  "A": 0.55},
    "фанера":  {"rho": 0.9e3, "k": 0.5,  "c": 2.5e3, "Tp": 1800, "A": 0.8},
    "керамика":{"rho": 1.5e3, "k": 0.8,  "c": 0.6e3, "Tp": 1500, "A": 0.9},
    "пластмасса":{"rho": 2.2e3,"k": 0.5,  "c": 0.6e3, "Tp": 600,  "A": 0.5},
    "стекло":  {"rho": 2.3e3, "k": 0.75, "c": 0.8e3, "Tp": 1700, "A": 0.9},
    # В таблице A для резины не заполнено — берем типовое допущение 0.5 (можно поменять в UI)
    "резина":  {"rho": 1.2e3, "k": 0.15, "c": 0.9e3, "Tp": 900,  "A": 0.5},
}

# Практика (стр. 4–5): варианты
TASK1_VARIANTS = {
    i+1: {"h_um": (i+1)*5, "P_W": 100*(i+2), "r0_um": (i+1)*5}  # 5..50, 200..1100, 5..50
    for i in range(10)
}
TASK2_VARIANTS = {
    i+1: {"h_mm": 0.10 + 0.02*i, "tau_s": (i+1)*1e-4}  # 0.10..0.28, 1e-4..1e-3
    for i in range(10)
}
TASK3_VARIANTS = {
    1: {"material": "бумага",   "V_cm_s": 500, "r0_um": 500},
    2: {"material": "фанера",   "V_cm_s": 2,   "r0_um": 500},
    3: {"material": "стекло",   "V_cm_s": 2.5, "r0_um": 500},
    4: {"material": "керамика", "V_cm_s": 5,   "r0_um": 500},
    5: {"material": "резина",   "V_cm_s": 2.5, "r0_um": 500},
    6: {"material": "Fe",       "V_cm_s": 3,   "r0_um": 200},
    7: {"material": "Al",       "V_cm_s": 10,  "r0_um": 200},
    8: {"material": "Cr",       "V_cm_s": 2.5, "r0_um": 200},
    9: {"material": "W",        "V_cm_s": 2,   "r0_um": 200},
    10:{"material": "Cu",       "V_cm_s": 5,   "r0_um": 200},
}
TASK4_VARIANTS = {
    1: {"h_um": 150}, 2: {"h_um": 170}, 3: {"h_um": 200}, 4: {"h_um": 140}, 5: {"h_um": 130},
    6: {"h_um": 160}, 7: {"h_um": 190}, 8: {"h_um": 210}, 9: {"h_um": 180}, 10: {"h_um": 220},
}
TASK4_FIXED = {"P_W": 100.0, "r0_mm": 0.3}

# =========================
# Вспомогательные функции
# =========================

def thermal_diffusivity(rho, k, c):
    """a = k/(rho*c), SI: m^2/s"""
    return k / (rho * c)

def power_density(P, r0_m):
    """q0 = P/(pi r0^2), W/m^2"""
    return P / (math.pi * r0_m**2)

def solve_task1_max_speed_cut_cu(h_um, P_W, r0_um):
    """
    Используем формулу (6) (стр. 2 PDF) для случая прогрева по толщине:
    T ≈ (q * r0^2)/(2 k h) * ln(2.25 a/(r0 V)) + T_amb
    где q = q0*(1-R), q0 = P/(pi r0^2)
    Решаем относительно V.
    """
    cu = METALS["Cu"]
    r0 = r0_um * 1e-6
    h = h_um * 1e-6

    a = thermal_diffusivity(cu["rho"], cu["k"], cu["c"])
    q0 = power_density(P_W, r0)
    q = q0 * (1.0 - cu["R"])

    T_target = cu["Tb"]
    dT = T_target
    if dT <= 0:
        return None, "Целевая температура ниже/равна комнатной."

    # ln_term = dT * 2 k h / (q r0^2)
    ln_term = dT * 2.0 * cu["k"] * h / (q * r0**2)

    # V = 2.25 a / (r0 * exp(ln_term))
    V = 2.25 * a / (r0 * math.exp(ln_term))
    return V, None

def classify_source(V_m_s, r0_m, a_m2_s):
    crit = V_m_s * r0_m / a_m2_s
    kind = "медленно движущийся" if crit <= 1 else "быстро движущийся"
    return crit, kind

def threshold_q0_for_cut(material_key, V_m_s, r0_m):
    """
    Для задачи 3:
    - критерий: V*r0/a (стр. 2)
    - порог q0 из (5) для медленно движущегося (с +Tн)
      T ≈ q0*(1-R)*r0/k * (1 - V r0/(4a)) + Tн  (металлы)
      Для неметаллов вместо (1-R) используем A
    - для быстро движущегося используем (7) (с +Tн)
      T ≈ (2/pi) * q0*Abs/k * sqrt(2 a r0 / V) + Tн
    """
    if material_key in METALS:
        m = METALS[material_key]
        rho, k, c = m["rho"], m["k"], m["c"]
        Abs = 1.0 - m["R"]
        T_target = m["Tm"]
    else:
        m = NONMETALS[material_key]
        rho, k, c = m["rho"], m["k"], m["c"]
        Abs = m["A"]
        T_target = m["Tp"]

    a = thermal_diffusivity(rho, k, c)
    crit, kind = classify_source(V_m_s, r0_m, a)
    dT = T_target - T_AMB
    if dT <= 0:
        return None, crit, kind, "Целевая температура ниже/равна комнатной."

    if kind == "медленно движущийся":
        denom = Abs * r0_m / k * (1.0 - (V_m_s * r0_m) / (4.0 * a))
        if denom <= 0:
            return None, crit, kind, "Деление на 0/отрицательное в формуле (5)."
        q0 = dT / denom  # W/m^2
    else:
        denom = (2.0 / math.pi) * (Abs / k) * math.sqrt(2.0 * a * r0_m / V_m_s)
        if denom <= 0:
            return None, crit, kind, "Деление на 0/отрицательное в формуле (7)."
        q0 = dT / denom  # W/m^2

    return q0, crit, kind, None

def solve_task2_weld_range_qabs(h_mm, tau_s):
    """
    Задача 2 (упрощенная энергетическая оценка):
    q_abs * tau = (ρ_Au*h + ρ_Cr*h) * c_eq * (T - T0) + ρ_Au*h*Lm_Au + ρ_Cr*h*Lm_Cr
    Берем:
      q_min: чтобы довести обе фольги до Tm(Cr) и расплавить Cr (и Au тоже расплавится по пути)
      q_max: чтобы не довести до Tb(min(Au,Cr)) (без учета теплоты испарения)
    Это приближение; в тексте задачи теплообмен "после расплавления" учитывается качественно.
    """
    au, cr = METALS["Au"], METALS["Cr"]
    h = h_mm * 1e-3

    # энергия на единицу площади (J/m^2)
    def area_energy_to(T):
        # нагрев обеих пластин до T + плавление обеих (для сварки) — можно менять логику
        E_heat = au["rho"] * h * au["c"] * (T - T_AMB) + cr["rho"] * h * cr["c"] * (T - T_AMB)
        E_melt = au["rho"] * h * au["Lm"] + cr["rho"] * h * cr["Lm"]
        return E_heat + E_melt

    # q_abs = E/tau
    Tm_cr = cr["Tm"]
    Tb_min = min(au["Tb"], cr["Tb"])

    qmin = area_energy_to(Tm_cr) / tau_s  # W/m^2
    qmax = (au["rho"] * h * au["c"] * (Tb_min - T_AMB) + cr["rho"] * h * cr["c"] * (Tb_min - T_AMB)) / tau_s

    return qmin, qmax, Tm_cr, Tb_min

def solve_task4_chrome_engraving(h_um, P_W, r0_mm):
    """
    Задача 4: оценка по энергии на испарение слоя толщины h.
    Время воздействия на точку ~ 2 r0 / V.
    Энергия на площадь, доставленная: E = q_abs * (2r0/V) = (P_abs/(pi r0^2))*(2r0/V)=2P_abs/(pi r0 V)
    Требуемая энергия на площадь для слоя h:
      E_req = ρ h [ c(Tb-T0) + Lm + Lv ]
    Тогда V = 2 P_abs / (pi r0 E_req)
    """
    cr = METALS["Cr"]
    r0 = r0_mm * 1e-3
    h = h_um * 1e-6

    P_abs = P_W * (1.0 - cr["R"])

    E_req = cr["rho"] * h * (cr["c"] * (cr["Tb"] - T_AMB) + cr["Lm"] + cr["Lv"])  # J/m^2
    if E_req <= 0:
        return None, "Некорректная энергия."

    V = 2.0 * P_abs / (math.pi * r0 * E_req)  # m/s
    return V, None


# =========================
# UI Streamlit
# =========================
def run():
    st.set_page_config(page_title="Практика 3 — Лазерная микрообработка", layout="wide")
    st.title("Практика 3: Лазерная микрообработка материалов и сварка")
    st.caption("Формулы и исходные данные взяты из PDF (стр. 2–6).")

    cat0 = os.path.join("media", "cat0.jpg")
    st.image(cat0)
    cat1 = os.path.join("media", "cat1.jpg")
    st.image(cat1)

    task = st.sidebar.selectbox("Выберите задачу", ["Задача 1", "Задача 2", "Задача 3", "Задача 4"])
    variant = st.sidebar.selectbox("Вариант", list(range(1, 11)))

    st.sidebar.markdown("---")

    if task == "Задача 1":
        st.header("Задача 1 — максимальная скорость резки Cu-фольги")
        v = TASK1_VARIANTS[variant]


        st.subheader("Дано")
        st.write(f"- Материал: медь (Cu)")
        st.write(f"- Толщина фольги: **h = {v['h_um']} мкм**")
        st.write(f"- Мощность: **P = {v['P_W']} Вт**")
        st.write(f"- Радиус пятна: **r0 = {v['r0_um']} мкм**")

        st.subheader("Найти")
        st.write("- Максимальную скорость резки **Vск** (оценка)")

        V, err = solve_task1_max_speed_cut_cu(v["h_um"], v["P_W"], v["r0_um"])

        st.subheader("Решение")
        st.write("Используем формулу для прогретой по толщине пластины:")
        st.write("T ≈ (q * r0^2) / (2 * k * h) * ln(2.25 * a / (r0 * V)) + Tн")
        st.write("где:")
        st.write("q = P / (π * r0^2) * (1 - R)")
        st.write("a = k / (ρ * c)")
        st.write("Решаем относительно V.")

        st.subheader("Ответ")
        if err:
            st.error(err)
        else:
            st.success(f"Vск ≈ **{V:.3e} м/с**  (≈ {V*100:.3e} см/с)")

    elif task == "Задача 2":
        st.header("Задача 2 — диапазон плотности мощности для сварки Au–Cr")
        v = TASK2_VARIANTS[variant]

        st.subheader("Дано")
        st.write("- Материалы: золото (Au) и хром (Cr), сварка «встык»")
        st.write(f"- Толщина фольги: **h = {v['h_mm']:.2f} мм**")
        st.write(f"- Длительность импульса: **τ = {v['tau_s']:.2e} c**")

        st.subheader("Найти")
        st.write("- Диапазон допустимых плотностей мощности (оценка): **q_min … q_max**")

        qmin, qmax, Tm_cr, Tb_min = solve_task2_weld_range_qabs(v["h_mm"], v["tau_s"])

        st.subheader("Решение")
        st.write("Энергетическая оценка на единицу площади (упрощение):")
        st.write(f"- минимум: довести обе фольги до Tпл(Cr) = {Tm_cr}°C и расплавить (Au и Cr)")
        st.write(f"- максимум: не довести до Tкип,min = {Tb_min}°C (без учета испарения)")
        st.write("Формула: q_abs = E / τ")
        st.info("Это приближенная оценка (теплопроводность/геометрия шва упрощены).")

        st.subheader("Ответ")
        st.success(f"q_min(abs) ≈ **{qmin:.3e} Вт/м²**")
        st.success(f"q_max(abs) ≈ **{qmax:.3e} Вт/м²**")

    elif task == "Задача 3":
        st.header("Задача 3 — критерий быстрый/медленный источник и порог q0 для резки")
        v = TASK3_VARIANTS[variant]
        mat = v["material"]

        st.subheader("Дано")
        st.write(f"- Материал: **{mat}**")
        st.write(f"- Скорость сканирования: **Vск = {v['V_cm_s']} см/с**")
        st.write(f"- Радиус пятна: **r0 = {v['r0_um']} мкм**")

        if mat == "резина":
            st.sidebar.caption("Для резины в таблице A не указано — используется допущение (по умолчанию 0.5).")
            NONMETALS["резина"]["A"] = st.sidebar.slider("A(резина)", 0.05, 0.95, float(NONMETALS["резина"]["A"]), 0.05)


        st.subheader("Найти")
        st.write("- Критерий \(V r_0/a\), тип источника (быстрый/медленный)")
        st.write("- Пороговая плотность мощности **q0** для резки при заданной скорости")

        V_m_s = v["V_cm_s"] / 100.0
        r0_m = v["r0_um"] * 1e-6

        q0, crit, kind, err = threshold_q0_for_cut(mat, V_m_s, r0_m)

        st.subheader("Решение")
        st.write("V * r0 / a  (≤ 1 — медленно движущийся, >1 — быстро движущийся)")
        st.write("где a = k / (ρ * c)")
        st.write("Далее порог q0 находим из формулы (5) (медленно движущийся) или из формулы (7) (быстро движущийся).")

        st.subheader("Ответ")
        st.write(f"- \(V r_0/a\) = **{crit:.3e}** → источник: **{kind}**")
        if err:
            st.error(err)
        else:
            st.success(f"- q0(порог) ≈ **{q0:.3e} Вт/м²** (≈ {q0/1e4:.3e} Вт/см²)")

    elif task == "Задача 4":
        st.header("Задача 4 — гравировка Cr испарением: V_max и скорость для снятия слоя h")
        v = TASK4_VARIANTS[variant]
        P_W = TASK4_FIXED["P_W"]
        r0_mm = TASK4_FIXED["r0_mm"] * 1e-6

        st.subheader("Дано")
        st.write("- Материал: хром (Cr)")
        st.write(f"- Мощность: **P = {P_W} Вт**")
        st.write(f"- Радиус пятна: **r0 = {r0_mm} мм**")
        st.write(f"- Толщина снимаемого слоя: **h = {v['h_um']} мкм**")

        cr = METALS["Cr"]

        a = thermal_diffusivity(cr["rho"], cr["k"], cr["c"])
        q0 = power_density(P_W, r0_mm)
        q = q0 * (1.0 - cr["R"])

        T_target = cr["Tb"]
        # ln_term = dT * 2 k h / (q r0^2)
        ln_term = T_target * 2.0 * cr["k"] * v["h_um"] * 1e-6 / (q * r0_mm**2)

        # V = 2.25 a / (r0 * exp(ln_term))
        V = 2.25 * a / (r0_mm * math.exp(ln_term))

        st.subheader("Решение")
        st.write("T ≈ (q * r0^2) / (2 * k * h) * ln(2.25 * a / (r0 * V)) + Tн")
        st.write("где:")
        st.write("q = P / (π * r0^2) * (1 - R)")
        st.write("a = k / (ρ * c)")
        st.write("Решаем относительно V.")

        st.subheader("Ответ")
        st.success(f"V ≈ **{V:.3e} м/с**  (≈ {V*100:.3e} см/с)")


