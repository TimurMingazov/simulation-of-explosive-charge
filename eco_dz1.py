import streamlit as st
import pandas as pd
from docxtpl import DocxTemplate
from io import BytesIO
import datetime
import os
from pathlib import Path
import traceback

# Настройка страницы
st.set_page_config(
    page_title="Экономика предприятия",
    layout="wide"
)

# Заголовок приложения
st.title("Экономика предприятия")
st.markdown("---")

# Создание вкладок для двух задач
tab1, tab2, tab3 = st.tabs(["Задача 1.1", "Задача 1.2", "📄 Генерация отчета"])

# Инициализация session_state для хранения результатов
if 'report_data' not in st.session_state:
    st.session_state.report_data = {}

# ---------- ПЕРВАЯ ЗАДАЧА ----------
with tab1:
    st.header("Расчет среднегодовой стоимости основных производственных фондов")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Данные в таблице являются изначальными, они одинаковы для все вариантов.\n" \
        "Изменять их или добавлять новые можно, чтобы посмотреть, что измениться и т.д.\n" \
        "Чтобы сдать задачу НЕОБХОДИМО ВВЕСТИ ТОЛЬКО СВОЙ ВАРИАНТ, остальное можно не трогать.")
        st.subheader("Ввод данных")

        # Словарь месяцев
        months = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                  'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

        # Словарь для номеров месяцев
        num_month = {month: i for i, month in enumerate(months)}

        # Ввод варианта
        var = st.number_input("Введите номер варианта:",
                             min_value=1, max_value=30, value=1, step=1, key="var1")

        if var == 10 or var == 20 or var == 30:
            cost_n_w = 16000
        else:
            cost_n_w = 15000 + 100 * (var %  10)
        st.info(f"💰 **Стоимость ОПФ на начало года:** {cost_n_w:,.2f} руб. (Вариант {var})")

        # Сохраняем в session_state для отчета
        st.session_state.report_data['var'] = var
        st.session_state.report_data['cost_n_w_1'] = cost_n_w

        st.markdown("---")

        # Инициализация данных в session_state
        if 'data_cost_in' not in st.session_state:
            # Начальные данные по вводу средств
            st.session_state.data_cost_in = {'Март': 200, 'Июнь': 150, 'Август': 250}

        if 'data_cost_out' not in st.session_state:
            # Начальные данные по выбытию средств
            st.session_state.data_cost_out = {'Февраль': 100, 'Октябрь': 300}

        # Отображение текущих данных
        st.write("**Текущие данные о вводе средств:**")
        if st.session_state.data_cost_in:
            df_current_in = pd.DataFrame([
                {"Месяц": month, "Сумма (руб.)": cost}
                for month, cost in st.session_state.data_cost_in.items()
            ])
            st.dataframe(df_current_in, use_container_width=True, hide_index=True)
        else:
            st.info("Нет данных о вводе средств")

        st.markdown("---")

        # Добавление новых данных по вводу
        st.write("**Добавить новые данные о вводе средств:**")
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            month_in = st.selectbox("Месяц ввода:", months, key="month_in")
        with col_in2:
            cost_in = st.number_input("Сумма ввода (руб.):", min_value=0, value=0, key="cost_in")

        if st.button("Добавить ввод средств", key="add_in"):
            if cost_in > 0:
                if month_in in st.session_state.data_cost_in:
                    st.session_state.data_cost_in[month_in] += cost_in
                else:
                    st.session_state.data_cost_in[month_in] = cost_in
                st.success(f"Добавлено: {month_in} - {cost_in} руб.")
                st.rerun()

        st.markdown("---")

        # Отображение текущих данных о выбытии
        st.write("**Текущие данные о выбытии средств:**")
        if st.session_state.data_cost_out:
            df_current_out = pd.DataFrame([
                {"Месяц": month, "Сумма (руб.)": cost}
                for month, cost in st.session_state.data_cost_out.items()
            ])
            st.dataframe(df_current_out, use_container_width=True, hide_index=True)
        else:
            st.info("Нет данных о выбытии средств")

        st.markdown("---")

        # Добавление новых данных по выбытию
        st.write("**Добавить новые данные о выбытии средств:**")
        col_out1, col_out2 = st.columns(2)
        with col_out1:
            month_out = st.selectbox("Месяц выбытия:", months, key="month_out")
        with col_out2:
            cost_out = st.number_input("Сумма выбытия (руб.):", min_value=0, value=0, key="cost_out")

        if st.button("➕ Добавить выбытие средств", key="add_out"):
            if cost_out > 0:
                if month_out in st.session_state.data_cost_out:
                    st.session_state.data_cost_out[month_out] += cost_out
                else:
                    st.session_state.data_cost_out[month_out] = cost_out
                st.success(f"Добавлено: {month_out} - {cost_out} руб.")
                st.rerun()

        # Кнопки управления
        col_reset1, col_reset2 = st.columns(2)
        with col_reset1:
            if st.button("🔄 Сбросить к начальным данным", key="reset_to_initial"):
                st.session_state.data_cost_in = {'Март': 200, 'Июнь': 150, 'Август': 250}
                st.session_state.data_cost_out = {'Февраль': 100, 'Октябрь': 300}
                st.success("Данные сброшены к начальным значениям")
                st.rerun()

        with col_reset2:
            if st.button("🗑️ Очистить все данные", key="clear_all"):
                st.session_state.data_cost_in = {}
                st.session_state.data_cost_out = {}
                st.success("Все данные очищены")
                st.rerun()

    with col2:
        st.subheader("Результаты расчета")

        # Получаем данные из session_state
        data_cost_in = st.session_state.get('data_cost_in', {})
        data_cost_out = st.session_state.get('data_cost_out', {})

        if data_cost_in or data_cost_out:
            # Функции для расчетов
            def calculate_cost_in(data_cost_in):
                """Расчет стоимости введенных средств"""
                cost_in = 0
                details = []
                for month, cost in data_cost_in.items():
                    n_month = (12 - num_month[month])
                    month_cost = cost * (n_month / 12)
                    cost_in += month_cost
                    details.append({
                        'month': month,
                        'cost': cost,
                        'n_month': n_month,
                        'month_cost': month_cost
                    })
                return cost_in, details

            def calculate_cost_out(data_cost_out):
                """Расчет стоимости выбывших средств"""
                cost_out = 0
                details = []
                for month, cost in data_cost_out.items():
                    n_month = (12 - num_month[month])
                    month_cost = cost * (n_month / 12)
                    cost_out += month_cost
                    details.append({
                        'month': month,
                        'cost': cost,
                        'n_month': n_month,
                        'month_cost': month_cost
                    })
                return cost_out, details

            # Детальный расчет
            with st.expander("Детальный расчет среднегодовой стоимости"):
                st.write(f"**Стоимость на начало года:** {cost_n_w:.2f} тыс.руб.")

                # Расчет введенных средств
                st.write("**Расчет стоимости введенных средств:**")
                cost_in_total, in_details = calculate_cost_in(data_cost_in)
                for detail in in_details:
                    st.write(f"  {detail['month']}: {detail['cost']} тыс.руб. × ({detail['n_month']}/12) = {detail['month_cost']:.2f} тыс.руб.")
                st.write(f"**Итого введено:** {cost_in_total:.2f} тыс.руб.")
                st.write("---")

                # Расчет выбывших средств
                st.write("**Расчет стоимости выбывших средств:**")
                cost_out_total, out_details = calculate_cost_out(data_cost_out)
                for detail in out_details:
                    st.write(f"  {detail['month']}: {detail['cost']} тыс.руб. × ({detail['n_month']}/12) = {detail['month_cost']:.2f} тыс.руб.")
                st.write(f"**Итого выбыло:** {cost_out_total:.2f} тыс.руб.")
                st.write("---")

                # Итоговый расчет
                average_cost = cost_n_w + cost_in_total - cost_out_total
                st.write(f"**Среднегодовая стоимость =** {cost_n_w:.2f} + {cost_in_total:.2f} - {cost_out_total:.2f} = {average_cost:.2f} тыс.руб.")

            # Основные расчеты
            cost_in_total, _ = calculate_cost_in(data_cost_in)
            cost_out_total, _ = calculate_cost_out(data_cost_out)
            average_cost = cost_n_w + cost_in_total - cost_out_total

            # Расчет коэффициентов
            total_in_sum = sum(data_cost_in.values()) if data_cost_in else 0
            total_out_sum = sum(data_cost_out.values()) if data_cost_out else 0

            coeff_in = total_in_sum / (cost_n_w + total_in_sum - total_out_sum) if (cost_n_w + total_in_sum - total_out_sum) != 0 else 0
            coeff_out = total_out_sum / cost_n_w if cost_n_w != 0 else 0

            # Сохраняем результаты в session_state для отчета
            st.session_state.report_data['coeff_in_1'] = coeff_in
            st.session_state.report_data['coeff_out_1'] = coeff_out
            st.session_state.report_data['average_cost_1'] = average_cost

            # Отображение результатов
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Среднегодовая стоимость", f"{average_cost:,.2f} тыс.руб.")
            with col_metric2:
                st.metric("Коэффициент ввода", f"{coeff_in:.4f}")
            with col_metric3:
                st.metric("Коэффициент выбытия", f"{coeff_out:.4f}")

            # Сводная таблица
            st.subheader("Сводная таблица данных")

            # Подготовка данных для сводной таблицы
            all_data = []
            for month, cost in data_cost_in.items():
                all_data.append({
                    "Операция": "Ввод",
                    "Месяц": month,
                    "Сумма": f"{cost} тыс.руб.",
                    "Среднегодовая стоимость": f"{cost * (12 - num_month[month]) / 12:.2f} тыс.руб."
                })

            for month, cost in data_cost_out.items():
                all_data.append({
                    "Операция": "Выбытие",
                    "Месяц": month,
                    "Сумма": f"{cost} тыс.руб.",
                    "Среднегодовая стоимость": f"{cost * (12 - num_month[month]) / 12:.2f} тыс.руб."
                })

            if all_data:
                df_all = pd.DataFrame(all_data)
                st.dataframe(df_all, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Добавьте данные о вводе и выбытии средств для расчета")

# ---------- ВТОРАЯ ЗАДАЧА ----------
with tab2:
    st.header("Расчет амортизационных отчислений")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Исходные данные")

        # Ввод варианта
        var = st.number_input("Введите номер варианта:",
                             min_value=1, max_value=30, value=10, step=1, key="var2")

        # Исправленная формула: 160 + 10 * номер_варианта
        if var == 10 or var == 20 or var == 30:
            cost_n_w = 260
        else:
            cost_n_w = 160 + 10 * (var % 10)

        # Параметры расчета
        st.info(f"💰 **Первоначальная стоимость ОПФ:** {cost_n_w} тыс. руб. (Вариант {var})")

        # Сохраняем в session_state для отчета
        st.session_state.report_data['cost_n_w_2'] = cost_n_w

        fact_exploitation = st.number_input("Фактический срок эксплуатации (лет):",
                                           min_value=1, max_value=20, value=3, key="fact_exp")
        full_exploitation = st.number_input("Нормативный срок эксплуатации (лет):",
                                           min_value=1, max_value=30, value=10, key="full_exp")
        k_boost = st.number_input("Коэффициент ускорения:",
                                 min_value=1.0, max_value=3.0, value=2.0, step=0.1, key="k_boost")

    with col2:
        st.subheader("Результаты расчетов")

        # Функции расчета
        def linear_method(cost, fact_exp, full_exp):
            norma_amort = 1 / full_exp
            amort_cost = norma_amort * cost
            cost_ost = cost - fact_exp * amort_cost
            coeff_iznosa = (fact_exp * amort_cost) / cost
            return cost_ost, coeff_iznosa, amort_cost

        def method_reducing_balance(cost, fact_exp, full_exp, k_boost):
            norma_amort = k_boost / full_exp
            amort_cost_arr = []
            remaining = cost
            for i in range(fact_exp):
                amort = norma_amort * remaining
                amort_cost_arr.append(amort)
                remaining -= amort
            cost_ost = remaining
            coeff_iznosa = sum(amort_cost_arr) / cost
            return cost_ost, coeff_iznosa, amort_cost_arr, remaining

        def method_sum_number_year(cost, fact_exp, full_exp):
            sum_num_year = (1 + full_exp) * full_exp / 2
            amort_cost_arr = []
            for i in range(fact_exp):
                amort = cost * (full_exp - i) / sum_num_year
                amort_cost_arr.append(amort)
            cost_ost = cost - sum(amort_cost_arr)
            coeff_iznosa = sum(amort_cost_arr) / cost
            return cost_ost, coeff_iznosa, amort_cost_arr, cost_ost

        # Расчеты
        lin_ost, lin_iznos, lin_amort = linear_method(cost_n_w, fact_exploitation, full_exploitation)
        bal_ost, bal_iznos, bal_amort_arr, bal_remaining = method_reducing_balance(cost_n_w, fact_exploitation, full_exploitation, k_boost)
        year_ost, year_iznos, year_amort_arr, year_remaining = method_sum_number_year(cost_n_w, fact_exploitation, full_exploitation)

        # Сохраняем результаты для отчета
        st.session_state.report_data['lin_amort'] = lin_amort
        st.session_state.report_data['lin_ost'] = lin_ost

        # Амортизация по методу уменьшаемого остатка
        if len(bal_amort_arr) >= 1:
            st.session_state.report_data['ao1_ost'] = bal_amort_arr[0]
        if len(bal_amort_arr) >= 2:
            st.session_state.report_data['ao2_ost'] = bal_amort_arr[1]
        if len(bal_amort_arr) >= 3:
            st.session_state.report_data['ao3_ost'] = bal_amort_arr[2]
        st.session_state.report_data['remaining'] = bal_remaining

        # Амортизация по методу суммы чисел лет
        if len(year_amort_arr) >= 1:
            st.session_state.report_data['year_amort1'] = year_amort_arr[0]
        if len(year_amort_arr) >= 2:
            st.session_state.report_data['year_amort2'] = year_amort_arr[1]
        if len(year_amort_arr) >= 3:
            st.session_state.report_data['year_amort3'] = year_amort_arr[2]
        st.session_state.report_data['remaining_year'] = year_remaining

        # Создаем таблицу результатов
        results_data = {
            "Метод": ["Линейный", "Уменьшаемого остатка", "По сумме чисел лет"],
            "Остаточная стоимость (тыс. руб.)": [
                f"{lin_ost:.2f}",
                f"{bal_ost:.2f}",
                f"{year_ost:.2f}"
            ],
            "Коэффициент износа": [
                f"{lin_iznos:.4f}",
                f"{bal_iznos:.4f}",
                f"{year_iznos:.4f}"
            ],
            "Амортизация за первый год (тыс. руб.)": [
                f"{lin_amort:.2f}",
                f"{bal_amort_arr[0]:.2f}" if bal_amort_arr else "0.00",
                f"{year_amort_arr[0]:.2f}" if year_amort_arr else "0.00"
            ]
        }

        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        # Детализация по годам
        with st.expander("Детализация по годам"):
            tab_lin, tab_bal, tab_year = st.tabs(["Линейный метод", "Метод уменьшаемого остатка", "Метод суммы чисел лет"])

            with tab_lin:
                st.write("**Линейный метод:**")
                st.write(f"Норма амортизации: {1/full_exploitation:.4f}")
                st.write(f"Годовая амортизация: {lin_amort:.2f} тыс. руб.")

                lin_data = []
                remaining = cost_n_w
                for year in range(1, fact_exploitation + 1):
                    amort = lin_amort
                    remaining -= amort
                    lin_data.append({
                        "Год": year,
                        "Амортизация": f"{amort:.2f}",
                        "Остаточная стоимость": f"{remaining:.2f}",
                        "Коэффициент износа": f"{(year * amort)/cost_n_w:.4f}"
                    })
                df_lin = pd.DataFrame(lin_data)
                st.dataframe(df_lin, use_container_width=True, hide_index=True)

            with tab_bal:
                st.write("**Метод уменьшаемого остатка:**")
                st.write(f"Норма амортизации с учетом ускорения: {k_boost/full_exploitation:.4f}")

                bal_data = []
                remaining = cost_n_w
                for year in range(1, fact_exploitation + 1):
                    amort = bal_amort_arr[year-1] if year <= len(bal_amort_arr) else 0
                    remaining -= amort
                    bal_data.append({
                        "Год": year,
                        "Амортизация": f"{amort:.2f}",
                        "Остаточная стоимость": f"{remaining:.2f}",
                        "Коэффициент износа": f"{sum(bal_amort_arr[:year])/cost_n_w:.4f}"
                    })
                df_bal = pd.DataFrame(bal_data)
                st.dataframe(df_bal, use_container_width=True, hide_index=True)

            with tab_year:
                st.write("**Метод суммы чисел лет:**")
                st.write(f"Сумма чисел лет: {((1+full_exploitation)*full_exploitation/2):.0f}")

                year_data = []
                remaining = cost_n_w
                for year in range(1, fact_exploitation + 1):
                    amort = year_amort_arr[year-1] if year <= len(year_amort_arr) else 0
                    remaining -= amort
                    year_data.append({
                        "Год": year,
                        f"Доля ({full_exploitation-year+1}/{((1+full_exploitation)*full_exploitation/2):.0f})": f"{amort:.2f}",
                        "Остаточная стоимость": f"{remaining:.2f}",
                        "Коэффициент износа": f"{sum(year_amort_arr[:year])/cost_n_w:.4f}"
                    })
                df_year = pd.DataFrame(year_data)
                st.dataframe(df_year, use_container_width=True, hide_index=True)

with tab3:
    st.header("📄 Генерация отчета в Word")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Данные для отчета")

        name = st.text_input("Введите ФИО:", value="", key="student_name_report")
        if name:
            st.session_state.report_data['name'] = name

        st.write("Текущие данные для подстановки:")

        preview_data = []

        # Задача 1
        if 'var' in st.session_state.report_data:
            preview_data.append({"Параметр": "Номер варианта", "Значение": st.session_state.report_data['var']})
        if 'cost_n_w_1' in st.session_state.report_data:
            preview_data.append({"Параметр": "Стоимость ОПФ на начало года (зад.1)", "Значение": f"{st.session_state.report_data['cost_n_w_1']:,.2f} руб."})
        if 'average_cost_1' in st.session_state.report_data:
            preview_data.append({"Параметр": "Среднегодовая стоимость", "Значение": f"{st.session_state.report_data['average_cost_1']:,.2f} тыс.руб."})
        if 'coeff_in_1' in st.session_state.report_data:
            preview_data.append({"Параметр": "Коэффициент ввода", "Значение": f"{st.session_state.report_data['coeff_in_1']:.4f}"})
        if 'coeff_out_1' in st.session_state.report_data:
            preview_data.append({"Параметр": "Коэффициент выбытия", "Значение": f"{st.session_state.report_data['coeff_out_1']:.4f}"})

        # Задача 2 (можно сократить, если всё аналогично)

        if preview_data:
            import pandas as pd
            st.table(pd.DataFrame(preview_data))
        else:
            st.warning("Сначала выполните расчёты в задачах 1 и 2")

    with col2:
        st.subheader("Генерация отчета")

        st.markdown("""
        Для скачивания отчёта нужно:
        1. Ввести ФИО  
        2. Выполнить обе задачи  
        3. Нажать кнопку ниже
        """)

        if st.button("📄 Сформировать и скачать отчёт", type="primary", use_container_width=True):

            if not name:
                st.error("Введите ФИО студента")
                st.stop()

            required_keys = ['var', 'cost_n_w_1', 'average_cost_1', 'cost_n_w_2']
            if not all(k in st.session_state.report_data for k in required_keys):
                st.error("Не все обязательные расчёты выполнены")
                st.stop()

            try:
                # Пытаемся найти шаблон несколькими способами
                possible_paths = [
                    Path("pattern_economica_dz1.docx"),
                        Path.cwd() / "pattern_economica_dz1.docx",
                ]

                template_path = None
                for p in possible_paths:
                    if p.exists():
                        template_path = p
                        break

                if not template_path:
                    st.error("Файл шаблона pattern_economica_dz1.docx не найден ни в одной из проверяемых директорий")
                    st.write("Проверялись пути:")
                    for p in possible_paths:
                        st.write(f"- {p}")
                    st.write(f"Текущая рабочая директория: {os.getcwd()}")
                    st.stop()

                # ─── Загрузка шаблона ───────────────────────────────
                doc = DocxTemplate(str(template_path))

                # Контекст
                context = {
                    'var':               st.session_state.report_data.get('var', ''),
                    'name':              name.strip(),
                    'cost_n_w_1':        f"{st.session_state.report_data.get('cost_n_w_1', 0):.2f}",
                    'coeff_in_1':        f"{st.session_state.report_data.get('coeff_in_1', 0):.4f}",
                    'coeff_out_1':       f"{st.session_state.report_data.get('coeff_out_1', 0):.4f}",
                    'average_cost_1':    f"{st.session_state.report_data.get('average_cost_1', 0):.2f}",
                    'cost_n_w_2':        f"{st.session_state.report_data.get('cost_n_w_2', 0):.2f}",
                    'lin_amort':         f"{st.session_state.report_data.get('lin_amort', 0):.2f}",
                    'lin_ost':         f"{st.session_state.report_data.get('lin_ost', 0):.2f}",
                    'ao1_ost':           f"{st.session_state.report_data.get('ao1_ost', 0):.2f}",
                    'ao2_ost':           f"{st.session_state.report_data.get('ao2_ost', 0):.2f}",
                    'ao3_ost':           f"{st.session_state.report_data.get('ao3_ost', 0):.2f}",
                    'remaining':         f"{st.session_state.report_data.get('remaining', 0):.2f}",
                    'year_amort1':       f"{st.session_state.report_data.get('year_amort1', 0):.2f}",
                    'year_amort2':       f"{st.session_state.report_data.get('year_amort2', 0):.2f}",
                    'year_amort3':       f"{st.session_state.report_data.get('year_amort3', 0):.2f}",
                    'remaining_year':    f"{st.session_state.report_data.get('remaining_year', 0):.2f}",
                    'date':              datetime.datetime.now().strftime("%d.%m.%Y"),
                    'year':              str(datetime.datetime.now().year),
                }

                # Рендерим
                doc.render(context)

                # Сохраняем в память
                output = BytesIO()
                doc.save(output)
                output.seek(0)

                # Чистое имя файла
                clean_name = "".join(c for c in name if c.isalnum() or c in ' -_').strip().replace(" ", "_")

                filename = f"И912С_{clean_name}_ЭкономикаПред_ДЗ1.docx"

                st.download_button(
                    label="💾 Скачать отчёт",
                    data=output,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True
                )

                st.success("Отчёт успешно сформирован!")
                st.balloons()

            except Exception as e:
                st.error("Произошла ошибка при создании отчёта")
                with st.expander("Подробности ошибки (для отладки)", expanded=True):
                    st.code(traceback.format_exc(), language="python")
                st.info("Чаще всего проблема в:\n• битый/неправильный шаблон .docx\n• конфликт кодировок\n• старая версия python-docx")