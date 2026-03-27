import os
from io import BytesIO

import streamlit as st
from docxtpl import DocxTemplate


st.set_page_config(
    page_title="Экономика ДЗ2",
    layout="wide"
)


# -----------------------------
# Вспомогательные функции
# -----------------------------
def normalize_variant(variant: int) -> int:

    if not 1 <= variant <= 30:
        raise ValueError("Вариант должен быть в диапазоне от 1 до 30")
    return (variant - 1) % 10


def safe_filename(text: str) -> str:
    forbidden = '<>:"/\\|?*'
    result = text.strip()
    for ch in forbidden:
        result = result.replace(ch, "_")
    result = "_".join(result.split())
    return result or "student"


def calculate_data(variant: int, name: str) -> dict:

    idx = normalize_variant(variant)
    base_variant = idx + 1

    # Исходные данные
    volume_prod_year = 200000

    # Численность
    pers_main_arr = [72, 74, 76, 78, 80, 82, 84, 86, 88, 90]
    pers_main = pers_main_arr[idx]
    pers_aux = 50
    leaders = 15
    specialists = 10
    employees = 5

    # Годовой фонд рабочего времени одного работника
    pers_main_time = 1712
    pers_aux_time = 1768
    leaders_time = 1701
    specialists_time = 1701
    employees_time = 1768

    # Производительность труда
    labor_product_main = volume_prod_year / pers_main
    labor_product_worker = volume_prod_year / (pers_main + pers_aux)
    labor_product_full = volume_prod_year / (
        pers_main + pers_aux + leaders + specialists + employees
    )

    # Трудоемкость
    labor_intensity_tehnog = (pers_main * pers_main_time) / volume_prod_year
    labor_intensity_production = (
        pers_main * pers_main_time + pers_aux * pers_aux_time
    ) / volume_prod_year
    labor_intensity_full = (
        pers_main * pers_main_time
        + pers_aux * pers_aux_time
        + leaders * leaders_time
        + specialists * specialists_time
        + employees * employees_time
    ) / volume_prod_year

    # Данные по оплате труда
    tarif_stavka_arr = [170, 180, 190, 200, 210, 220, 230, 240, 250, 260]
    tarif_stavka = tarif_stavka_arr[idx]  # руб./час (V разряд)

    chas_v_den = 7
    dney_v_mesyace = 20
    norma_vyrabotki_smena = 20
    fakt_vyrabotka = 460
    raschenka = 7.2
    premija_povremennaya = 10
    premija_procent = 0.5
    koeff_progres = 1.8

    # Расчёты заработной платы
    norma_vyrabotki_mesyac = norma_vyrabotki_smena * dney_v_mesyace

    zarabotok_a = tarif_stavka * chas_v_den * dney_v_mesyace
    zarabotok_b = zarabotok_a * (1 + premija_povremennaya / 100)
    zarabotok_v = raschenka * fakt_vyrabotka

    procent_perev = (
        (fakt_vyrabotka - norma_vyrabotki_mesyac) / norma_vyrabotki_mesyac
    ) * 100
    premija_g = zarabotok_v * (procent_perev * premija_procent / 100)
    zarabotok_g = zarabotok_v + premija_g

    raschenka_povyshennaya = raschenka * koeff_progres
    zarabotok_d = (
        raschenka * norma_vyrabotki_mesyac
        + (fakt_vyrabotka - norma_vyrabotki_mesyac) * raschenka_povyshennaya
    )

    # Полные данные
    data = {
        "variant_input": variant,
        "variant_base": base_variant,
        "name": name,

        "volume_prod_year": volume_prod_year,

        "pers_main": pers_main,
        "pers_aux": pers_aux,
        "leaders": leaders,
        "specialists": specialists,
        "employees": employees,

        "pers_main_time": pers_main_time,
        "pers_aux_time": pers_aux_time,
        "leaders_time": leaders_time,
        "specialists_time": specialists_time,
        "employees_time": employees_time,

        "labor_product_main": labor_product_main,
        "labor_product_worker": labor_product_worker,
        "labor_product_full": labor_product_full,

        "labor_intensity_tehnog": labor_intensity_tehnog,
        "labor_intensity_production": labor_intensity_production,
        "labor_intensity_full": labor_intensity_full,

        "tarif_stavka": tarif_stavka,
        "chas_v_den": chas_v_den,
        "dney_v_mesyace": dney_v_mesyace,
        "norma_vyrabotki_smena": norma_vyrabotki_smena,
        "fakt_vyrabotka": fakt_vyrabotka,
        "raschenka": raschenka,
        "premija_povremennaya": premija_povremennaya,
        "premija_procent": premija_procent,
        "koeff_progres": koeff_progres,
        "norma_vyrabotki_mesyac": norma_vyrabotki_mesyac,
        "procent_perev": procent_perev,
        "premija_g": premija_g,
        "zarabotok_a": zarabotok_a,
        "zarabotok_b": zarabotok_b,
        "zarabotok_v": zarabotok_v,
        "zarabotok_g": zarabotok_g,
        "raschenka_povyshennaya": raschenka_povyshennaya,
        "zarabotok_d": zarabotok_d,
    }

    return data


def build_docx_context(data: dict) -> dict:

    return {
        "var": data["variant_input"],
        "name": data["name"],
        "pers_main": data["pers_main"],

        "labor_product_main": f'{data["labor_product_main"]:.2f}',
        "labor_product_worker": f'{data["labor_product_worker"]:.2f}',
        "labor_product_full": f'{data["labor_product_full"]:.2f}',

        "labor_intensity_tehnog": f'{data["labor_intensity_tehnog"]:.4f}',
        "labor_intensity_production": f'{data["labor_intensity_production"]:.4f}',
        "labor_intensity_full": f'{data["labor_intensity_full"]:.4f}',

        "tarif_stavka": data["tarif_stavka"],

        "zarabotok_a": f'{data["zarabotok_a"]:.0f}',
        "zarabotok_b": f'{data["zarabotok_b"]:.0f}',
        "zarabotok_v": f'{data["zarabotok_v"]:.0f}',
        "premija_g": f'{data["premija_g"]:.1f}',
        "zarabotok_d": f'{data["zarabotok_d"]:.1f}',
    }


def generate_docx_file(template_path: str, context: dict) -> BytesIO:
    doc = DocxTemplate(template_path)
    doc.render(context)

    file_stream = BytesIO()
    doc.save(file_stream)
    file_stream.seek(0)
    return file_stream

def run():
    st.title("Экономика — ДЗ 2")

    image1 = os.path.join("media", "hun_mem.jpg")
    image2 = os.path.join("media", "cofe_mem.jpg")
    st.image(image1)
    st.image(image2)


    st.write(f"Просто поставьте свой вариант и напишите фамилию в виде  Иванов И.И. Далее скачайте отчет")

    with st.sidebar:
        st.header("Ввод данных")
        fio = st.text_input("ФИО", placeholder="Иванов И.И.")
        variant = st.number_input(
            "Вариант",
            min_value=1,
            max_value=30,
            value=1,
            step=1
        )
        template_path = "pattern_economika_dz2.docx"

    try:
        data = calculate_data(variant=variant, name=fio.strip())
    except Exception as e:
        st.error(f"Ошибка расчёта: {e}")
        st.stop()

    st.info(
        f"Выбран вариант: **{data['variant_input']}**. "
        f"Базовый вариант для массивов: **{data['variant_base']}**."
    )

    # -----------------------------
    # Исходные данные
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Исходные данные по производительности")
        st.write(f"**Объём продукции за год:** {data['volume_prod_year']:,} т")
        st.write(f"**Основные рабочие:** {data['pers_main']}")
        st.write(f"**Вспомогательные рабочие:** {data['pers_aux']}")
        st.write(f"**Руководители:** {data['leaders']}")
        st.write(f"**Специалисты:** {data['specialists']}")
        st.write(f"**Служащие:** {data['employees']}")

    with col2:
        st.subheader("Исходные данные по оплате труда")
        st.write(f"**Тарифная ставка:** {data['tarif_stavka']} руб./час")
        st.write(f"**Часов в день:** {data['chas_v_den']}")
        st.write(f"**Рабочих дней в месяце:** {data['dney_v_mesyace']}")
        st.write(f"**Норма выработки за смену:** {data['norma_vyrabotki_smena']} деталей")
        st.write(f"**Фактическая выработка:** {data['fakt_vyrabotka']} деталей")
        st.write(f"**Расценка:** {data['raschenka']} руб./дет.")

    st.divider()

    # -----------------------------
    # Результаты производительности
    # -----------------------------
    st.subheader("1. Показатели производительности труда")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Выработка на 1 основного рабочего",
            f'{data["labor_product_main"]:.2f} т'
        )
        st.metric(
            "Трудоёмкость технологическая",
            f'{data["labor_intensity_tehnog"]:.4f} чел.-ч/т'
        )

    with c2:
        st.metric(
            "Выработка на 1 рабочего",
            f'{data["labor_product_worker"]:.2f} т'
        )
        st.metric(
            "Трудоёмкость производственная",
            f'{data["labor_intensity_production"]:.4f} чел.-ч/т'
        )

    with c3:
        st.metric(
            "Выработка на 1 работающего",
            f'{data["labor_product_full"]:.2f} т'
        )
        st.metric(
            "Трудоёмкость полная",
            f'{data["labor_intensity_full"]:.4f} чел.-ч/т'
        )

    st.divider()

    st.subheader("2. Расчёт заработной платы")

    c4, c5 = st.columns(2)

    with c4:
        st.markdown("### Повременные системы")
        st.write(f"**а) Простая повременная:** {data['zarabotok_a']:.0f} руб.")
        st.write(f"**б) Повременно-премиальная:** {data['zarabotok_b']:.0f} руб.")

    with c5:
        st.markdown("### Сдельные системы")
        st.write(f"**в) Прямая сдельная:** {data['zarabotok_v']:.0f} руб.")
        st.write(f"**г) Сдельно-премиальная:** {data['zarabotok_g']:.1f} руб.")
        st.write(f"Премия за перевыполнение: {data['premija_g']:.1f} руб.")
        st.write(f"**д) Сдельно-прогрессивная:** {data['zarabotok_d']:.1f} руб.")
        st.write(f"Повышенная расценка: {data['raschenka_povyshennaya']:.2f} руб./дет.")

    st.divider()

    # -----------------------------
    # Таблица итогов
    # -----------------------------
    st.subheader("Итоговые значения")

    table_data = {
        "Показатель": [
            "Численность основных рабочих",
            "Выработка на 1 основного рабочего",
            "Выработка на 1 рабочего",
            "Выработка на 1 работающего",
            "Трудоёмкость технологическая",
            "Трудоёмкость производственная",
            "Трудоёмкость полная",
            "Тарифная ставка",
            "Простая повременная оплата",
            "Повременно-премиальная оплата",
            "Прямая сдельная оплата",
            "Премия при сдельно-премиальной",
            "Сдельно-прогрессивная оплата",
        ],
        "Значение": [
            f"{data['pers_main']}",
            f'{data["labor_product_main"]:.2f}',
            f'{data["labor_product_worker"]:.2f}',
            f'{data["labor_product_full"]:.2f}',
            f'{data["labor_intensity_tehnog"]:.4f}',
            f'{data["labor_intensity_production"]:.4f}',
            f'{data["labor_intensity_full"]:.4f}',
            f"{data['tarif_stavka']}",
            f'{data["zarabotok_a"]:.0f}',
            f'{data["zarabotok_b"]:.0f}',
            f'{data["zarabotok_v"]:.0f}',
            f'{data["premija_g"]:.1f}',
            f'{data["zarabotok_d"]:.1f}',
        ]
    }
    st.table(table_data)

    st.divider()


    st.subheader("3. Выгрузка в Word")

    if not fio.strip():
        st.warning("Чтобы сформировать Word-файл, введите ФИО.")
    else:
        if not os.path.exists(template_path):
            st.error(
                f"Шаблон `{template_path}` не найден. "
                f"Положите файл шаблона рядом с `app.py`."
            )
        else:
            context = build_docx_context(data)

            try:
                docx_file = generate_docx_file(template_path, context)
                output_filename = (
                    f"Экономика_ДЗ2_{safe_filename(fio)}.docx"
                )

                st.download_button(
                    label="Скачать Word-файл",
                    data=docx_file,
                    file_name=output_filename,
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )


            except Exception as e:
                st.error(f"Ошибка при генерации Word-файла: {e}")
