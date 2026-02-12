import streamlit as st
from laba1 import run as run_laba1
from laba2 import run as run_laba2
from laba3 import run as run_laba3
from dz_2_evd import run as run_dz2_evd
from dz_3_evd import run as run_dz3_evd
from eco_dz1 import run as run_eco_dz1

def main():
    st.title("Выбор работы")

    option = st.selectbox(
        "Выберите лабораторную работу",
        ("Лабораторная 1 ДСП", "Лабораторная 2 ДСП", "Лабораторная 3 ДСП", "Домашняя работа 2 Технология приборостроения", "Домашняя работа 3 Технология приборостроения", "ДЗ 1 Экономика предприятия")
    )

    if option == "Лабораторная 1 ДСП":
        run_laba1()
    elif option == "Лабораторная 2 ДСП":
        run_laba2()
    elif option == "Лабораторная 3 ДСП":
        run_laba3()
    elif option == "Домашняя работа 2 Технология приборостроения":
        run_dz2_evd()
    elif option == "Домашняя работа 3 Технология приборостроения":
        run_dz3_evd()
    elif option == "ДЗ 1 Экономика предприятия":
        run_eco_dz1()

if __name__ == "__main__":
    main()




