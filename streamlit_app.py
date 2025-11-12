import streamlit as st
from laba1 import run as run_laba1
from laba2 import run as run_laba2
from laba3 import run as run_laba3

def main():
    st.title("Выбор лабораторной работы")

    option = st.selectbox(
        "Выберите лабораторную работу",
        ("Лабораторная 1", "Лабораторная 2", "Лабораторная 3")
    )

    if option == "Лабораторная 1":
        run_laba1()
    elif option == "Лабораторная 2":
        run_laba2()
    elif option == "Лабораторная 3":
        run_laba3()

if __name__ == "__main__":
    main()

