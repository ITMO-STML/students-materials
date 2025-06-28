# Запустить команду:
#   streamlit run app.py

import streamlit as st
from tools import all_actual_tools
from main import LLM_FC

api_key = st.text_input("🔐 Введите API-ключ", type="password")
# Настройка страницы
st.set_page_config(page_title="GigaChat Tool Interface")

# Заголовок и описание
st.title("🛠️ GigaChat Tool Interface")
st.write("Введите вопрос, система подберёт нужный инструмент и покажет вызов функции и результат.")

# Ввод вопроса
question = st.text_area("Ваш вопрос:", height=100)

max_num_fc = 5

if st.button("Отправить"):
    if not question or not api_key:
        st.error("❗ Введите вопрос и API-ключ.")
    else:
        bot = LLM_FC(api_key=api_key)
        with st.spinner("🧠 Думаю…"):
            answer, steps = bot.run(question)

        st.subheader("Ответ")
        st.write(answer)

        if steps:
            st.subheader("Ход выполнения")
            for s in steps:
                st.code(f"function_call → {s['name']}({s['arguments']})")
                st.write("Result:", s["result"])
        else:
            st.info("Функция не потребовалась — модель ответила сама.")