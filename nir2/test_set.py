
import streamlit as st

from app import api_key
from tools import all_actual_tools
from main import LLM_FC
max_num_fc = 3


def run_te(api_key = ''):
    main_llm = LLM_FC(api_key=api_key)
    question = 'Поставь глагол в скобках в правильную форму. We ___ football yesterday. (play)'
    result = question
    for _ in range(max_num_fc):
        result = main_llm.run(result)
        if main_llm.has_fc(result):
            func_name, args = main_llm.fc_prerun_desc(result)
            result = main_llm.fc(result)
            st.subheader("Function Call")
            st.json({"name": func_name, "arguments": args})
            st.subheader("Response")
            st.write(result)
        else:
            break

if __name__ == '__main__':
    api_key = input('Enter API')
    run_te(api_key)
