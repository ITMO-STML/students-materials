from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function
import json
from tools import tool_schemas, all_actual_tools

SYSTEM_PROMPT = """
Ты — ассистент по английскому языку. Твоя задача — отвечать на вопросы о грамматике. 
Если уверен на 100 % — отвечай сразу. Если сомневаешься в случаях, которые определены функциями, 
(они описаны в definitions), вызывай соответствующую функцию.

Пример:
user: What is the past tense of "arise"?
assistant → (function_call: check_regularity, { "infinitive": "arise" })
function returns → "arise → arose / arisen"
assistant: Past Simple = "arose", Past Participle = "arisen".
"""


class LLM_FC:
    def __init__(self, api_key='', max_tokens=4096):
        self.context = []
        self.model = GigaChat(
            base_url='https://gigachat.devices.sberbank.ru/api/v1',
            auth_url='https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
            credentials=api_key,
            scope='GIGACHAT_API_CORP',
            model='GigaChat-Max',
            timeout=60.0,
            verbose=True,
            verify_ssl_certs=False,
            temperature=1e-8,
            profanity=False,
            max_tokens=max_tokens,
        )
        self.functions = []
        self.get_all_functions()

    def get_all_functions(self):
        self.functions = tool_schemas()

    def has_fc(self, message):
        return 'function_call' in message[-1]["choices"][-1]['finish_reason']

    def fc_prerun_desc(self, message):
        func_to_be_called = message[-1]["choices"][-1]['message']['function_call']['name']
        its_args = message[-1]["choices"][-1]['message']['function_call']['arguments']
        return func_to_be_called, its_args

    def fc(self, message):
        # 1. Получить имя и аргументы функции
        func_name, func_args = self.fc_prerun_desc(message)
        # 2. Найти соответствующий callable в all_actual_tools
        tools_map = all_actual_tools()
        if func_name not in tools_map:
            # Если функция неизвестна — кидаем ошибку или возвращаем сообщение
            message.append({
                'role': 'system',
                'content': f"Ошибка: неизвестная функция '{func_name}'"
            })
            return message
        # 3. Вызываем функцию
        try:
            result = tools_map[func_name](**func_args)
        except Exception as e:
            # Обработка ошибок в вызове функции
            message.append({
                'role': 'system',
                'content': f"Ошибка при выполнении функции '{func_name}': {str(e)}"
            })
            return message
        # 4. Добавляем сообщение от роли 'function' с результатом
        message.append({
            'role': 'function',
            'name': func_name,
            'content': str(result)
        })
        return message

    def run(self, request: str):
        steps = []  # ход
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
            Messages(role=MessagesRole.USER, content=request),
        ]
        response = self.model.chat(Chat(messages=messages,
                                        functions=self.functions,
                                        function_call="auto"))

        if response.choices[0].finish_reason == "function_call":
            call = response.choices[0].message.function_call
            fn_name = call.name
            fn_args = call.arguments
            py_result = all_actual_tools()[fn_name](**fn_args)

            # сохраняем шаг
            steps.append({
                "name": fn_name,
                "arguments": fn_args,
                "result": py_result,
            })

            # сообщения назад в модель
            messages.append(
                Messages(role=MessagesRole.ASSISTANT,
                         content="",
                         function_call={"name": fn_name, "arguments": fn_args})
            )
            messages.append(
                Messages(role=MessagesRole.FUNCTION,
                         name=fn_name,
                         content=json.dumps(py_result))
            )
            response = self.model.chat(Chat(messages=messages,
                                            functions=self.functions,
                                            function_call="auto"))

        self.context = messages
        final_answer = response.choices[0].message.content
        return final_answer, steps


if __name__ == '__main__':
    print('Hello!')