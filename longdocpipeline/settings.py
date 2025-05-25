import os

CONFIG_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conf')

APPLICATION_CONFIG_PATH = os.getenv("APPLICATION_CONFIG_PATH", os.path.join(CONFIG_BASE_PATH, "application.yaml"))

APPLICATION_CONFIG_PATHS = [APPLICATION_CONFIG_PATH]
APPLICATION_ETC_CONFIG_PATH = os.getenv("APPLICATION_ETC_CONFIG_PATH", "")
if os.path.isfile(APPLICATION_ETC_CONFIG_PATH):
    with open(APPLICATION_ETC_CONFIG_PATH) as etc:
        # If this file is empty then error raise: TypeError: 'NoneType' object is not iterable by pydantic_settings/sources.py
        if etc.read().strip():
            APPLICATION_CONFIG_PATHS.append(APPLICATION_ETC_CONFIG_PATH)


DEFAULT_PROMPT_PATH = os.path.join(CONFIG_BASE_PATH, "default_prompts.json")
