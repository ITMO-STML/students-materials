from typing import cast

from langchain_core.runnables import ConfigurableField
from langchain_core.runnables.configurable import RunnableConfigurableFields

from lib.gigachain_extension.gigachat_client_holder import GigaChatClientHolder
from lib.gigachain_extension.gigachat_ext import GigaChatExt
from lib.gigachain_extension.gigachat_properties import GigaChatProperties


class GigaChatExtFactory:
    def __init__(self, gigachat_properties: GigaChatProperties):
        self.gigachat_properties = gigachat_properties

    def create_gigachat(self) -> RunnableConfigurableFields:
        return cast(RunnableConfigurableFields,
                    GigaChatExt.create(gigachat_client_holder=GigaChatClientHolder(self.gigachat_properties))
                    .configurable_fields(
                        model=ConfigurableField(
                            id="model",
                            name="model",
                            description="model",
                        ),
                        profanity_check=ConfigurableField(
                            id="profanity_check",
                            name="profanity_check",
                            description="profanity_check",
                        ),
                        temperature=ConfigurableField(
                            id="temperature",
                            name="temperature",
                            description="temperature",
                        ),
                        max_tokens=ConfigurableField(
                            id="max_tokens",
                            name="max_tokens",
                            description="max_tokens",
                        ),
                        repetition_penalty=ConfigurableField(
                            id="repetition_penalty",
                            name="repetition_penalty",
                            description="repetition_penalty",
                        ),
                        top_p=ConfigurableField(
                            id="top_p",
                            name="top_p",
                            description="top_p",
                        ),
                    ))
