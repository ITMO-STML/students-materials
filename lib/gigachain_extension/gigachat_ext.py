import json
from typing import List, Any

import gigachat.models as gm
from gigachat import GigaChat as GigaChatClient
from langchain_community.chat_models import GigaChat as GigaChatModel
from langchain_community.chat_models.gigachat import logger
from langchain_core.messages import BaseMessage

from lib.gigachain_extension.gigachat_client_holder import GigaChatClientHolder


class GigaChatExt(GigaChatModel):
    gigachat_client_holder: GigaChatClientHolder = None

    @staticmethod
    def create(gigachat_client_holder: GigaChatClientHolder):
        return GigaChatExt(gigachat_client_holder=gigachat_client_holder,
                           **gigachat_client_holder.gigachat_properties.model_dump())

    @property
    def _client(self) -> GigaChatClient:
        return self.gigachat_client_holder.instance

    def _build_payload(self, messages: List[BaseMessage], **kwargs: Any) -> gm.Chat:
        current_verbose = self.verbose
        # Disable logging in _build_payload as it doesn't show the model field being set
        self.verbose = False
        payload = super()._build_payload(messages, **kwargs)
        self.verbose = current_verbose
        payload.model = self.model

        dictionary_of_parameters = self._move_messages_field_to_the_end(payload.dict(exclude_none=True))
        # Copy-past from GigaChat._build_payload
        if self.verbose:
            # noinspection PyDeprecation
            logger.warning(
                "Giga request: %s", json.dumps(dictionary_of_parameters, ensure_ascii=False)
            )
        return payload

    @staticmethod
    def _move_messages_field_to_the_end(dictionary: dict) -> dict:
        new_dict = dictionary.copy()
        messages = new_dict.pop("messages")
        new_dict["messages"] = messages
        return new_dict
