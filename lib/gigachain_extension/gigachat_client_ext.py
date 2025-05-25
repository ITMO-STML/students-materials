import logging
import threading
from functools import cached_property
from typing import Union

import httpx
from gigachat import GigaChat
from gigachat.client import _get_kwargs


logger = logging.getLogger(__name__)

thread_local = threading.local()

_CONNECTION_POOL_KEY = "connection_pool"


class GigaChatClientExt(GigaChat):
    _update_token_lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        if _CONNECTION_POOL_KEY in kwargs:
            self.__connection_pool_properties = kwargs[_CONNECTION_POOL_KEY]
            del kwargs[_CONNECTION_POOL_KEY]
        super().__init__(*args, **kwargs)

    @cached_property
    def _client(self) -> httpx.Client:
        # Add connection pooling to limit number of connection to Gigachat
        kwargs = _get_kwargs(self._settings)
        if self.__connection_pool_properties is not None:
            kwargs["limits"] = httpx.Limits(
                max_keepalive_connections=self.__connection_pool_properties.max_keepalive_connections,
                max_connections=self.__connection_pool_properties.max_connections,
                keepalive_expiry=self.__connection_pool_properties.keepalive_expiry)
        return httpx.Client(**kwargs)

    def _check_validity_token(self) -> bool:
        thread_local.before_check_token = self._access_token
        logger.debug("before check token: %s", thread_local.before_check_token)
        # Do not call gigachat if token is being updated.
        return hasattr(self, "_is_token_valid") and self._is_token_valid

    # Make sure token is only modified in _update_token
    def _reset_token(self) -> None:
        pass

    def _update_token(self) -> None:
        GigaChatClientExt._update_token_lock.acquire()
        logger.debug("_update_token_lock acquired")
        try:
            if thread_local.before_check_token != self._access_token:
                logger.debug(
                    f"Skip updating token. prev: {thread_local.before_check_token}. current: {self._access_token}")
                # Token is update by another thread.
                return
            self._is_token_valid = False
            logger.debug("token is being updated. prev: %s, current: %s", thread_local.before_check_token, self._access_token)
            super()._update_token()
            logger.debug("current: %s", self._access_token)
            self._is_token_valid = True
        finally:
            GigaChatClientExt._update_token_lock.release()
            logger.debug("_update_token_lock released")

    def tokens_count(self, *args, **kwargs):
        logger.info(f"Tokens count request: {{\"model\": {kwargs['model'] if kwargs.get('model') else self._settings.model}, \"input\": {self._tokens_count_parse_arguments(*args, **kwargs)}}}")
        result = super().tokens_count(*args, **kwargs)
        logger.info(f"Tokens count response: {result[0].dict()}")
        return result

    @staticmethod
    def _tokens_count_parse_arguments(*args: Union[tuple, list, None], **kwargs: dict) -> list:
        if args:
            if isinstance(args[0], list):
                args_argument = args[0]
            else:
                args_argument = [args[0]]
        else:
            args_argument = []

        kwargs_ = kwargs.copy()
        if "model" in kwargs_:
            kwargs_.pop("model")
        kwargs_argument = [str(value) for key, value in kwargs_.items()]
        return args_argument + kwargs_argument
