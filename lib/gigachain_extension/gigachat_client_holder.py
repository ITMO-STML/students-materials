from functools import cached_property

from lib.gigachain_extension.gigachat_client_ext import GigaChatClientExt
from lib.gigachain_extension.gigachat_properties import GigaChatProperties


class GigaChatClientHolder:

    def __init__(self, gigachat_properties: GigaChatProperties):
        self.gigachat_properties = gigachat_properties

    @cached_property
    # copy-past + small corrections from langchain_community.llms.gigachat._BaseGigaChat._client
    def instance(self) -> GigaChatClientExt:
        """Returns GigaChat API client"""
        return GigaChatClientExt(
            base_url=self.gigachat_properties.base_url,
            auth_url=self.gigachat_properties.auth_url,
            credentials=self.gigachat_properties.credentials,
            scope=self.gigachat_properties.scope,
            access_token=self.gigachat_properties.access_token,
            model=self.gigachat_properties.model,
            profanity_check=self.gigachat_properties.profanity_check,
            user=self.gigachat_properties.user,
            password=self.gigachat_properties.password,
            timeout=self.gigachat_properties.timeout,
            verify_ssl_certs=self.gigachat_properties.verify_ssl_certs,
            ca_bundle_file=self.gigachat_properties.ca_bundle_file,
            cert_file=self.gigachat_properties.cert_file,
            key_file=self.gigachat_properties.key_file,
            key_file_password=self.gigachat_properties.key_file_password,
            verbose=self.gigachat_properties.verbose,
            connection_pool=self.gigachat_properties.connection_pool,
        )
