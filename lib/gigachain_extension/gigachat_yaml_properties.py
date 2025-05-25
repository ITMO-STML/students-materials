from typing import Optional

#/home/karysheva@ad.speechpro.com/Документы/diploma/structured_summarization/lib/gigachain_extension/gigachat_properties.py

from lib.gigachain_extension.gigachat_properties import GigaChatProperties, ConnectionPoolProperties
from lib.llmcommon.pydentic.yaml_alias_generator import to_hyphen
from lib.llmcommon.pydentic.base_yaml_model import BaseYamlModel

from pydantic import BaseModel


class YamlConnectionPoolProperties(ConnectionPoolProperties, BaseYamlModel):
    pass


class GigaChatYamlProperties(GigaChatProperties, BaseYamlModel):
    connection_pool: Optional[YamlConnectionPoolProperties] = None