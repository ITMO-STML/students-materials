from enum import StrEnum
from typing import Type, Tuple, Optional, List

from lib.gigachain_extension.gigachat_yaml_properties import GigaChatYamlProperties
from lib.llmcommon.pydentic.yaml_alias_generator import to_hyphen
from lib.llmcommon.pydentic.base_yaml_model import BaseYamlModel
from lib.llmcommon.pydentic.yaml_config_utils import YamlConfigSettingsSourceDeepCopy, YamlEnvSettingSource
from pydantic import PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource


import longdocpipeline.settings as settings
from longdocpipeline.pipeline.constants import TaskType, SplitModeType


class StandType(StrEnum):
    LOCAL = "local"
    EXTERNAL = "external"

class RunnableProperties(BaseYamlModel):
    refine_recursion_limit: str = 10000


class PreprocessingProperties(BaseYamlModel):
    class Config:
        use_enum_values = True

    split_mode: SplitModeType = SplitModeType.SENTENCE
    custom_chunk_size: Optional[PositiveInt] = None
    custom_chunk_overlap: Optional[int] = None
    custom_separator: Optional[str] = None


class ApplicationProperties(BaseSettings, extra="ignore"):
    model_config = SettingsConfigDict(yaml_file=settings.APPLICATION_CONFIG_PATHS,
                                      env_nested_delimiter='__',
                                      use_enum_values=True,
                                      populate_by_name=True,
                                      alias_generator=to_hyphen)
    gigachat: GigaChatYamlProperties
    token_limit: Optional[PositiveInt] = 32768
    preprocessing: PreprocessingProperties = PreprocessingProperties()
    runnable: RunnableProperties = RunnableProperties()
    task_types: List[TaskType] = [TaskType.SUM,
                                  TaskType.QA,
                                  TaskType.GENERAL,
                                  TaskType.NER_UNIQUE,
                                  TaskType.TRANSLATION]

    default_prompt_path: Optional[str] = settings.DEFAULT_PROMPT_PATH

    stand_type: Optional[StandType] = StandType.LOCAL

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: Type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            YamlEnvSettingSource(settings_cls),
            dotenv_settings,
            file_secret_settings,
            YamlConfigSettingsSourceDeepCopy(settings_cls),
        )


application_properties = ApplicationProperties()