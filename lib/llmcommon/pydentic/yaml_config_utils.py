import os
import re
from pathlib import Path
from typing import Union, Any

from pydantic.v1.utils import deep_update
from pydantic_settings.sources import ConfigFileSourceMixin, PathType, YamlConfigSettingsSource, EnvSettingsSource
from pydantic_settings.sources import parse_env_vars

class ConfigFileSourceMixinDeepCopy(ConfigFileSourceMixin):
    """
    Например есть два файла: config1.yaml и config2.yaml:
        config1.yaml
            # Настройки базы данных
            database:
              host: localhost
              user: user1
              password: pass1

        config2.yaml
            # Настройки базы данных
            database:
              port: 5432
              password: pass2

    После объединения yaml файлов будут установлены следующие параметры:
        database:
          host: localhost
          port: 5432
          user: user1
          password: pass2  # Переопределенный пароль из config2.yaml
    """
    def _read_files(self, files: Union[PathType,None]) -> dict[str, Any]:
        if files is None:
            return {}
        if isinstance(files, (str, os.PathLike)):
            files = [files]
        vars: dict[str, Any] = {}
        for file in files:
            file_path = Path(file).expanduser()
            if file_path.is_file():
                # file_content может быть null(NoneType) в случае если файл будет пустым или все строчки закомментированны
                file_content = self._read_file(file_path)
                if file_content:
                    vars = deep_update(vars, file_content)
        return vars


class YamlConfigSettingsSourceDeepCopy(YamlConfigSettingsSource, ConfigFileSourceMixinDeepCopy):
    """
    Этот класс нужен для корректного мержа yaml-конфигов
    """
    pass

class YamlEnvSettingSource(EnvSettingsSource):
    """
    Use only for application.yaml with dashes.

    This source repalce all underscore by dashes in env variables for correct alias mapping in BaseSttings.

    This fix needs to use shells with IEEE Std 1003.1-2001 format env variables(docker or sh/bash).
    """

    def _load_env_vars(self):
        return  {re.sub(r'([a-zA-Z])_([a-zA-Z])', r'\1-\2', k): v for k, v in super()._load_env_vars().items()}
