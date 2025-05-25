from pydantic import BaseModel

from lib.llmcommon.pydentic.yaml_alias_generator import to_hyphen


# This class helps to solve shell compatibility with different shells for pydentic-settings config in yaml format
# problem example for docker-python-3.9: bash: export: 'SUPER-test_var=test'
class BaseYamlModel(BaseModel):
    class Config:
        populate_by_name = True
        alias_generator = to_hyphen
