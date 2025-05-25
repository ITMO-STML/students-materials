from pydantic.alias_generators import to_snake


def to_hyphen(field: str):
    field = to_snake(field)
    return field.replace("_", "-")
