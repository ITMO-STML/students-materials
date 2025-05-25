import re

UNKNOWN_TYPE = 'Unknown'
junk_entities = {
    "",
    "нет именованных сущностей",
    "Текст: нет именованных сущностей",
    "Ответ: нет именованных сущностей",
}

def remove_duplicates(entities: str) -> str:
    entities = set(entity.strip() for entity in entities.split('\n'))
    entities = entities - junk_entities
    pattern = re.compile(r'(.+)\[([^\[\]]+)\]')
    entities_by_types = {}
    for entity in entities:
        match = pattern.match(entity)
        if match:
            entity_name = match.group(1).strip()
            entity_type = match.group(2).strip()
        else:
            entity_name = entity
            entity_type = UNKNOWN_TYPE


        if entity_type in entities_by_types:
            entities_by_types[entity_type].add(entity_name)
        else:
            entities_by_types[entity_type] = {entity_name}

    entities_by_types = dict(sorted(entities_by_types.items(), key = lambda item : len(item[1]), reverse=True))
    result = ''
    for key, value in entities_by_types.items():
        values = '\n'.join(sorted(value, key=str.casefold))
        result += f"{key}:\n{values}\n\n"

    return result.strip()
