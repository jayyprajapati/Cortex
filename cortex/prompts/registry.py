from cortex.prompts.templates.default import build_prompt as build_default_prompt


PROMPT_BUILDERS = {
    "default": build_default_prompt,
}


def get_prompt_builder(app_name):
    normalized_name = (app_name or "default").strip().lower()
    return PROMPT_BUILDERS.get(normalized_name, PROMPT_BUILDERS["default"])
