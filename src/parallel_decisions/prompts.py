"""Prompt construction.

The engine's own prompt shape, kept deliberately small: field names + descriptions
(the choices are enforced at the logit level, but describing them helps the model
choose sensibly), then the context, then the opening brace of the JSON object that
is never actually generated.
"""

from __future__ import annotations

from .schema import Schema


def field_line(name: str, description: str, choices: list[str] | None) -> str:
    text = description.strip()
    if choices:
        options = ", ".join(choices)
        text = f"{text} Options: {options}" if text else f"Options: {options}"
    return f'  "{name}": {text}'


def build_prompt(context: str, schema: Schema) -> str:
    lines = []
    for f in schema.fields.values():
        choices = None
        if not f.is_boolean:
            if f.choice_descriptions:
                choices = [f"{k} ({v})" if v else k for k, v in f.choice_descriptions.items()]
            else:
                choices = list(f.choices)
        lines.append(field_line(f.name, f.description, choices))

    system = "Classify JSON attributes:\n" + "\n".join(lines)
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        "<|im_start|>assistant\n{{\n"
    )
