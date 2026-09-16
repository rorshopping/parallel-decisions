"""Prompt construction.

The engine's own prompt shape, kept deliberately small: field names + descriptions
(the choices are enforced at the logit level, but describing them helps the model
choose sensibly), then the context, then the opening brace of the JSON object that
is never actually generated.

Multi-select fields are described as one line naming every option, and are answered
one yes/no question per option at keys like `"actions[1]": `.
"""

from __future__ import annotations

import re

from .schema import Schema


def _options_already_described(description: str, choices: list[str]) -> bool:
    """True when every choice is already named in the description.

    Repeating an option list the description already carries changes the prompt
    text for no gain (it costs prompt tokens and shifts the schema block). The
    reference evaluation passes its option lists inside the description, so this
    rule is what keeps the two prompts identical.
    """
    if not description:
        return False
    return all(re.search(rf"\b{re.escape(choice)}\b", description) for choice in choices)


def field_line(name: str, description: str, choices: list[str] | None) -> str:
    text = description.strip()
    if choices and not _options_already_described(text, choices):
        # "; " separates options: descriptions often contain commas, and this is the
        # exact shape the reference evaluation used
        options = "; ".join(choices)
        text = f"{text} Options: {options}" if text else f"Options: {options}"
    return f'  "{name}": {text}'


def _option_labels(f) -> list[str]:
    """Human-readable option list: 'name=description' when descriptions exist.

    The `k=v; k=v` shape is what the reference evaluation used, so a schema that
    supplies descriptions produces the same prompt text here as it did there.
    """
    if f.choice_descriptions:
        return [f"{k}={v}" if v else k for k, v in f.choice_descriptions.items()]
    return list(f.choices)


def build_prompt(context: str, schema: Schema) -> str:
    lines = []
    for f in schema.fields.values():
        if f.is_multi:
            options = ", ".join(_option_labels(f))
            text = f"{f.description.strip()} " if f.description.strip() else ""
            lines.append(
                f'  "{f.name}": {text}[multi-select] answer true or false for each '
                f'indexed key "{f.name}[i]" (option i belongs in the subset). '
                f"Options: {options}")
            continue
        if not f.is_boolean:
            lines.append(field_line(f.name, f.description, _option_labels(f)))
        elif f.choice_descriptions:
            # boolean with true/false definitions: keep the same option shape
            details = "; ".join(f"{k}: {v}" for k, v in f.choice_descriptions.items() if v)
            text = f"{f.description} ({details})" if details else f.description
            lines.append(field_line(f.name, text, None))
        else:
            lines.append(field_line(f.name, f.description, None))

    system = "Classify JSON attributes:\n" + "\n".join(lines)
    # A single opening brace: the JSON is never generated past this point, and the
    # field suffixes are compiled as `  "name": ` relative to exactly one `{`.
    return (
        "<|im_start|>system\n"
        f"{system}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context}<|im_end|>\n"
        "<|im_start|>assistant\n{\n"
    )
