import json
import re
from pathlib import Path
from .models import FunctionDefinition, FunctionCall

def	load_function_definitions(path: str) -> dict[str, FunctionDefinition]:
	with open(path, "r") as f:
			raw = json.load(f)
	return {fd["fn_name"]: FunctionDefinition(**fd) for fd in raw}

def	load_prompts(path: str) -> list[str]:
	with open(path, "r") as f:
			data = json.load(f)
	return [item["prompt"] for item in data]


def guess_function(prompt: str, function_defs: dict[str, FunctionDefinition]) -> tuple[str, dict]:
    if "sum" in prompt or "add" in prompt:
        nums = list(map(float, re.findall(r"\d+", prompt)))
        return "fn_add_numbers", {"a": nums[0], "b": nums[1]}
    elif "product" in prompt or "multiply" in prompt:
        nums = list(map(float, re.findall(r"\d+", prompt)))
        return "fn_multiply_numbers", {"a": nums[0], "b": nums[1]}
    elif "square root" in prompt:
        num = float(re.findall(r"\d+", prompt)[0])
        return "fn_get_square_root", {"a": num}
    elif "reverse" in prompt:
        text = re.findall(r"'([^']+)'", prompt)[0]
        return "fn_reverse_string", {"s": text}
    elif "even" in prompt:
        num = int(re.findall(r"\d+", prompt)[0])
        return "fn_is_even", {"n": num}
    elif "greet" in prompt.lower():
        name = prompt.split()[-1]
        return "fn_greet", {"name": name}
    elif "substitute" in prompt or "replace" in prompt:
        # More advanced regex parsing needed
        return "fn_substitute_string_with_regex", {
            "source_string": "TODO",
            "regex": "TODO",
            "replacement": "TODO"
        }
    else:
        raise ValueError(f"Could not map prompt: {prompt}")
