import json
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
