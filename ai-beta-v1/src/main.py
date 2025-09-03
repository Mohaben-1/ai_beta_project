from pathlib import Path
import json
from .runner import load_function_definitions, load_prompts, guess_function
from .models import FunctionCall

def main():
    base = Path(__file__).resolve().parent.parent
    defs_path = base / "input" / "functions_definition.json"
    prompts_path = base / "input" / "function_calling_tests.json"
    output_path = base / "output" / "function_calling_name.json"

    function_defs = load_function_definitions(defs_path)
    prompts = load_prompts(prompts_path)

    results = []
    for prompt in prompts:
        try:
            fn_name, args = guess_function(prompt, function_defs)
            call = FunctionCall(prompt=prompt, fn_name=fn_name, args=args)
            results.append(call.model_dump())
        except Exception as e:
            print(f"[ERROR] {prompt}: {e}")

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()