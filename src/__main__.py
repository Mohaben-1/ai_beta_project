import json
from pydantic import BaseModel
from src import Small_LLM_Model
import numpy as np

# File paths
PROMPTS_FILE = 'input/function_calling_tests.json'
FUNCTION_DEFINITIONS_FILE = 'input/function_definitions.json'
OUTPUT_FILE = 'output/function_calling_name.json'

# Initialize the model
model = Small_LLM_Model()

# Load the vocabulary file from the model
try:
    vocab_path = model.get_path_to_vocabulary_json()
except Exception as e:
    exit(f"Error when fetching the vocab file path: {e}")

# Read the vocabulary file
try:
    with open(vocab_path, 'r') as vocab_file:
        vocab_raw = vocab_file.read()
except FileNotFoundError:
    exit(f"{vocab_path} not found")
except IOError:
    exit(f"Error reading {vocab_path}")

# Parse the vocabulary JSON
try:
    vocab = json.loads(vocab_raw)
except json.JSONDecodeError as e:
    exit(f"Invalid JSON in vocabulary file: {e}")


# Define the structure of a prompt
class Prompt(BaseModel):
    prompt: str


# Define the structure of a function definition
class FunctionDefinition(BaseModel):
    fn_name: str
    args_names: list[str]
    args_types: dict
    return_type: str


def main():
    print("Loading function definitions...")
    function_definitions_raw = read_file_content(FUNCTION_DEFINITIONS_FILE)

    print("Loading prompts...")
    prompts_raw = read_file_content(PROMPTS_FILE)

    print("Parsing function definitions...")
    function_definitions = parse_function_definitions(function_definitions_raw)
    if not function_definitions:
        exit("No function definitions found!")

    print("Initializing prompts...")
    prompts, raw_prompts = initialize_prompts(prompts_raw, function_definitions_raw)

    print(f"\nTotal prompts: {len(prompts)}")
    for index, (prompt, raw_prompt) in enumerate(zip(prompts, raw_prompts), start=1):
        print(f"\nProcessing prompt {index}:")
        print(f"    Original Prompt: {raw_prompt}")
        print("    Selecting function name...")
        selected_function_name = select_function_name(prompt, function_definitions)
        print(f"        Selected Function Name: {selected_function_name}")

    print("\nProcessing complete!")


def set_function_arguments(function_definition: FunctionDefinition, prompt: str):
    """
    Generate arguments for a function based on the prompt and function definition.
    """
    for argument_name in function_definition.args_names:
        prompt += f"The value of the argument {argument_name} should be {argument_name}: "
        input_ids = tokenize_text(prompt)
        logits = model.get_logits_from_input_ids(input_ids)
        max_index = np.argmax(logits)
        token = [key for key, value in vocab.items() if value == max_index]
        print(token)
    return {}


def select_function_name(prompt: str, function_definitions: list[FunctionDefinition]) -> str:
    """
    Select the most appropriate function name based on the prompt.
    """
    function_names = [fn.fn_name for fn in function_definitions]
    input_ids = tokenize_text(prompt)
    tokens = []
    token_ids = []

    while True:
        logits = model.get_logits_from_input_ids(input_ids + token_ids)
        max_index = np.argmax(logits)
        token = [key for key, value in vocab.items() if value == max_index]
        tokens += token

        # Filter function names based on matching tokens
        matching_functions = [
            fn_name for fn_name in function_names if all(tok in fn_name for tok in tokens)
        ]

        if len(matching_functions) == 1:
            return matching_functions[0]
        elif not matching_functions:
            if tokens:
                tokens.pop(0)  # Remove the first token
                tokens.pop(-1)  # Remove the last token
                continue
            else:
                return function_names[0]  # Default to the first function name

        token_ids += tokenize_text(token[0])


def read_file_content(file_path: str) -> str:
    """
    Read the content of a file and return it as a string.
    """
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        exit(f"{file_path} not found")
    except IOError:
        exit(f"Error reading {file_path}")


def initialize_prompts(prompts_raw: str, function_definitions_raw: str) -> tuple[list[str], list[str]]:
    """
    Initialize prompts by combining function definitions and raw prompts.
    """
    try:
        data = json.loads(prompts_raw)
        if not isinstance(data, list):
            exit(f"Invalid structure in: {PROMPTS_FILE}")

        formatted_prompts = []
        raw_prompts = []

        for obj in data:
            try:
                prompt_obj = Prompt(**obj)
            except Exception:
                exit(f"Invalid structure in: {PROMPTS_FILE}")

            prompt = (
                f"Function definitions:\n{function_definitions_raw}\n"
                f"Prompt: {prompt_obj.prompt}\n"
                "The function name of the most suitable function to answer the prompt is: "
            )
            # Clean up the prompt text
            prompt = prompt.replace('_', " ").replace('-', " ").replace('"', " ").replace("'", " ")
            raw_prompts.append(prompt_obj.prompt)
            formatted_prompts.append(prompt)

        return formatted_prompts, raw_prompts
    except json.JSONDecodeError as e:
        exit(f"Invalid JSON in prompts file: {e}")


def parse_function_definitions(content: str) -> list[FunctionDefinition]:
    """
    Parse the function definitions from a JSON string.
    """
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            exit(f"Invalid structure in: {FUNCTION_DEFINITIONS_FILE}")

        function_definitions = []
        for obj in data:
            try:
                function_definition = FunctionDefinition(**obj)
            except Exception:
                exit(f"Invalid structure in: {FUNCTION_DEFINITIONS_FILE}")
            function_definitions.append(function_definition)

        return function_definitions
    except json.JSONDecodeError as e:
        exit(f"Invalid JSON in function definitions file: {e}")


def tokenize_text(text: str) -> list[int]:
    """
    Tokenize a given text into a list of token IDs using the vocabulary.
    """
    token_ids = []
    remaining_text = text.lower()

    while remaining_text:
        best_match = ""
        best_id = None

        for token, token_id in vocab.items():
            if remaining_text.startswith(token.lower()) and len(token) > len(best_match):
                best_match = token.lower()
                best_id = token_id

        if best_match:
            token_ids.append(best_id)
            remaining_text = remaining_text[len(best_match):]
        else:
            remaining_text = remaining_text[1:]

    return token_ids


if __name__ == "__main__":
    main()