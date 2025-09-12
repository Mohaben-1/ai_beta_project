install:
	@uv sync

run:
	@uv run python -m src

debug:
	@uv run python -m pdb src/__main__.py

clean:
	@rm -rf ./src/__pycache__ ./.venv
	
lint:
	@flake8 ./src/__main__.py
