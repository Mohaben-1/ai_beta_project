install:
	uv sync

run:
	uv run python -m src.main

debug:
	uv run python -m pdb src/main.py

clean:
	rm -rf __pycache__ .pytest_cache output/*.json

lint:
	flake8 src
