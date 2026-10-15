.PHONY: test lint format format-check typecheck

all: test lint lint-check format format-check typecheck

lint-check:
	$(info ****************** checking linting ******************)
	uv run ruff check

lint:
	$(info ****************** linting ******************)
	uv run ruff check --fix

format:
	$(info ****************** formatting ******************)
	uv run ruff format

format-check:
	$(info ****************** checking formatting ******************)
	uv run ruff format --check

typecheck:
	$(info ****************** type checking ******************)
	uv run ty check
