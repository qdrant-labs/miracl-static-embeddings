.PHONY: test lint format format-check typecheck

all: test lint lint-check lint-unsafe format format-check typecheck

lint-check:
	$(info ****************** checking linting ******************)
	uv run ruff check

lint:
	$(info ****************** linting ******************)
	uv run ruff check --fix

lint-unsafe:
	$(info ****************** linting (unsafe fixes) ******************)
	uv run ruff check --fix --unsafe-fixes

format:
	$(info ****************** formatting ******************)
	uv run ruff format

format-check:
	$(info ****************** checking formatting ******************)
	uv run ruff format --check

typecheck:
	$(info ****************** type checking ******************)
	uv run ty check
