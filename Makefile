
PID := $(shell cat streamlit.pid)

.PHONY:  test coverage-report test-pip jupyter install-poetry setuppy


install-poetry:
	@bash scripts/install-poetry.sh


install:
	@echo "Installing ideas-python-utils..."
	poetry check --lock || poetry lock
	poetry install --verbose


install-test:
	@echo "Installing ideas-python-utils & dependencies for testing..."
	poetry check --lock || poetry lock
	poetry install --extras "plotting extras test plots test_isx ideas_commons ideas_schemas" --verbose


test: install-poetry install-test
	@echo "Running tests with coverage..."
	@poetry run coverage run -m pytest -sx --failed-first $(TEST_ARGS)

	
coverage-report: test
	poetry run coverage html --omit="*/test*"
	open htmlcov/index.html


test-pip:
	@echo "Running tests for code installed with pip:"
	@coverage run -m pytest -sx 

jupyter: install
	@echo "Installing this kernel in jupyter"
	poetry run python -m ipykernel install --user --name ideas_python_utils



setuppy: 
	poetry run poetry2setup > setup.py