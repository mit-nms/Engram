# Variables
PROJECT_DIR := $(shell pwd)
DOCKER_IMAGE := openevolve
VENV_DIR := $(PROJECT_DIR)/env
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all              - Install dependencies and run unit tests"
	@echo "  venv             - Create a virtual environment"
	@echo "  install          - Install Python dependencies"
	@echo "  install-dev      - Install development dependencies including optillm"
	@echo "  lint             - Run Black code formatting"
	@echo "  test             - Run unit tests only"
	@echo "  test-unit        - Run unit tests only (same as test)"
	@echo "  test-integration - Run integration tests with local LLM"
	@echo "  test-all         - Run both unit and integration tests"
	@echo "  docker-build     - Build the Docker image"
	@echo "  docker-run       - Run the Docker container with the example"
	@echo "  visualizer       - Run the visualization script"

.PHONY: all
all: install test

# Create and activate the virtual environment
.PHONY: venv
venv:
	python3 -m venv $(VENV_DIR)

# Install Python dependencies in the virtual environment
.PHONY: install
install: venv
	$(PIP) install -e .

# Install development dependencies including optillm for integration tests
.PHONY: install-dev
install-dev: venv
	$(PIP) install -e .
	$(PIP) install pytest optillm

# Run Black code formatting
.PHONY: lint
lint: venv
	$(PYTHON) -m black openevolve examples tests scripts

# Run unit tests only (fast, no LLM required)
.PHONY: test
test: venv
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

# Alias for test
.PHONY: test-unit
test-unit: test

# Run integration tests with local LLM (requires optillm)
.PHONY: test-integration
test-integration: install-dev
	@echo "Starting optillm server for integration tests..."
	@OPTILLM_API_KEY=optillm $(VENV_DIR)/bin/optillm --model google/gemma-3-270m-it --port 8000 &
	@OPTILLM_PID=$$! && \
	echo $$OPTILLM_PID > /tmp/optillm.pid && \
	echo "Waiting for optillm server to start..." && \
	sleep 10 && \
	echo "Running integration tests..." && \
	OPENAI_API_KEY=optillm $(PYTHON) -m pytest tests/integration -v --tb=short; \
	TEST_EXIT_CODE=$$?; \
	echo "Stopping optillm server..."; \
	kill $$OPTILLM_PID 2>/dev/null || true; \
	pkill -f "optillm.*8000" 2>/dev/null || true; \
	rm -f /tmp/optillm.pid; \
	exit $$TEST_EXIT_CODE

# Run integration tests with existing optillm server (for development)
.PHONY: test-integration-dev
test-integration-dev: venv
	@echo "Using existing optillm server at localhost:8000"
	@curl -s http://localhost:8000/health > /dev/null || (echo "Error: optillm server not running at localhost:8000" && exit 1)
	OPENAI_API_KEY=optillm $(PYTHON) -m pytest tests/integration -v

# Run all tests (unit first, then integration)
.PHONY: test-all
test-all: test test-integration

# Build the Docker image
.PHONY: docker-build
docker-build:
	docker build -t $(DOCKER_IMAGE) .

# Run the Docker container with the example
.PHONY: docker-run
docker-run:
	docker run --rm -v $(PROJECT_DIR):/app --network="host" $(DOCKER_IMAGE) examples/function_minimization/initial_program.py examples/function_minimization/evaluator.py --config examples/function_minimization/config.yaml --iterations 1000

# Run the visualization script
.PHONY: visualizer
visualizer:
	$(PYTHON) scripts/visualizer.py --path examples/
