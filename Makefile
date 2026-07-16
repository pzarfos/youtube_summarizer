.PHONY: all build dev sync lock install clean run

UV ?= uv
APP := youtube_summarizer
PEX_FILE := youtube_summarizer.pex
PEX_ROOT := build/pex-root
BUILD_REQUIREMENTS := build/requirements.txt

# Usage
all:
	@echo "make all        Show this message"
	@echo "make dev        Sync the uv development environment"
	@echo "make sync       Sync runtime dependencies only"
	@echo "make lock       Update uv.lock"
	@echo "make build      Build the PEX file"
	@echo "make deploy     Deploy the PEX file to ~/.local/bin"
	@echo "make outdated   List outdated packages"
	@echo "make upgrade    Upgrade packages"
	@echo "make clean      Clean build artifacts"
	@echo "make run        Run the application with uv"

# Build the PEX file
build: $(BUILD_REQUIREMENTS)
	$(UV) run --locked --group dev pex --pip-version latest-compatible --pex-root $(PEX_ROOT) . -r $(BUILD_REQUIREMENTS) -c $(APP) -o $(PEX_FILE)

$(BUILD_REQUIREMENTS): pyproject.toml uv.lock
	@[ -d build ] || mkdir -p build
	$(UV) export --quiet --locked --no-dev --no-emit-project --no-hashes --output-file $(BUILD_REQUIREMENTS)

deploy: build
	@[ -d "${HOME}/.local/bin" ] || mkdir -p "${HOME}/.local/bin"
	/bin/cp -pf $(PEX_FILE) "${HOME}/.local/bin/$(APP)"

# Install development dependencies
dev:
	$(UV) sync --locked

# Install runtime dependencies only
sync:
	$(UV) sync --locked --no-dev

# Update the lockfile
lock:
	$(UV) lock

outdated:
	@echo "Finding outdated packages..."
	uv tree --outdated

upgrade:
	@echo "Upgrading packages..."
	uv lock --upgrade
	uv sync --all-groups

# Clean build artifacts
clean:
	/bin/rm -rf __pycache__ build dist youtube_summarizer.egg-info $(PEX_FILE)

# Run the application
run:
	$(UV) run --locked $(APP)
