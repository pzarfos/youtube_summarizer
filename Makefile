.PHONY: build dev clean run

# Build the PEX file
build:
	pip install -r requirements.txt
	pex . -r requirements.txt -c youtube_summarizer -o youtube_summarizer.pex

# Install development dependencies
dev:
	pip install -r requirements.txt
	pip install -e .

# Clean build artifacts
clean:
	/bin/rm -rf __pycache__ build youtube_summarizer.egg-info youtube_summarizer.pex

# Run the PEX file
run:
	./youtube_summarizer.pex
