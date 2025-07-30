# YouTube Summarizer

A tool for summarizing YouTube videos using AI.

## CLI

```
PYTHONPATH="." python youtube_summarizer/youtube.py --url <URL>
# optional: --query <QUERY>
```

## Development Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install development dependencies:
   ```bash
   make dev
   ```

## Building with PEX

To create a standalone PEX executable:

```bash
make build
```

This will create a `youtube_summarizer.pex` file that contains all dependencies.

## Running the Application

### Development Mode
```bash
python -m youtube_summarizer.youtube
```

### Using PEX
After building:
```bash
./youtube_summarizer.pex
```

## Cleanup

To remove build artifacts:
```bash
make clean
```
