# YouTube Summarizer

A tool for summarizing YouTube videos using AI.

## CLI

Set your OpenAI API key before running the tool:

```bash
export OPENAI_API_KEY="..."
```

Run against a YouTube URL:

```bash
uv run youtube_summarizer --url <URL>
# optional: --query <QUERY>
```

## Development Setup

1. Install uv:
   ```bash
   brew install uv
   ```

2. Sync the project environment:
   ```bash
   make dev
   ```

uv creates and manages `.venv` automatically from `pyproject.toml` and `uv.lock`.

## Building with PEX

To create a standalone PEX executable:

```bash
make build
```

This will create a `youtube_summarizer.pex` file that contains all dependencies.

## Running the Application

### Development Mode

```bash
uv run youtube_summarizer --url <URL>
```

### Using PEX

After building:

```bash
./youtube_summarizer.pex --url <URL>
```

## Cleanup

To remove build artifacts:

```bash
make clean
```
