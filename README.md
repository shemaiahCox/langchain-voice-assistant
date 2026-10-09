# langchain-voice-assistant

Scaffold for a LangChain voice assistant.

## Prerequisites

- Python 3.12 (`/opt/homebrew/bin/python3.12`)
- [Ollama](https://ollama.com), if you run a local model through `langchain-ollama`

## Setup

The virtual environment in this folder is already created. On a new machine, recreate it:

```bash
cd ~/Documents/Development/ai/langchain-voice-assistant
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Daily use

```bash
cd ~/Documents/Development/ai/langchain-voice-assistant
source .venv/bin/activate
```

In Cursor, open this folder and choose **Python: Select Interpreter** → `.venv/bin/python`.

To add a package:

```bash
source .venv/bin/activate
pip install <package>
pip freeze > requirements.txt
```

## Installed packages

`requirements.txt` pins the environment. The packages installed for this project are `langchain`, `langchain-ollama`, and `python-dotenv`.

## Configuration

Copy `.env.example` to `.env`. `.env` is gitignored.

| Variable | Required | Purpose |
| --- | --- | --- |
| `LANGSMITH_API_KEY` | No | Enables LangSmith tracing when set |
| `LANGSMITH_TRACING` | No | Set to `true` to turn tracing on |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `LANGSMITH_ENDPOINT` | No | LangSmith API endpoint |

## Remote

`git@github.com:shemaiahCox/langchain-voice-assistant.git`
