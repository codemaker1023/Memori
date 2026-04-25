# Memori

The memory fabric for enterprise AI.

Memori adds persistent memory to your LLM applications without changing your architecture. It is model, framework, and datastore agnostic.

---

## Why Memori

Memori captures LLM interactions, enriches them, and makes them retrievable as high-quality context for future generations.

- Low integration overhead: wrap your existing LLM client and keep your current stack.
- Attribution-aware memory: organize memory by `entity`, `process`, and `session`.
- Asynchronous augmentation: extract structured memory without adding user-facing latency.
- Flexible infrastructure: supports multiple models, frameworks, and databases.

## Install

```bash
pip install memori
```

Optional one-time optimization:

```bash
python -m memori setup
```

## Quickstart (OpenAI + SQLite)

```python
import os
import sqlite3

from memori import Memori
from openai import OpenAI


def get_sqlite_connection():
    return sqlite3.connect("memori.db")


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memori = Memori(conn=get_sqlite_connection).llm.register(client)

# Required so Memori can store contextual memory for this actor/workflow.
memori.attribution(entity_id="user_123", process_id="assistant_demo")
memori.config.storage.build()

client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}],
)

# Wait for async augmentation in short-lived scripts.
memori.augmentation.wait()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "What is my favorite color?"}],
)
print(response.choices[0].message.content)
```

## Core Concepts

### Attribution

Attribution tells Memori who the interaction belongs to and what workflow produced it.

```python
memori.attribution(entity_id="user_123", process_id="support_agent")
```

If attribution is not set, memory cannot be built correctly.

### Session Management

Sessions group related interactions together.

```python
memori.new_session()
```

Reuse an existing session:

```python
session_id = memori.config.session_id
memori.set_session(session_id)
```

### Schema Setup

Build or migrate storage schema during deployment or after upgrades:

```python
Memori(conn=db_session_factory).config.storage.build()
```

## Supported Integrations

### LLM Providers

- OpenAI
- Anthropic
- Google Gemini
- xAI (Grok)
- Bedrock (via LangChain)

Supports sync, async, streamed, and unstreamed interaction modes.

### Frameworks and Platforms

- LangChain
- Agno
- Pydantic AI
- Nebius AI Studio

### Database Connection Methods

- SQLAlchemy
- DB API 2.0 (PEP 249 drivers such as `psycopg`, `pymysql`, `sqlite3`, and others)
- Django ORM integration

### Datastores

- SQLite
- PostgreSQL
- MySQL
- MariaDB
- Oracle
- MongoDB
- Neon
- Supabase
- CockroachDB

## Advanced Augmentation

Memori can enrich captured conversations into structured memory such as:

- attributes
- events
- facts
- people
- preferences
- relationships
- rules
- skills

Augmentation runs asynchronously and is available without an account (rate limited).

Get higher limits:

```bash
python -m memori sign-up <email_address>
```

Set your API key:

```bash
export MEMORI_API_KEY=<api_key>
```

Check usage quota:

```bash
python -m memori quota
```

## CLI

```bash
python -m memori
```

See full CLI docs in `docs/cli.md`.

## Contributing

Contributions are welcome. See `CONTRIBUTING.md` for setup, standards, and PR guidance.

## License

Apache 2.0. See `LICENSE`.
