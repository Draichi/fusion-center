# Agent v2 (PydanticAI)

A cleaner implementation of the Deep Research Agent using [PydanticAI](https://ai.pydantic.dev/) instead of LangChain/LangGraph.

## Why v2?

- **Better visibility**: Explicit Python control flow instead of implicit graph routing
- **Simpler architecture**: ~350 lines vs ~700 lines in the original
- **Native Pydantic**: Direct structured output support
- **Multi-step reasoning**: Preserved from v1 (decompose → hypothesize → reflect → verify)

## Architecture

```
Plan → Decompose → Hypothesize → [Gather → Analyze → Reflect]* → Verify → Synthesize
                                       ↑__________________|
                                       (iterative loop)
```

### Dual-LLM Configuration

| LLM | Default Model | Used For |
|-----|---------------|----------|
| Structured | `ollama:qwen2.5:7b` | Planning, Decomposition, Hypothesis, Analysis |
| Thinking | `openai:gpt-oss:120b-cloud` | Reflection, Verification, Synthesis |

## Usage

```bash
# Start MCP server first
uv run python -m src.mcp_server.server --transport sse

# Run agent
uv run python -m src.agent_v2 "Analyze military activity in Ukraine"

# Custom models
uv run python -m src.agent_v2 --thinking-model openai:gpt-4 "Deep analysis"

# JSON output
uv run python -m src.agent_v2 --json "Search for news"
```

## Debug Prompts

Set `DEBUG_PROMPTS=1` to log all prompts to files:

```bash
DEBUG_PROMPTS=1 uv run python -m src.agent_v2 "Your query"
# → logs/prompts_20241220_123456/
#   ├── 01_planning.txt
#   ├── 02_decomposition.txt
#   ├── 03_hypothesis.txt
#   ├── 04_gathering_iter1.txt
#   └── ...
```

## Files

| File | Purpose |
|------|---------|
| `agent.py` | Main `ResearchAgent` class with dual-LLM |
| `schemas.py` | Pydantic output models (SITREP, Analysis, etc.) |
| `prompts.py` | System prompts for each phase |
| `state.py` | `ResearchContext` dataclass |
| `debug.py` | Prompt logging utility |
| `__main__.py` | CLI entry point |
