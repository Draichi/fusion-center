# Project Overwatch 🌐

**Fusion Center - MCP Server and AI Agent for OSINT and Geopolitical Intelligence**

Project Overwatch is an autonomous intelligence system that combines a Model Context Protocol (MCP) server with an AI agent for Open Source Intelligence (OSINT) analysis. It correlates data from news media, satellite imagery, and internet infrastructure monitoring.

## 🎯 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FUSION CENTER                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     MCP/SSE      ┌──────────────────────┐    │
│   │  Overwatch  │ ◄──────────────► │    MCP Server        │    │
│   │    Agent    │                  │  (project-overwatch) │    │
│   │   (LLM)     │                  └──────────────────────┘    │
│   └─────────────┘                            │                  │
│         │                          ┌─────────┴─────────┐       │
│         │                          ▼         ▼         ▼       │
│         ▼                    ┌─────────┐ ┌──────┐ ┌────────┐   │
│   ┌───────────┐              │  GDELT  │ │ NASA │ │  IODA  │   │
│   │ Analysis  │              │  News   │ │FIRMS │ │ Outage │   │
│   │ & Reports │              └─────────┘ └──────┘ └────────┘   │
│   └───────────┘                                                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Features

### MCP Server Tools

| Category | Tool | Description |
|----------|------|-------------|
| 📰 **News** | `search_news` | Search GDELT for global news |
| 🛰️ **Satellite** | `detect_thermal_anomalies` | NASA FIRMS fire/explosion detection |
| 🌐 **Cyber** | `check_connectivity` | IODA internet outage detection |
| 🌐 **Cyber** | `check_traffic_metrics` | Cloudflare Radar analysis |
| 📱 **Telegram** | `search_telegram` | Search OSINT Telegram channels |
| 📱 **Telegram** | `get_channel_info` | Get Telegram channel metadata |
| 📱 **Telegram** | `list_osint_channels` | List curated OSINT channels |
| 🚫 **Sanctions** | `search_sanctions` | Search sanctions lists (stub) |
| 🚫 **Sanctions** | `screen_entity` | Entity compliance screening (stub) |

### AI Agent

- Autonomous OSINT analysis
- Multi-source data correlation
- LLM-driven tool selection
- Structured intelligence reports
- **Multi-step Reasoning** with hypothesis testing, self-reflection, and verification

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
cd fusion-center

# Create virtual environment and install
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[agent]"  # Include agent dependencies

# Copy environment template
cp .env.example .env
```

### Configuration

Edit `.env`:

```bash
# Required for satellite data
NASA_FIRMS_API_KEY=your_key_here

# Required for Telegram monitoring (get from https://my.telegram.org)
TELEGRAM_API_ID=your_api_id
TELEGRAM_API_HASH=your_api_hash
# After setting these, run: python scripts/telegram_auth.py

# For agent (choose based on provider)
GOOGLE_API_KEY=your_google_key      # for gemini provider
XAI_API_KEY=your_xai_key            # for grok provider
# ollama and docker providers don't need API keys

# Optional
LOG_LEVEL=INFO
MCP_SERVER_PORT=8080
```

## 📦 Running

### Start the MCP Server

```bash
# HTTP/SSE mode (default)
python -m src.mcp_server.server --transport sse --port 8080

# Or stdio mode
python -m src.mcp_server.server --transport stdio
```

### Run the Agent

```bash
# Start analysis task
python -m src.agent "Analyze military activity in Ukraine over the past week"

# With custom server
python -m src.agent --server http://localhost:9000/sse "Check internet status in Iran"

# Output as JSON
python -m src.agent --json "Search for news about protests in China"
```

### Run Both Together

```bash
# Terminal 1: Start MCP Server
python -m src.mcp_server.server --transport sse --port 8080

# Terminal 2: Run Agent
python -m src.agent "Correlate thermal anomalies with news near Kyiv"
```

## 📁 Project Structure

```
fusion-center/
├── pyproject.toml              # Dependencies and config
├── .env.example                # Environment template
├── README.md
│
├── scripts/
│   └── telegram_auth.py        # One-time Telegram authentication
│
├── output/                     # Research outputs
│   └── {session_id}/
│       ├── report.md           # Final intelligence report
│       ├── reasoning.log       # Full reasoning trace
│       └── state.json          # Complete state snapshot
│
└── src/
    ├── __init__.py
    │
    ├── mcp_server/             # 🔧 MCP Server
    │   ├── __init__.py
    │   ├── server.py           # Server entry point
    │   └── tools/
    │       ├── geo.py          # NASA FIRMS
    │       ├── news.py         # GDELT
    │       ├── cyber.py        # IODA/Cloudflare
    │       ├── telegram.py     # Telegram OSINT channels
    │       └── sanctions.py    # OpenSanctions (stub)
    │
    ├── agent/                  # 🤖 AI Agent
    │   ├── __init__.py
    │   ├── __main__.py         # CLI entry point
    │   ├── core.py             # Agent exports
    │   ├── graph.py            # LangGraph definition
    │   ├── nodes.py            # Graph nodes (incl. multi-step reasoning)
    │   ├── state.py            # Agent state schema
    │   ├── tools.py            # MCP tool executor
    │   └── prompts/            # System prompts & reasoning prompts
    │
    └── shared/                 # 🔗 Shared Code
        ├── __init__.py
        ├── config.py           # Centralized config
        ├── logger.py           # Rich logging
        └── output_writer.py    # Report & reasoning log writer
```

## 🔌 Integration Examples

### Python Client

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async def analyze():
    async with sse_client("http://127.0.0.1:8080/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Search news
            result = await session.call_tool(
                "search_news",
                arguments={
                    "keywords": "military activity",
                    "country_code": "UA",
                    "timespan": "3d"
                }
            )
            print(result)
```

### Using the Agent Programmatically

```python
from src.agent.core import OverwatchAgent

async def run_analysis():
    agent = OverwatchAgent()
    result = await agent.run_analysis(
        task="Analyze internet outages in Iran and correlate with news",
        context={"country_code": "IR"}
    )
    return result
```

## 📊 Data Sources

| Source | Description | Auth |
|--------|-------------|------|
| [GDELT](https://www.gdeltproject.org/) | Global news monitoring | Free |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Satellite fire detection | Free API key |
| [IODA](https://ioda.inetintel.cc.gatech.edu/) | Internet outages | Free |
| [Cloudflare Radar](https://radar.cloudflare.com/) | Traffic analytics | Free (limited) |
| [Telegram](https://my.telegram.org/) | OSINT channel monitoring | Free API credentials |
| [OpenSanctions](https://www.opensanctions.org/) | Sanctions database | Planned |

## 🧪 Development

```bash
# Install dev dependencies
uv pip install -e ".[dev,agent]"

# Linting
ruff check src/

# Type checking
mypy src/

# Test server
python -m src.mcp_server.server --transport sse --port 8080
```

## 🧠 Multi-step Reasoning

The agent uses advanced multi-step reasoning for deeper analysis:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           MULTI-STEP REASONING FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐             │
│   │ PLANNING │───►│ DECOMPOSING │───►│ HYPOTHESIZING│───►│ GATHERING │◄────┐       │
│   └──────────┘    └─────────────┘    └──────────────┘    └─────┬─────┘     │       │
│                                                                 │           │       │
│                                                           (update hyp)     │       │
│                                                                 ▼           │       │
│                                                           ┌───────────┐     │       │
│                                                           │ ANALYZING │     │       │
│                                                           └─────┬─────┘     │       │
│                                                                 │           │       │
│                                                    ┌────────────┼───────────┘       │
│                                                    │ (follow-up)│                   │
│                                                    │            ▼                   │
│                                                    │      ┌────────────┐            │
│                                                    │      │ REFLECTING │◄──────┐    │
│                                                    │      └─────┬──────┘       │    │
│                                                    │            │              │    │
│                                                    │ (gaps)     │              │    │
│                                                    └────────────┤     (not ready)   │
│                                                                 ▼              │    │
│                                                           ┌────────────┐       │    │
│                                                           │ CORRELATING│       │    │
│                                                           └─────┬──────┘       │    │
│                                                                 ▼              │    │
│                                                           ┌───────────┐        │    │
│                                                           │ VERIFYING │────────┘    │
│                                                           └─────┬─────┘             │
│                                                                 │ (ready)           │
│                                                                 ▼                   │
│                                                           ┌─────────────┐           │
│                                                           │ SYNTHESIZING│           │
│                                                           └──────┬──────┘           │
│                                                                  ▼                  │
│                                                            ┌──────────┐             │
│                                                            │ COMPLETE │             │
│                                                            └──────────┘             │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Phase Descriptions

| Phase | Description |
|-------|-------------|
| **Planning** | Creates research plan with objectives, regions, keywords, and initial queries |
| **Decomposing** | Breaks complex tasks into manageable sub-tasks, assesses complexity |
| **Hypothesizing** | Generates testable hypotheses with support/refutation criteria |
| **Gathering** | Executes MCP queries, updates hypothesis confidence (Bayesian) |
| **Analyzing** | Chain-of-Thought analysis, pattern recognition, relates to hypotheses |
| **Reflecting** | Self-critique: bias check, gap analysis, alternative explanations |
| **Correlating** | Finds cross-source connections (temporal, geospatial, causal) |
| **Verifying** | Validates conclusions, checks consistency, adjusts confidence |
| **Synthesizing** | Generates final report from **verified** insights and correlations |

### Phase Transitions

| From | To | Condition |
|------|----|-----------|
| Planning | Decomposing | Plan created |
| Decomposing | Hypothesizing | Task is moderate/complex |
| Hypothesizing | Gathering | Hypotheses generated |
| Gathering | Analyzing | No more pending queries |
| Gathering | Gathering | More queries to execute |
| Analyzing | Reflecting | Analysis complete |
| Analyzing | Gathering | Follow-up queries needed |
| Reflecting | Correlating | No critical issues |
| Reflecting | Gathering | Gaps need more investigation |
| Correlating | Verifying | Correlations found |
| Verifying | Synthesizing | Verification passed |
| Verifying | Reflecting | Issues found, needs review |
| Synthesizing | Complete | Report generated |

### Benefits

- **Chain-of-Thought**: Explicit step-by-step reasoning for transparency
- **Hypothesis Testing**: Evidence-based approach to intelligence analysis
- **Confidence Calibration**: Adjusts confidence based on reflection
- **Bias Detection**: Self-critique to identify potential blind spots
- **Consistency Checking**: Verifies conclusions don't contradict each other

### Reasoning Trace

All reasoning steps are logged to `reasoning.log` including:
- Thought process at each step
- Hypothesis status updates with confidence scores
- Self-reflection notes and identified issues
- Verification results for insights and correlations

## 🗺️ Roadmap

### ✅ Completed
- [x] MCP Server with OSINT tools
- [x] Rich logging system
- [x] Project restructuring (monorepo)
- [x] Agent skeleton
- [x] LLM integration (Gemini/Grok/Ollama/Docker)
- [x] Multi-step reasoning

### 🔴 Priority: New Data Sources
- [x] **Telegram Channels** - Real-time OSINT from conflict zones (Telethon API)
- [ ] **ACLED** - Armed Conflict Location & Event Data for structured conflict data
- [ ] **AlienVault OTX** - Open Threat Exchange for cyber threat intelligence
- [ ] **OpenSanctions** - Complete implementation (replace current stub)
- [ ] **Meduza/The Insider RSS** - Independent Russian news sources

### 🟡 Future
- [ ] Two agents, one for reasoning and onde for strictly JSON output
- [ ] Event correlation engine
- [ ] Real-time alerting
- [ ] Web dashboard

## 📄 License

MIT License

## ⚠️ Disclaimer

This tool is for research and educational purposes. Verify information from multiple sources and comply with applicable laws and API terms of service.
