# Project Overwatch 🌐

**Fusion Center - MCP Server and AI Agent for OSINT and Geopolitical Intelligence**

Project Overwatch is an autonomous intelligence system that combines a Model Context Protocol (MCP) server with an AI agent for Open Source Intelligence (OSINT) analysis. It correlates data from news media, satellite imagery, and internet infrastructure monitoring.

## 🌟 Introduction & Proven Capabilities

Fusion Center isn't just a data fetcher; it's a **Correlation Engine**. Its true power lies in "triangulating" the truth by cross-referencing signals from three distinct domains:
1.  **Physical** (Satellite Thermal Anomalies)
2.  **Digital** (Internet Traffic & Outages)
3.  **Informational** (News, Telegram, Threat Intel)

<div align="center">
  <h3>🎥 Watch the Agent in Action</h3>
  <!-- VIDEO PLACEHOLDER: Fusion Center Agent Generating Intelligence Report -->
  <img src="https://placehold.co/800x450?text=Fusion+Center+Agent+Demo+Video+(Coming+Soon)" alt="Fusion Center Demo" width="100%" />
  <p><i>Demonstration: The Agent performing a multi-step analysis on live data.</i></p>
</div>

### 💡 Exemplary Queries (Showcase)

To see the agent at its full potential, try these queries that force multi-domain correlation:

#### 1. The "Words vs. Actions" Divergence Test (Diplomacy vs. Reality)
> "Analyze the trend of 'thermal anomaly' intensity in the Donbas region over the last 14 days using NASA FIRMS data and correlate it with the sentiment of GDELT news articles containing keywords 'negotiation', 'peace talks', or 'diplomats'. **Hypothesis:** If news sentiment is positive (talks) but thermal anomalies are rising (kinetic), the probability of a ceasefire is LOW."
*   **Why it works:** Validates if diplomatic rhetoric is supported by ground-truth physics or if it's a distraction.

#### 2. The "Silent De-escalation" (Cyber & Infrastructure)
> "Check IODA and Cloudflare Radar for a reduction in critical infrastructure outages or DDoS attack traffic in Ukraine over the past week. Correlate this with any 'goodwill' statements found in RSS feeds. **Premise:** Ceasefires often begin with a reduction in strategic infrastructure targeting."

#### 3. The "Propaganda Verification" (Information War)
> "Compare the official narrative found in GDELT state media reports regarding border tensions with the raw sentiment and photos describing troop movements in local Telegram OSINT channels. Also, check AlienVault OTX for any spike in cyber threat indicators targeting that region."

<details>
<summary>📄 <b>Click to view Sample Intelligence Report Output</b></summary>

```markdown
🔴 **COMPREHENSIVE OSINT SITREP – PROJECT OVERWATCH**

**CLASSIFICATION:** OSINT / PUBLIC
**DTG:** 18 DEC 2025 / 19:30Z
**QUERY:** "Compare official GDELT state media narrative vs. Telegram OSINT on Ukraine border tensions"
**INTELLIGENCE SOURCES:** GDELT, Telegram OSINT, AlienVault OTX, NASA FIRMS

–––

## SECTION I – EXECUTIVE INTELLIGENCE SUMMARY

### A. DIRECT RESPONSE TO QUERY

**Analysis reveals a significant narrative divergence:** State media (GDELT) reports "de-escalation" and "diplomatic progress" (65% positive sentiment), while local Telegram OSINT channels document **increased troop movements** and **artillery repositioning** near Kharkiv and Sumy oblasts.

### B. KEY INTELLIGENCE HIGHLIGHTS

• **Narrative Divergence Detected:** GDELT state media sources published 47 articles (Dec 14-18) emphasizing "ceasefire talks" and "withdrawal agreements". Simultaneously, Telegram channels (@CITeam_ua, @DeepStateUA, @Rybar) reported 23 confirmed sightings of armored columns moving **toward** the border, not away.

• **Kinetic Reality Check:** NASA FIRMS detected 156 thermal anomalies in Kharkiv Oblast (Dec 16-17), a **340% increase** from the 7-day baseline. Coordinates cluster around (49.9°N, 36.2°E), consistent with artillery positions reported by @CITeam_ua 4 hours prior.

• **Cyber Indicators Align:** AlienVault OTX shows a **spike in DDoS pulses** targeting Ukrainian energy infrastructure IPs (+180% vs. baseline). Attack vectors match Russian APT28 TTPs, contradicting the "peaceful" narrative.

• **Temporal Correlation:** State media "peace talks" articles peaked on Dec 16 at 14:00 UTC, exactly **6 hours before** Telegram channels reported renewed shelling near Vovchansk (50.29°N, 36.93°E).

### C. CONFIDENCE ASSESSMENT

**Overall Confidence:** 82%
**Intelligence Quality:** HIGH  
**Query Complexity:** MODERATE

**Assessment:** Strong multi-source corroboration between kinetic signals (FIRMS), cyber activity (OTX), and ground truth (Telegram). State media narrative assessed as **disinformation or strategic misdirection**.

–––

## SECTION IV – ACTIONABLE INTELLIGENCE & RECOMMENDATIONS

### A. IMMEDIATE ACTIONS

1. **Monitor Artillery Deployment:** Track thermal anomalies in cluster zone (49.8-50.1°N, 36.0-36.5°E)
2. **Cross-Reference Telegram Geotagged Posts:** Validate troop movement claims with FIRMS data
3. **Cyber Defense Alert:** Warn energy sector of probable escalation in DDoS attacks

### B. MONITORING INDICATORS

• Thermal anomaly density >100/day in Kharkiv Oblast = High kinetic alert
• State media "peace" keyword volume inversely correlated with FIRMS detections
• OTX pulse velocity >50 new IoCs/day targeting UA infrastructure

### C. FOLLOW-UP COLLECTION

• **Satellite Tasking:** Request commercial SAR imagery of Vovchansk area for vehicle count validation
• **SIGINT Cross-Check:** Correlate with radio chatter reports from Telegram (if available)
• **GDELT Deep Dive:** Analyze which state outlets push "peace" narrative hardest (identify amplifiers)

–––

## SECTION V – INTELLIGENCE ASSESSMENT METADATA

### A. SOURCE RELIABILITY MATRIX

| Source | Reliability | Credibility | Timeliness | Grade | Notes |
|--------|-------------|-------------|------------|-------|-------|
| GDELT News | C | 4 | Current | C-4 | State bias detected |
| Telegram OSINT | B | 2 | Real-time | B-2 | Verified channels only |
| NASA FIRMS | A | 1 | 3-hour lag | A-1 | Physics-based, no bias |
| AlienVault OTX | B | 2 | Current | B-2 | Community-sourced |

**Grading Key:** Reliability (A-F), Credibility (1-6, lower=better)

### B. ANALYTICAL CONFIDENCE

- **Methodology:** Dual-LLM reasoning with Bayesian hypothesis updating
- **Primary Evidence:** 156 thermal anomalies + 23 Telegram sightings + 47 DDoS pulses
- **Assumptions:** Telegram channels are not compromised; FIRMS anomalies exclude wildfires (verified via intensity >350K)

–––

## SECTION VI – INTELLIGENCE NARRATIVE ANALYSIS

**Assessed Purpose of State Media Narrative:**
The timing and intensity of "de-escalation" messaging suggests a **strategic deception operation** to:
1. Lower international alert posture before kinetic action
2. Create plausible deniability for troop repositioning ("exercises")
3. Exploit Western holiday period (Dec 20-25) for reduced monitoring

**Risk:** If pattern holds, expect significant military action within **72-96 hours** of peak "peace" messaging.

–––

**CLASSIFICATION:** OSINT / PUBLIC  
**ANALYST:** Project Overwatch Dual-LLM Intelligence System  
**SESSION:** 20251218_194200_narrative_divergence_analysis

〔END SITREP〕
```
</details>

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
| 📰 **News** | `fetch_rss_news` | Fetch articles from RSS feeds (Meduza, The Insider, The Cradle) |
| 🔍 **Search** | `search_internet` | General web search via DuckDuckGo |
| 🔍 **Search** | `search_leaks` | Search leaked datasets (DDoS Secrets) |
| 🛰️ **Satellite** | `detect_thermal_anomalies` | NASA FIRMS fire/explosion detection |
| 🌐 **Cyber** | `check_connectivity` | IODA internet outage detection |
| 🌐 **Cyber** | `check_traffic_metrics` | Cloudflare Radar analysis |
| 📱 **Telegram** | `search_telegram` | Search OSINT Telegram channels |
| 📱 **Telegram** | `get_channel_info` | Get Telegram channel metadata |
| 📱 **Telegram** | `list_osint_channels` | List curated OSINT channels |
|  🔍 **Threat Intel** | `check_ioc` | Look up IoC in AlienVault OTX |
| 🔍 **Threat Intel** | `get_threat_pulse` | Get OTX pulse details |
| 🔍 **Threat Intel** | `search_threats` | Search OTX threat pulses |

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

# Required for threat intelligence (get from https://otx.alienvault.com)
OTX_API_KEY=your_otx_key

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

### Run the Dashboard

```bash
# Start dashboard (requires MCP server to be running)
python -m src.dashboard.server

# Custom port
python -m src.dashboard.server --port 9000

# Custom MCP server URL
python -m src.dashboard.server --mcp-url http://localhost:9000/sse
```

The dashboard provides a web interface at `http://127.0.0.1:8000` showing:
- Latest news from GDELT
- Thermal anomalies on an interactive 3D globe
- Telegram OSINT channel posts
- Threat intelligence pulses

### Run All Components Together

```bash
# Terminal 1: Start MCP Server
python -m src.mcp_server.server --transport sse --port 8080

# Terminal 2: Run Agent (optional)
python -m src.agent "Correlate thermal anomalies with news near Kyiv"

# Terminal 3: Start Dashboard (optional)
python -m src.dashboard.server --port 8000
```

You can run the MCP server alone and connect multiple clients (agent, dashboard, or custom clients) to it.

## 📁 Project Structure

<details>
<summary>📂 <b>Click to view Project Structure</b></summary>

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
    │       └── threat_intel.py # AlienVault OTX
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
    ├── dashboard/              # 🌐 Web Dashboard
    │   ├── __init__.py
    │   ├── server.py           # Dashboard server (FastAPI)
    │   ├── api.py              # API endpoints (MCP client)
    │   └── static/             # Frontend files
    │       ├── index.html      # Dashboard page
    │       ├── style.css       # Terminal DOS styling
    │       └── app.js          # Frontend logic
    │
    └── shared/                 # 🔗 Shared Code
        ├── __init__.py
        ├── config.py           # Centralized config
        ├── logger.py           # Rich logging
        └── output_writer.py    # Report & reasoning log writer
```
</details>

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
| [DuckDuckGo](https://duckduckgo.com/) | Privacy-focused web search | Free (no API key) |
| [DDoS Secrets](https://ddosecrets.com/) | Leaked/hacked data archive  | Free (web scraping) |
| [GDELT](https://www.gdeltproject.org/) | Global news monitoring | Free |
| [Meduza](https://meduza.io/) | Independent Russian news | Free (RSS) |
| [The Insider](https://theinsider.me/) | Russian investigative journalism | Free (RSS) |
| [The Cradle](https://thecradle.co/) | Geopolitical news (West Asia) | Free (RSS) |
| [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/) | Satellite fire detection | Free API key |
| [IODA](https://ioda.inetintel.cc.gatech.edu/) | Internet outages | Free |
| [Cloudflare Radar](https://radar.cloudflare.com/) | Traffic analytics | Free (limited) |
| [Telegram](https://my.telegram.org/) | OSINT channel monitoring | Free API credentials |
| [AlienVault OTX](https://otx.alienvault.com/) | Threat intelligence | Free API key |

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

<details>
<summary>🧠 <b>Click to view Multi-step Reasoning Flow</b></summary>

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
</details>

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
- [x] **AlienVault OTX** - Open Threat Exchange for cyber threat intelligence
- [x] **RSS Feeds** - Independent news sources (Meduza, The Insider, The Cradle)

### 🟡 Future
- [x] Two agents, one for reasoning and one for strictly JSON output
- [x] Event correlation engine
- [x] Web dashboard
- [x] Add DuckDuckGo search tool
- [x] Add https://ddosecrets.com/ as a tool (web scraping)
- [ ] Create a PoC agent using PydanticAI instead of the current langchain agent

## 📄 License

MIT License

## ⚠️ Disclaimer

This tool is for research and educational purposes. Verify information from multiple sources and comply with applicable laws and API terms of service.
