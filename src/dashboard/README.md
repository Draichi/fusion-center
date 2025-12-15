# Project Overwatch Dashboard

Terminal DOS-style web dashboard for OSINT intelligence visualization.

## Features

- **Latest News**: GDELT news articles displayed in terminal style
- **Thermal Anomalies**: Interactive 3D globe showing NASA FIRMS thermal anomaly detections
- **Telegram Feed**: Recent posts from OSINT Telegram channels
- **Threat Intelligence**: AlienVault OTX threat pulses

## Prerequisites

**The MCP server must be running before starting the dashboard!**

The dashboard connects to the MCP server via HTTP/SSE to fetch data. You can run them separately:

1. Start the MCP server in one terminal
2. Start the dashboard in another terminal

## Running the Dashboard

### Using the script entry point:

```bash
# Default configuration (connects to MCP server at http://127.0.0.1:8080/sse)
python -m src.dashboard.server

# Or using the entry point
overwatch-dashboard

# Custom port
python -m src.dashboard.server --port 9000

# Custom host and port
python -m src.dashboard.server --host 0.0.0.0 --port 9000

# Custom MCP server URL
python -m src.dashboard.server --mcp-url http://localhost:9000/sse
```

### Configuration

The dashboard runs on `http://127.0.0.1:8000` by default and connects to the MCP server at `http://127.0.0.1:8080/sse`.

You can configure:
- Dashboard host/port via `--host` and `--port` arguments
- MCP server URL via `--mcp-url` argument
- Or via environment variables in `.env`:
  - `DASHBOARD_HOST` (default: `127.0.0.1`)
  - `DASHBOARD_PORT` (default: `8000`)
  - `MCP_SERVER_HOST` (default: `127.0.0.1`)
  - `MCP_SERVER_PORT` (default: `8080`)

## API Endpoints

- `GET /api/dashboard` - Get all dashboard data in one request
- `GET /api/news` - Get latest news articles
- `GET /api/thermal-anomalies` - Get thermal anomalies (with optional lat/lng params)
- `GET /api/telegram` - Get Telegram posts
- `GET /api/threat-intel` - Get threat intelligence pulses

## Architecture

```
┌─────────────────┐         HTTP/SSE         ┌──────────────────┐
│   Dashboard     │ ◄──────────────────────► │   MCP Server     │
│   (Port 8000)   │                           │   (Port 8080)    │
└─────────────────┘                           └──────────────────┘
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Data Sources    │
                                              │  (GDELT, NASA,   │
                                              │   Telegram, OTX)  │
                                              └──────────────────┘
```

The dashboard is a **client** of the MCP server. It does not directly access data sources - it calls MCP tools via HTTP/SSE.

## Requirements

The MCP server requires the same API keys (configured in `.env`):
- NASA FIRMS API key (for thermal anomalies)
- Telegram API credentials (for Telegram feed)
- AlienVault OTX API key (for threat intel)

If any service is not configured, the dashboard will show an error message for that section but continue to work for other sections.

## Design

The dashboard features a retro terminal DOS aesthetic with:
- Dark background (#0a0a0a)
- Neon green text (#00ff41)
- Scanline effects
- Monospace font (JetBrains Mono)
- Glowing borders and shadows

