"""
Dashboard server for Project Overwatch.

Serves two dashboards with terminal DOS aesthetic:
- / (Agent Brain): Real-time visualization of agent_v2 research process
- /correlation-engine: Original dashboard with news, thermal anomalies, Telegram, threat intel

Features:
- WebSocket endpoint for real-time agent state streaming
- REST API for agent control and session management
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard.api import router
from src.agent_v2.websocket import get_ws_manager
from src.agent_v2.session import get_session_manager

# Create FastAPI app
app = FastAPI(title="Project Overwatch Dashboard", version="0.2.0")

# Include API routes
app.include_router(router)

# Get dashboard directory
dashboard_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    """Serve the Agent Brain dashboard (new main page)."""
    index_file = dashboard_dir / "index.html"
    if not index_file.exists():
        return {"error": "Dashboard files not found. Please ensure static files are present."}
    return FileResponse(index_file)


@app.get("/correlation-engine")
async def correlation_engine():
    """Serve the Correlation Engine dashboard (original dashboard)."""
    ce_file = dashboard_dir / "correlation-engine.html"
    if not ce_file.exists():
        return {"error": "Correlation Engine dashboard not found."}
    return FileResponse(ce_file)


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent state streaming.

    Clients connect here to receive:
    - session_start: When a new research begins
    - phase_change: When agent moves to a new phase with current state
    - tool_call: Real-time tool call notifications
    - session_complete: When research finishes
    - error: Error notifications
    - historical_session: On connect, if no active research
    """
    ws_manager = get_ws_manager()
    session_manager = get_session_manager()

    await ws_manager.connect(websocket)

    try:
        # Send initial state on connect
        if ws_manager.is_research_active:
            # Active research - client will receive live updates
            await websocket.send_json({
                "type": "connection_status",
                "status": "connected",
                "is_live": True,
                "session_id": ws_manager.current_session_id,
                "current_phase": ws_manager.current_phase,
            })
        else:
            # No active research - send last session
            last_session = session_manager.get_latest_session()
            if last_session:
                await websocket.send_json({
                    "type": "historical_session",
                    "is_live": False,
                    "data": last_session,
                })
            else:
                await websocket.send_json({
                    "type": "connection_status",
                    "status": "connected",
                    "is_live": False,
                    "message": "No previous sessions found",
                })

        # Keep connection alive, listen for client messages
        while True:
            data = await websocket.receive_json()
            # Handle client commands if needed (e.g., request state refresh)
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif data.get("type") == "request_status":
                await websocket.send_json({
                    "type": "status",
                    **ws_manager.get_status(),
                })

    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception as e:
        await ws_manager.disconnect(websocket)


# Mount static files
static_dir = dashboard_dir
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def main():
    """Run the dashboard server."""
    import argparse
    import uvicorn
    
    from src.shared.config import settings
    
    parser = argparse.ArgumentParser(
        description="Project Overwatch Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start dashboard on default port (8000)
  python -m src.dashboard.server

  # Start dashboard on custom port
  python -m src.dashboard.server --port 9000

  # Start dashboard on custom host and port
  python -m src.dashboard.server --host 0.0.0.0 --port 9000
        """,
    )
    parser.add_argument(
        "--host",
        default=settings.dashboard_host,
        help=f"Host to bind to (default: {settings.dashboard_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.dashboard_port,
        help=f"Port to listen on (default: {settings.dashboard_port})",
    )
    parser.add_argument(
        "--mcp-url",
        type=str,
        default=None,
        help=f"MCP server URL (default: http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse)",
    )
    
    args = parser.parse_args()
    
    # Store MCP URL in app state for use by API endpoints
    app.state.mcp_url = args.mcp_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
    
    print(f"🌐 Starting Project Overwatch Dashboard...")
    print(f"📍 Agent Brain:         http://{args.host}:{args.port}/")
    print(f"📍 Correlation Engine:  http://{args.host}:{args.port}/correlation-engine")
    print(f"🔌 WebSocket:           ws://{args.host}:{args.port}/ws/agent")
    print(f"📊 API docs:            http://{args.host}:{args.port}/docs")
    print(f"🔗 MCP Server:          {app.state.mcp_url}")
    print(f"\n⚠️  Make sure the MCP server is running!")
    print(f"   Start with: python -m src.mcp_server.server --transport sse --port {settings.mcp_server_port}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

