"""
Dashboard server for Project Overwatch.

Serves a web dashboard with terminal DOS aesthetic showing:
- Latest news
- Thermal anomalies on a 3D globe
- Telegram posts
- Threat intelligence
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.dashboard.api import router

# Create FastAPI app
app = FastAPI(title="Project Overwatch Dashboard", version="0.1.0")

# Include API routes
app.include_router(router)

# Get dashboard directory
dashboard_dir = Path(__file__).parent / "static"


@app.get("/")
async def index():
    """Serve the main dashboard page."""
    index_file = dashboard_dir / "index.html"
    if not index_file.exists():
        return {"error": "Dashboard files not found. Please ensure static files are present."}
    return FileResponse(index_file)


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
    print(f"📍 Dashboard available at: http://{args.host}:{args.port}")
    print(f"📊 API docs at: http://{args.host}:{args.port}/docs")
    print(f"🔗 MCP Server: {app.state.mcp_url}")
    print(f"\n⚠️  Make sure the MCP server is running!")
    print(f"   Start with: python -m src.mcp_server.server --transport sse --port {settings.mcp_server_port}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

