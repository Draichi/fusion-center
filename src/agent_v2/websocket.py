"""
WebSocket Manager for Agent v2 real-time state broadcasting.

Implements a singleton pattern to manage WebSocket connections
and broadcast agent state updates to connected dashboard clients.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, TYPE_CHECKING

from fastapi import WebSocket

from src.shared.logger import get_logger

if TYPE_CHECKING:
    from src.agent_v2.state import ResearchContext

logger = get_logger()


class WebSocketManager:
    """
    Singleton manager for WebSocket connections and agent state broadcasting.

    Handles:
    - Multiple concurrent dashboard connections
    - Broadcasting phase changes, tool calls, and state updates
    - Graceful handling of disconnected clients
    """

    _instance: WebSocketManager | None = None
    _initialized: bool = False

    def __new__(cls) -> WebSocketManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.connections: set[WebSocket] = set()
        self.current_session_id: str | None = None
        self.is_research_active: bool = False
        self.current_phase: str = "idle"
        self._lock = asyncio.Lock()
        self._initialized = True

        logger.info("WebSocketManager initialized")

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.connections.add(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.connections)}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self.connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.connections)}")

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected clients."""
        if not self.connections:
            return

        dead_connections: set[WebSocket] = set()
        message_json = json.dumps(message, default=str)

        async with self._lock:
            for ws in self.connections:
                try:
                    await ws.send_text(message_json)
                except Exception as e:
                    logger.warning(f"Failed to send to WebSocket client: {e}")
                    dead_connections.add(ws)

            # Clean up dead connections
            self.connections -= dead_connections

    async def broadcast_session_start(
        self,
        session_id: str,
        task: str,
        max_iterations: int,
    ) -> None:
        """Broadcast that a new research session has started."""
        self.current_session_id = session_id
        self.is_research_active = True
        self.current_phase = "planning"

        await self.broadcast({
            "type": "session_start",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "is_live": True,
            "data": {
                "task": task,
                "max_iterations": max_iterations,
            }
        })
        logger.info(f"[WS] Broadcast session_start: {session_id}")

    async def broadcast_phase_change(
        self,
        phase: str,
        ctx: ResearchContext,
    ) -> None:
        """Broadcast a phase change with current context state."""
        self.current_phase = phase

        await self.broadcast({
            "type": "phase_change",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.current_session_id,
            "is_live": True,
            "phase": phase,
            "data": self._serialize_context(ctx),
        })
        logger.info(f"[WS] Broadcast phase_change: {phase}")

    async def broadcast_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        result_preview: str | None = None,
    ) -> None:
        """Broadcast a tool call event."""
        await self.broadcast({
            "type": "tool_call",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.current_session_id,
            "is_live": True,
            "tool_name": tool_name,
            "arguments": args,
            "result_preview": result_preview[:200] if result_preview else None,
        })
        logger.debug(f"[WS] Broadcast tool_call: {tool_name}")

    async def broadcast_session_complete(
        self,
        ctx: ResearchContext,
        sitrep: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Broadcast that the research session has completed."""
        self.is_research_active = False
        self.current_phase = "complete"

        await self.broadcast({
            "type": "session_complete",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.current_session_id,
            "is_live": False,
            "success": success,
            "data": self._serialize_context(ctx),
            "sitrep": sitrep,
        })
        logger.info(f"[WS] Broadcast session_complete: {self.current_session_id}")

    async def broadcast_error(self, error: str) -> None:
        """Broadcast an error event."""
        await self.broadcast({
            "type": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.current_session_id,
            "is_live": self.is_research_active,
            "error": error,
        })
        logger.error(f"[WS] Broadcast error: {error}")

    def _serialize_context(self, ctx: ResearchContext) -> dict[str, Any]:
        """Serialize ResearchContext for WebSocket transmission."""
        return {
            "task": ctx.task,
            "iteration": ctx.iteration,
            "max_iterations": ctx.max_iterations,
            "task_complexity": ctx.task_complexity,
            "research_plan": ctx.research_plan,
            "sub_tasks": [
                {
                    "id": st.id,
                    "description": st.description,
                    "status": st.status,
                    "focus_area": st.focus_area,
                }
                for st in ctx.sub_tasks
            ],
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.statement,
                    "status": h.status,
                    "confidence": h.confidence,
                    "supporting_evidence": h.supporting_evidence,
                    "contradicting_evidence": h.contradicting_evidence,
                }
                for h in ctx.hypotheses
            ],
            "findings_count": len(ctx.findings),
            "key_insights": ctx.key_insights,
            "correlations": ctx.correlations,
            "uncertainties": ctx.uncertainties,
            "reflection_notes": [
                {
                    "category": n.category,
                    "content": n.content,
                    "severity": n.severity,
                }
                for n in ctx.reflection_notes
            ],
            "verified_insights": ctx.verified_insights,
            "started_at": ctx.started_at.isoformat() if ctx.started_at else None,
        }

    def get_status(self) -> dict[str, Any]:
        """Get current agent status for API endpoint."""
        return {
            "is_running": self.is_research_active,
            "session_id": self.current_session_id,
            "current_phase": self.current_phase,
            "connected_clients": len(self.connections),
        }


# Global singleton instance
ws_manager = WebSocketManager()


def get_ws_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return ws_manager
