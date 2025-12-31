"""
Session Manager for Agent v2 persistence using SQLite.

Stores research sessions for historical viewing when no agent is running.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

from src.shared.config import settings
from src.shared.logger import get_logger

if TYPE_CHECKING:
    from src.agent_v2.state import ResearchContext

logger = get_logger()


class SessionManager:
    """
    Manages persistence of agent research sessions in SQLite.

    Sessions are stored to provide:
    - Historical session viewing when no agent is running
    - Session continuation/reference
    - Audit trail of research activities
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        Initialize the session manager.

        Args:
            db_path: Path to SQLite database. Defaults to config setting.
        """
        self.db_path = db_path or settings.agent_sessions_db

        # Ensure data directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"SessionManager initialized with DB: {self.db_path}")

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    current_phase TEXT DEFAULT 'planning',
                    iteration INTEGER DEFAULT 0,
                    max_iterations INTEGER DEFAULT 5,
                    state_json TEXT,
                    sitrep_json TEXT,
                    success BOOLEAN DEFAULT NULL
                )
            """)

            # Create index for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_started_at
                ON sessions(started_at DESC)
            """)

            conn.commit()

    def create_session(
        self,
        session_id: str,
        task: str,
        max_iterations: int = 5,
    ) -> None:
        """Create a new session record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, task, max_iterations, started_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, task, max_iterations, datetime.utcnow()),
            )
            conn.commit()
        logger.info(f"Session created: {session_id}")

    def update_session_state(
        self,
        session_id: str,
        ctx: ResearchContext,
        current_phase: str,
    ) -> None:
        """Update session with current state (incremental update during research)."""
        state_json = self._serialize_context(ctx)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET state_json = ?,
                    current_phase = ?,
                    iteration = ?
                WHERE id = ?
                """,
                (state_json, current_phase, ctx.iteration, session_id),
            )
            conn.commit()

    def complete_session(
        self,
        session_id: str,
        ctx: ResearchContext,
        sitrep: dict[str, Any] | None = None,
        success: bool = True,
    ) -> None:
        """Mark session as complete with final state and SITREP."""
        state_json = self._serialize_context(ctx)
        sitrep_json = json.dumps(sitrep, default=str) if sitrep else None

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET state_json = ?,
                    sitrep_json = ?,
                    current_phase = 'complete',
                    iteration = ?,
                    completed_at = ?,
                    success = ?
                WHERE id = ?
                """,
                (
                    state_json,
                    sitrep_json,
                    ctx.iteration,
                    datetime.utcnow(),
                    success,
                    session_id,
                ),
            )
            conn.commit()
        logger.info(f"Session completed: {session_id} (success={success})")

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get a specific session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def get_latest_session(self) -> dict[str, Any] | None:
        """Get the most recent session."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM sessions
                ORDER BY started_at DESC
                LIMIT 1
                """
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_dict(row)

    def list_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """List recent sessions (metadata only, not full state)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, task, started_at, completed_at, current_phase,
                       iteration, max_iterations, success
                FROM sessions
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def cleanup_old_sessions(self, days: int | None = None) -> int:
        """
        Delete sessions older than specified days.

        Args:
            days: Number of days to retain. Defaults to config setting.

        Returns:
            Number of sessions deleted.
        """
        retention_days = days or settings.session_retention_days
        cutoff = datetime.utcnow() - timedelta(days=retention_days)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE started_at < ?",
                (cutoff,),
            )
            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} sessions older than {retention_days} days")

        return deleted

    def _serialize_context(self, ctx: ResearchContext) -> str:
        """Serialize ResearchContext to JSON string."""
        data = {
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
                    "reasoning": h.reasoning,
                }
                for h in ctx.hypotheses
            ],
            "findings": [
                {
                    "source": f.source,
                    "source_type": f.source_type,
                    "timestamp": f.timestamp,
                    "content": f.content,
                    "relevance_score": f.relevance_score,
                    "confidence": f.confidence,
                    "location": f.location,
                }
                for f in ctx.findings
            ],
            "executed_queries": ctx.executed_queries,
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
            "verified_correlations": ctx.verified_correlations,
            "started_at": ctx.started_at.isoformat() if ctx.started_at else None,
        }
        return json.dumps(data, default=str)

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary with parsed JSON."""
        result = dict(row)

        # Parse JSON fields
        if result.get("state_json"):
            try:
                result["state"] = json.loads(result["state_json"])
            except json.JSONDecodeError:
                result["state"] = None
            del result["state_json"]

        if result.get("sitrep_json"):
            try:
                result["sitrep"] = json.loads(result["sitrep_json"])
            except json.JSONDecodeError:
                result["sitrep"] = None
            del result["sitrep_json"]

        return result


# Module-level instance for convenience
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
