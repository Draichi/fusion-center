"""
Reflection Node - Performs critical self-reflection on analysis.
"""

import json
from datetime import datetime
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import ReflectionOutput
from src.agent.prompts import REFLECTION_PROMPT
from src.agent.tools import get_tool_definitions
from src.shared.logger import get_logger

logger = get_logger()


class ReflectionNode(BaseNode):
    """Node that performs critical self-reflection on the analysis."""
    
    def get_phase_name(self) -> str:
        return "reflecting"
    
    def get_node_type(self) -> str:
        return "thinking"
    
    def get_prompt(self, state: AgentState) -> str:
        return f"""
{REFLECTION_PROMPT}

## Original Task
{state["task"]}

## Current Hypotheses
{json.dumps(state.get("hypotheses", []), indent=2)}

## Key Insights So Far
{json.dumps(state.get("key_insights", []), indent=2)}

## Correlations Found
{json.dumps(state.get("correlations", []), indent=2)}

## Uncertainties Identified
{json.dumps(state.get("uncertainties", []), indent=2)}

## Available Tools for Additional Investigation
{json.dumps([t["name"] for t in get_tool_definitions()], indent=2)}

Critically reflect on this analysis. Be thorough but constructive.
"""
    
    def get_output_schema(self) -> type:
        return ReflectionOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        result = output
        
        # Build reflection notes
        reflection_notes = []
        for note in result.get("reflection_notes", []):
            reflection_notes.append({
                "category": note.get("category", "gap_analysis"),
                "content": note.get("content", ""),
                "severity": note.get("severity", "info"),
                "action_required": note.get("action_required", False),
                "suggested_action": note.get("suggested_action"),
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        needs_investigation = result.get("needs_more_investigation", False)
        investigation_queries = result.get("investigation_suggestions", [])
        
        # Add reasoning step
        reflection_chain = result.get("reflection_chain", [])
        reasoning_step = self._add_reasoning_step(
            state,
            phase="reflecting",
            thought="\n".join(reflection_chain) if reflection_chain else "Examining analysis for biases and gaps",
            action=f"Generated {len(reflection_notes)} reflection notes",
            observation=f"Critical issues: {sum(1 for n in reflection_notes if n['severity'] == 'critical')}",
            conclusion="Needs more investigation" if needs_investigation else "Analysis is robust",
            confidence=0.75,
        )
        
        # Log summary with severity-based formatting
        summary_lines = ["[bold]Reflection Results:[/bold]"]
        severity_counts = {"info": 0, "warning": 0, "critical": 0}
        
        for note in reflection_notes:
            severity_counts[note["severity"]] = severity_counts.get(note["severity"], 0) + 1
            emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(note["severity"], "📝")
            summary_lines.append(f"  {emoji} [{note['category']}] {note['content']}")
            if note["action_required"] and note["suggested_action"]:
                summary_lines.append(f"      → Action: {note['suggested_action']}")
        
        summary_lines.append(f"\n[bold]Summary:[/bold] {severity_counts['critical']} critical, {severity_counts['warning']} warnings, {severity_counts['info']} info")
        
        style = "red" if severity_counts["critical"] > 0 else "yellow" if severity_counts["warning"] > 0 else "green"
        logger.panel("\n".join(summary_lines), title="🪞 Self-Reflection", style=style)
        
        # Determine pending queries - preserve existing and add new ones
        existing_pending = state.get("pending_queries", [])
        pending = list(existing_pending)  # Copy to avoid mutation
        
        # Add investigation suggestions if available
        if needs_investigation and investigation_queries and state.get("reflection_iterations", 0) < 2:
            # Convert investigation_suggestions to pending_queries format
            for query in investigation_queries[:3]:
                if isinstance(query, dict) and "tool" in query:
                    pending.append(query)
        
        # Track analysis-reflection cycles
        analysis_reflection_cycles = state.get("analysis_reflection_cycles", 0)
        # Increment cycle counter if going back to analysis
        if any(n["severity"] == "critical" and n["action_required"] for n in reflection_notes):
            analysis_reflection_cycles += 1
        
        return {
            "reflection_notes": state.get("reflection_notes", []) + reflection_notes,
            "reflection_iterations": state.get("reflection_iterations", 0) + 1,
            "needs_more_reflection": needs_investigation,
            "pending_queries": pending,
            "analysis_reflection_cycles": analysis_reflection_cycles,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "chain_of_thought": state.get("chain_of_thought", []) + reflection_chain,
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        needs_investigation = output.get("needs_more_investigation", False)
        investigation_queries = output.get("investigation_suggestions", [])
        reflection_notes = output.get("reflection_notes", [])
        
        # Check if there are pending queries in state (from analysis or previous iterations)
        pending_queries = state.get("pending_queries", [])
        
        # Always prioritize GATHERING if there are queries to execute
        if pending_queries:
            return ResearchPhase.GATHERING.value
        
        # Check if reflection suggests new investigation queries
        if needs_investigation and investigation_queries and state.get("reflection_iterations", 0) < 2:
            return ResearchPhase.GATHERING.value
        
        # Avoid infinite loop: limit analysis-reflection cycles
        analysis_reflection_cycles = state.get("analysis_reflection_cycles", 0)
        if analysis_reflection_cycles >= 3:
            logger.warning("Too many analysis-reflection cycles, forcing correlation")
            return ResearchPhase.CORRELATING.value
        
        # If there are critical issues, go back to analysis (but increment cycle counter)
        if any(n["severity"] == "critical" and n["action_required"] for n in reflection_notes):
            return ResearchPhase.ANALYZING.value
        
        # Otherwise, proceed to correlation
        return ResearchPhase.CORRELATING.value

