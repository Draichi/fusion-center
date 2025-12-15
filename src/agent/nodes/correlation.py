"""
Correlation Node - Finds correlations between findings from different sources.
"""

import json
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import CorrelationOutput
from src.shared.logger import get_logger

logger = get_logger()


class CorrelationNode(BaseNode):
    """Node that finds correlations between findings from different sources."""
    
    def get_phase_name(self) -> str:
        return "correlating"
    
    def get_prompt(self, state: AgentState) -> str:
        findings = state.get("findings", [])
        
        # Group findings by type
        by_type = {}
        for i, f in enumerate(findings):
            ftype = f.get("source_type", "unknown")
            if ftype not in by_type:
                by_type[ftype] = []
            by_type[ftype].append({"index": i, **f})
        
        return f"""
You are correlating intelligence from multiple sources to find connections.

## Findings by Source Type
{json.dumps(by_type, indent=2)}

## Task Context
{state["task"]}

Look for:
1. **Temporal correlations**: Events happening around the same time
2. **Geospatial correlations**: Events in the same location from different sources
3. **Causal correlations**: One event potentially causing another
4. **Pattern correlations**: Similar patterns across different data types

For each correlation found, explain the connection and its implications.
"""
    
    def get_output_schema(self) -> type:
        return CorrelationOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        correlations = output
        
        logger.success(f"Found {len(correlations.get('correlations', []))} correlations")
        
        # Log correlation details
        corr_summary = []
        for corr in correlations.get("correlations", []):
            corr_type = corr.get("correlation_type", "unknown")
            desc = corr.get("description", "No description")
            confidence = corr.get("confidence", "unknown")
            corr_summary.append(f"[bold]{corr_type}[/bold] ({confidence})")
            corr_summary.append(f"  {desc}")
        
        if correlations.get("synthesis_notes"):
            corr_summary.append(f"\n[bold]Synthesis:[/bold] {correlations['synthesis_notes']}")
        
        if corr_summary:
            logger.panel("\n".join(corr_summary), title="🔗 Correlations Found", style="magenta")
        
        # Add reasoning step for correlation
        reasoning_step = self._add_reasoning_step(
            state,
            phase="correlating",
            thought="Looking for connections across different data sources",
            action=f"Found {len(correlations.get('correlations', []))} correlations",
            observation=correlations.get("synthesis_notes", "No synthesis notes"),
            conclusion="Ready for verification before final synthesis",
            confidence=0.8,
        )
        
        return {
            "correlations": state.get("correlations", []) + correlations.get("correlations", []),
            "key_insights": state.get("key_insights", []) + [correlations.get("synthesis_notes", "")],
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        # Go to verification before synthesis
        return ResearchPhase.VERIFYING.value

