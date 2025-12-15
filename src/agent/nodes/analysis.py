"""
Analysis Node - Analyzes collected findings and extracts insights.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import AnalysisOutput
from src.agent.prompts import ENHANCED_ANALYST_PROMPT
from src.agent.tools import get_tool_definitions
from src.shared.logger import get_logger

logger = get_logger()


class AnalysisNode(BaseNode):
    """Node that analyzes collected findings and extracts insights."""
    
    def get_phase_name(self) -> str:
        return "analyzing"
    
    def get_prompt(self, state: AgentState) -> str:
        findings = state.get("findings", [])
        available_tools = get_tool_definitions()
        tool_names = [t["name"] for t in available_tools]
        
        return f"""
{ENHANCED_ANALYST_PROMPT}

## Original Task
{state["task"]}

## Current Hypotheses
{json.dumps(state.get("hypotheses", []), indent=2) if state.get("hypotheses") else "No hypotheses generated yet."}

## Collected Findings ({len(findings)} total)
{json.dumps(findings[:30], indent=2)}  # Limit to 30 for context window

## Available Tools for Follow-up Queries
You can ONLY use these tool names for follow_up_queries:
{json.dumps(tool_names, indent=2)}

Tool details:
{json.dumps(available_tools, indent=2)}

## Instructions
Think step by step:
1. First, survey all the evidence - what types of data do we have?
2. What patterns emerge from individual sources?
3. How does this evidence relate to our hypotheses?
4. What areas still need investigation?

Then provide your analysis with:
1. Key patterns observed (with evidence citations)
2. Notable events or anomalies
3. Preliminary correlations between different data sources
4. How findings support/refute each hypothesis
5. Confidence assessment for each insight
6. Areas requiring further investigation

IMPORTANT: For follow_up_queries, you MUST use one of the exact tool names listed above.
Do NOT use data source names like "GDELT" or "IODA" - use the actual tool names like "search_news" or "check_connectivity".
"""
    
    def get_output_schema(self) -> type:
        return AnalysisOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        analysis = output
        findings = state.get("findings", [])
        
        if not findings:
            logger.warning("No findings to analyze")
            return {}
        
        # Check if follow-up queries are needed
        follow_up = analysis.get("follow_up_queries", [])
        
        logger.success(f"Analysis complete: {len(analysis.get('key_insights', []))} insights")
        
        # Log thinking/reasoning chain
        if thinking:
            self.writer.log_reasoning(
                phase="analyzing",
                step=state["iteration"] + 1,
                action="Chain of thought reasoning",
                details={"thinking": thinking},
            )
        
        # Add reasoning step
        reasoning_step = self._add_reasoning_step(
            state,
            phase="analyzing",
            thought=thinking if thinking else "Analyzing patterns in collected evidence",
            action=f"Identified {len(analysis.get('key_insights', []))} insights",
            observation=f"Found {len(analysis.get('correlations', []))} correlations, {len(analysis.get('uncertainties', []))} uncertainties",
            conclusion="Analysis complete, proceeding to reflection" if not follow_up else f"Need {len(follow_up)} more queries",
            confidence=0.75,
        )
        
        # Log analysis details
        analysis_summary = []
        if analysis.get("key_insights"):
            analysis_summary.append("[bold]Key Insights:[/bold]")
            for insight in analysis["key_insights"]:
                if isinstance(insight, dict):
                    analysis_summary.append(f"  • {insight.get('description', str(insight))}")
                else:
                    analysis_summary.append(f"  • {insight}")
        
        if analysis.get("hypothesis_implications"):
            analysis_summary.append(f"[bold]Hypothesis Implications:[/bold]")
            for hi in analysis["hypothesis_implications"]:
                emoji = {"supporting": "✅", "contradicting": "❌", "neutral": "➖"}.get(hi.get("evidence_type", "neutral"), "➖")
                analysis_summary.append(f"  {emoji} [{hi.get('hypothesis_id')}] {hi.get('explanation', '')}")
        
        if analysis.get("correlations"):
            analysis_summary.append(f"[bold]Correlations Found:[/bold] {len(analysis['correlations'])}")
        if analysis.get("uncertainties"):
            analysis_summary.append(f"[bold]Uncertainties:[/bold] {len(analysis['uncertainties'])}")
        if follow_up:
            analysis_summary.append(f"[bold]Follow-up Queries:[/bold] {len(follow_up)}")
            for q in follow_up:
                analysis_summary.append(f"  • {q.get('tool')}: {q.get('reason', 'No reason provided')}")
        
        if analysis_summary:
            logger.panel("\n".join(analysis_summary), title="🔍 Analysis Results", style="yellow")
        
        # Preserve existing pending queries and add new follow-ups
        existing_pending = state.get("pending_queries", [])
        new_pending = existing_pending + follow_up[:3]  # Limit new follow-ups
        
        return {
            "key_insights": state.get("key_insights", []) + analysis.get("key_insights", []),
            "correlations": state.get("correlations", []) + analysis.get("correlations", []),
            "uncertainties": state.get("uncertainties", []) + analysis.get("uncertainties", []),
            "pending_queries": new_pending,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "reasoning_depth": state.get("reasoning_depth", 0) + 1,
            "chain_of_thought": state.get("chain_of_thought", []) + ([thinking] if thinking else []),
            "messages": state.get("messages", []) + [
                AIMessage(content=f"Analysis: {json.dumps(analysis, indent=2)}"),
            ],
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        follow_up = output.get("follow_up_queries", [])
        # Check if there are pending queries in state (from previous iterations)
        pending_queries = state.get("pending_queries", [])
        
        # Always go to GATHERING if there are queries to execute
        if follow_up or pending_queries:
            return ResearchPhase.GATHERING.value
        
        # Use REFLECTING phase for multi-step reasoning instead of going directly to CORRELATING
        return ResearchPhase.REFLECTING.value

