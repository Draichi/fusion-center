"""
Synthesis Node - Synthesizes all findings into final intelligence report.
"""

import json
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import SynthesisOutput
from src.agent.prompts import ENHANCED_SYNTHESIZER_PROMPT
from src.shared.logger import get_logger

logger = get_logger()


class SynthesisNode(BaseNode):
    """Node that synthesizes all findings into a final intelligence report."""
    
    def get_phase_name(self) -> str:
        return "synthesizing"
    
    def get_prompt(self, state: AgentState) -> str:
        # Use verified data if available, otherwise fall back to original
        insights_to_use = state.get("verified_insights") or state.get("key_insights", [])
        correlations_to_use = state.get("verified_correlations") or state.get("correlations", [])
        
        # Build hypothesis summary
        hypotheses_summary = ""
        if state.get("hypotheses"):
            hyp_lines = []
            for h in state["hypotheses"]:
                status_emoji = {
                    "supported": "✅",
                    "refuted": "❌",
                    "investigating": "🔍",
                    "inconclusive": "❓",
                    "proposed": "📝",
                }.get(h.get("status", "proposed"), "📝")
                hyp_lines.append(f"- {status_emoji} **{h['id']}**: {h['statement']} (confidence: {h.get('confidence', 0):.0%})")
            hypotheses_summary = "\n## Hypothesis Results\n" + "\n".join(hyp_lines)
        
        # Build reflection summary
        reflection_summary = ""
        if state.get("reflection_notes"):
            critical_notes = [n for n in state["reflection_notes"] if n.get("severity") == "critical"]
            if critical_notes:
                reflection_summary = "\n## Critical Reflection Notes\n" + "\n".join([f"- {n['content']}" for n in critical_notes])
        
        # Prepare findings with source information for citation
        findings_for_citation = []
        for i, finding in enumerate(state.get("findings", [])):
            findings_for_citation.append({
                "finding_id": i + 1,
                "source": finding.get("source"),
                "source_type": finding.get("source_type"),
                "timestamp": finding.get("timestamp"),
                "content": finding.get("content"),
                "location": finding.get("location"),
            })
        
        # Prepare executed queries for reference
        executed_queries_summary = []
        for q in state.get("executed_queries", []):
            executed_queries_summary.append({
                "tool": q.get("tool"),
                "args": q.get("args"),
                "timestamp": q.get("timestamp"),
            })
        
        return f"""
{ENHANCED_SYNTHESIZER_PROMPT}

## Original Task
{state["task"]}

## Research Plan
{json.dumps(state.get("research_plan", {}), indent=2)}
{hypotheses_summary}

## Verified Key Insights
{json.dumps(insights_to_use, indent=2)}

## Verified Correlations
{json.dumps(correlations_to_use, indent=2)}

## Uncertainties
{json.dumps(state.get("uncertainties", []), indent=2)}
{reflection_summary}

## Reasoning Process Summary
- Reasoning depth: {state.get("reasoning_depth", 0)} steps
- Reflection iterations: {state.get("reflection_iterations", 0)}
- Verification results: {len(state.get("verification_results", []))} items verified

## Raw Findings (Sources for Citation)
These are the actual data points collected from tools. CITE THESE in your report.
{json.dumps(findings_for_citation, indent=2)}

## Queries Executed
{json.dumps(executed_queries_summary, indent=2)}

Create a comprehensive intelligence report as JSON. 
IMPORTANT: The detailed_report MUST include inline citations for every claim and a "## Sources" section at the end.
"""
    
    def get_output_schema(self) -> type:
        return SynthesisOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        report = output
        
        logger.success("Final report synthesized")
        
        # Validate and clean detailed_report
        detailed_report = report.get("detailed_report", "")
        if not detailed_report or detailed_report.strip() in ["{", "}", "{}", ""]:
            logger.warning("⚠️ Detailed report is empty or malformed, generating fallback")
            # Generate fallback from insights and correlations
            insights = state.get("verified_insights") or state.get("key_insights", [])
            correlations = state.get("verified_correlations") or state.get("correlations", [])
            
            fallback_parts = []
            if insights:
                fallback_parts.append("## Key Findings\n")
                for i, insight in enumerate(insights[:5], 1):
                    if isinstance(insight, dict):
                        insight_text = insight.get("description", str(insight))
                    else:
                        insight_text = str(insight)
                    fallback_parts.append(f"{i}. {insight_text}\n")
                fallback_parts.append("\n")
            
            if correlations:
                fallback_parts.append("## Correlations\n")
                for corr in correlations[:3]:
                    corr_type = corr.get("correlation_type", "unknown")
                    desc = corr.get("description", "No description")
                    fallback_parts.append(f"**{corr_type.title()}**: {desc}\n")
                fallback_parts.append("\n")
            
            if not fallback_parts:
                detailed_report = "*No detailed analysis available. Please refer to the executive summary and key insights above.*"
            else:
                detailed_report = "".join(fallback_parts)
                detailed_report += "\n*Note: This analysis was auto-generated from verified insights due to incomplete LLM response.*"
        else:
            # Clean up common malformed patterns
            detailed_report = detailed_report.strip()
            # Remove standalone JSON braces
            if detailed_report.startswith("{") and not detailed_report.startswith("{"):
                # Check if it's just a brace
                if detailed_report == "{" or detailed_report.startswith("{\n") and len(detailed_report) < 10:
                    logger.warning("⚠️ Detailed report appears to be just a JSON brace, using fallback")
                    detailed_report = "*No detailed analysis available. Please refer to the executive summary and key insights above.*"
        
        # Log the executive summary
        if report.get("executive_summary"):
            logger.panel(
                report["executive_summary"],
                title="📊 Executive Summary",
                style="green"
            )
        
        # Log recommendations
        if report.get("recommendations"):
            rec_lines = ["[bold]Recommendations:[/bold]"]
            for rec in report["recommendations"]:
                rec_lines.append(f"  • {rec}")
            logger.panel("\n".join(rec_lines), title="💡 Recommendations", style="cyan")
        
        # Log confidence assessment
        if report.get("confidence_assessment"):
            logger.thinking(f"Confidence: {report['confidence_assessment']}")
        
        return {
            "executive_summary": report.get("executive_summary"),
            "detailed_report": detailed_report,
            "recommendations": report.get("recommendations", []),
            "confidence_assessment": report.get("confidence_assessment"),
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        return ResearchPhase.COMPLETE.value

