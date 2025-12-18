"""
Synthesis Node - Synthesizes all findings into final SITREP intelligence report.
"""

import json
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import SITREPOutput
from src.agent.prompts import SITREP_SYNTHESIZER_PROMPT
from src.shared.logger import get_logger

logger = get_logger()


class SynthesisNode(BaseNode):
    """Node that synthesizes all findings into a final SITREP intelligence report."""
    
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
                "status": q.get("status"),
            })
        
        # Categorize sources used
        sources_used = set()
        for q in state.get("executed_queries", []):
            tool = q.get("tool", "")
            if "news" in tool.lower() or "rss" in tool.lower() or "gdelt" in tool.lower():
                sources_used.add("GDELT News Database")
            elif "thermal" in tool.lower() or "firms" in tool.lower() or "satellite" in tool.lower():
                sources_used.add("NASA FIRMS Satellite Data")
            elif "connectivity" in tool.lower() or "ioda" in tool.lower() or "outage" in tool.lower():
                sources_used.add("IODA Internet Monitoring")
            elif "traffic" in tool.lower() or "cloudflare" in tool.lower():
                sources_used.add("Cloudflare Radar")
            elif "telegram" in tool.lower():
                sources_used.add("Telegram OSINT Channels")
            elif "ioc" in tool.lower() or "threat" in tool.lower() or "otx" in tool.lower() or "pulse" in tool.lower():
                sources_used.add("AlienVault OTX Threat Intelligence")
        
        return f"""
{SITREP_SYNTHESIZER_PROMPT}

## ORIGINAL QUERY
{state["task"]}

## INTELLIGENCE SOURCES AVAILABLE
{', '.join(sources_used) if sources_used else 'Multiple OSINT sources'}

## RESEARCH PLAN
{json.dumps(state.get("research_plan", {}), indent=2)}
{hypotheses_summary}

## VERIFIED KEY INSIGHTS
{json.dumps(insights_to_use, indent=2)}

## VERIFIED CORRELATIONS
{json.dumps(correlations_to_use, indent=2)}

## UNCERTAINTIES & INTELLIGENCE GAPS
{json.dumps(state.get("uncertainties", []), indent=2)}
{reflection_summary}

## REASONING PROCESS SUMMARY
- Reasoning depth: {state.get("reasoning_depth", 0)} steps
- Reflection iterations: {state.get("reflection_iterations", 0)}
- Verification results: {len(state.get("verification_results", []))} items verified

## RAW FINDINGS (Sources for Citation)
These are the actual data points collected from tools. CITE THESE in your report.
{json.dumps(findings_for_citation, indent=2)}

## QUERIES EXECUTED
{json.dumps(executed_queries_summary, indent=2)}

Create a comprehensive SITREP intelligence report as JSON following the schema exactly.
IMPORTANT: 
- Include ALL 6 sections (section_i through section_vi)
- Cite sources for every claim
- Use intelligence community language
- Include probability assessments where appropriate
- Complete the source reliability matrix in section_v
"""
    
    def get_output_schema(self) -> type:
        return SITREPOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        """Process the SITREP output and prepare state update."""
        report = output
        
        logger.success("SITREP intelligence report synthesized")
        
        # Extract key fields for state update
        section_i = report.get("section_i", {})
        section_ii = report.get("section_ii", {})
        section_iii = report.get("section_iii", {})
        section_iv = report.get("section_iv", {})
        section_v = report.get("section_v", {})
        section_vi = report.get("section_vi", {})
        
        # Build executive summary from Section I
        direct_response = section_i.get("direct_response", "")
        confidence = section_i.get("overall_confidence_percent", 75)
        
        # Build recommendations from Section IV
        recommendations = (
            section_iv.get("immediate_actions", []) +
            section_iv.get("monitoring_indicators", [])[:2]
        )
        
        # Store the full SITREP report for output writer
        return {
            # Legacy fields for compatibility
            "executive_summary": direct_response,
            "detailed_report": "",  # Will be generated by output writer
            "recommendations": recommendations,
            "confidence_assessment": f"{confidence}% - {section_i.get('intelligence_quality', 'GOOD')} quality",
            
            # New SITREP fields
            "sitrep_output": report,
            "classification": report.get("classification", "OSINT / PUBLIC"),
            "query_summary": report.get("query_summary", state.get("task", "")),
            "intelligence_sources_used": report.get("intelligence_sources_used", []),
            "section_i": section_i,
            "section_ii": section_ii,
            "section_iii": section_iii,
            "section_iv": section_iv,
            "section_v": section_v,
            "section_vi": section_vi,
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        return ResearchPhase.COMPLETE.value
