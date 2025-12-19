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
    
    def get_node_type(self) -> str:
        return "thinking_markdown"
    
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
        
        # Prepare findings with source information for citation (LIMIT TO LAST 50)
        all_findings = state.get("findings", [])
        findings_for_citation = []
        for i, finding in enumerate(all_findings[-50:]):  # Last 50 for more context
            content = finding.get("content", "")
            # Handle both dict and string content types
            if isinstance(content, dict):
                content_str = json.dumps(content)
            else:
                content_str = str(content)
            
            findings_for_citation.append({
                "finding_id": len(all_findings) - 50 + i + 1 if len(all_findings) > 50 else i + 1,
                "source": finding.get("source"),
                "source_type": finding.get("source_type"),
                "content": content_str[:500],  # Increased to 500 chars for more detail
            })
        
        
        
        # Prepare executed queries for reference (LIMIT TO LAST 20)
        all_queries = state.get("executed_queries", [])
        executed_queries_summary = []
        for q in all_queries[-20:]:  # Last 20 for better coverage
            executed_queries_summary.append({
                "tool": q.get("tool"),
                "args": {k: str(v)[:100] for k, v in q.get("args", {}).items()},  # Increased to 100 chars
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
        
        
        findings_count = len(all_findings)
        queries_count = len(all_queries)
        
        return f"""
{SITREP_SYNTHESIZER_PROMPT}

## ORIGINAL QUERY
{state["task"]}

## DATA COLLECTION SUMMARY
- Total Findings Collected: {findings_count} (showing last 50)
- Total Queries Executed: {queries_count} (showing last 20)
- Intelligence Sources: {', '.join(sources_used) if sources_used else 'Multiple OSINT'}
{hypotheses_summary}

## KEY INSIGHTS (Top 20)
{json.dumps(insights_to_use[:20], indent=2) if insights_to_use else "No insights"}

## CORRELATIONS (Top 10)
{json.dumps(correlations_to_use[:10], indent=2) if correlations_to_use else "No correlations"}

## SAMPLE FINDINGS (Last 50 of {findings_count})
{json.dumps(findings_for_citation, indent=2)}

## SAMPLE QUERIES (Last 20 of {queries_count})
{json.dumps(executed_queries_summary, indent=2)}

Create a comprehensive SITREP intelligence report in MARKDOWN format.
IMPORTANT:
- Use proper markdown formatting with headers, lists, tables, etc.
- Include ALL standard SITREP sections
- Cite sources for every claim using findings
- Use intelligence community language and terminology
- Include probability assessments where appropriate
- Use tables for source reliability matrix
- Make it comprehensive and professional
"""
    
    def get_output_schema(self) -> type:
        # Not used for thinking_markdown mode, but required by interface
        # BaseNode will handle markdown output directly
        return SITREPOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        """Process the markdown report output and prepare state update."""
        
        # Get markdown report from output
        markdown_report = output.get("markdown_report", "")
        
        if not markdown_report:
            logger.warning("No markdown report generated")
            markdown_report = "# Error\n\nFailed to generate report."
        
        logger.success(f"SITREP intelligence report synthesized ({len(markdown_report)} characters)")
        
        # Extract first paragraph as executive summary
        lines = markdown_report.split("\n")
        exec_summary = ""
        for line in lines:
            if line.strip() and not line.startswith("#"):
                exec_summary = line.strip()
                break
        
        # Store the markdown report directly
        return {
            "executive_summary": exec_summary or "SITREP report completed",
            "detailed_report": markdown_report,
            "markdown_report": markdown_report,  # Primary field for markdown content
            "recommendations": [],  # Can be extracted from markdown if needed
            "confidence_assessment": "Report generated via dual-LLM reasoning",
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        return ResearchPhase.COMPLETE.value
