"""
Output Writer Module for Project Overwatch.

Handles writing research outputs:
- Final report as Markdown (.md)
- Reasoning steps as a separate log file
"""

import os
import json
import re
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


class OutputWriter:
    """Handles writing agent outputs to files."""

    def __init__(self, output_dir: str | None = None, session_id: str | None = None, query: str | None = None):
        """
        Initialize the output writer.

        Args:
            output_dir: Directory for output files (default from settings)
            session_id: Unique session identifier (auto-generated if not provided)
            query: The research query (used to name the output folder)
        """
        self.output_dir = Path(output_dir or settings.output_dir)
        
        # Build session_id with timestamp + sanitized query
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if query:
            # Sanitize query for folder name: lowercase, replace spaces with _, remove special chars
            sanitized = re.sub(r'[^\w\s-]', '', query.lower())
            sanitized = re.sub(r'[\s-]+', '_', sanitized).strip('_')
            # Limit to 50 chars to avoid overly long folder names
            sanitized = sanitized[:50].rstrip('_')
            self.session_id = session_id or f"{timestamp}_{sanitized}"
        else:
            self.session_id = session_id or timestamp
        
        # Create session directory
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.report_path = self.session_dir / "report.md"
        self.reasoning_log_path = self.session_dir / "reasoning.log"
        self.state_path = self.session_dir / "state.json"
        
        # Initialize reasoning log
        self._init_reasoning_log()

    def _init_reasoning_log(self) -> None:
        """Initialize the reasoning log file with header."""
        header = f"""================================================================================
PROJECT OVERWATCH - REASONING LOG
Session: {self.session_id}
Started: {datetime.now().isoformat()}
================================================================================

"""
        with open(self.reasoning_log_path, "w", encoding="utf-8") as f:
            f.write(header)

    def log_reasoning(
        self,
        phase: str,
        step: int,
        action: str,
        details: dict[str, Any] | str | None = None,
    ) -> None:
        """
        Log a reasoning step to the log file.

        Args:
            phase: Current phase (planning, gathering, analyzing, etc.)
            step: Step/iteration number
            action: Description of the action
            details: Additional details (dict or string)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
--------------------------------------------------------------------------------
[{timestamp}] STEP {step} - {phase.upper()}
--------------------------------------------------------------------------------
Action: {action}
"""
        
        if details:
            if isinstance(details, dict):
                log_entry += f"\nDetails:\n{json.dumps(details, indent=2, default=str)}\n"
            else:
                log_entry += f"\nDetails:\n{details}\n"
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_llm_response(
        self,
        phase: str,
        prompt_summary: str,
        response: str,
    ) -> None:
        """
        Log an LLM response to the reasoning log.

        Args:
            phase: Current phase
            prompt_summary: Brief summary of the prompt
            response: The LLM's response
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
--------------------------------------------------------------------------------
[{timestamp}] LLM RESPONSE - {phase.upper()}
--------------------------------------------------------------------------------
Prompt Summary: {prompt_summary}

Response:
{response}
"""
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_tool_execution(
        self,
        tool_name: str,
        args: dict[str, Any],
        result_summary: str,
        success: bool,
    ) -> None:
        """
        Log a tool execution to the reasoning log.

        Args:
            tool_name: Name of the tool executed
            args: Arguments passed to the tool
            result_summary: Brief summary of the result
            success: Whether the tool execution was successful
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status = "SUCCESS" if success else "FAILED"
        
        log_entry = f"""
[{timestamp}] TOOL CALL: {tool_name} [{status}]
  Args: {json.dumps(args, default=str)}
  Result: {result_summary}
"""
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_reasoning_step(
        self,
        step: dict[str, Any],
    ) -> None:
        """
        Log a multi-step reasoning step to the reasoning log.

        Args:
            step: A reasoning step dict with thought, action, observation, conclusion
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = f"""
================================================================================
[{timestamp}] REASONING STEP {step.get('step_number', '?')} - {step.get('phase', 'unknown').upper()}
================================================================================
💭 THOUGHT:
{step.get('thought', 'No thought recorded')}

🎯 ACTION:
{step.get('action', 'No action recorded')}

👁️ OBSERVATION:
{step.get('observation', 'No observation recorded')}

✅ CONCLUSION:
{step.get('conclusion', 'No conclusion recorded')}

📊 Confidence: {step.get('confidence', 0):.0%}
--------------------------------------------------------------------------------
"""
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_hypothesis_update(
        self,
        hypothesis: dict[str, Any],
        update_reason: str,
    ) -> None:
        """
        Log a hypothesis update to the reasoning log.

        Args:
            hypothesis: The updated hypothesis dict
            update_reason: Reason for the update
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        status_emoji = {
            "supported": "✅",
            "refuted": "❌",
            "investigating": "🔍",
            "inconclusive": "❓",
            "proposed": "📝",
        }.get(hypothesis.get("status", "proposed"), "📝")
        
        conf = hypothesis.get('confidence', 0)
        conf_bar = "█" * int(conf * 10) + "░" * (10 - int(conf * 10))
        
        log_entry = f"""
[{timestamp}] HYPOTHESIS UPDATE: {hypothesis.get('id', 'unknown')}
  {status_emoji} Status: {hypothesis.get('status', 'unknown')}
  📊 Confidence: [{conf_bar}] {conf:.0%}
  📝 Statement: {hypothesis.get('statement', 'No statement')}
  📖 Reason: {update_reason}
"""
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_reflection(
        self,
        reflection_notes: list[dict[str, Any]],
    ) -> None:
        """
        Log reflection notes to the reasoning log.

        Args:
            reflection_notes: List of reflection note dicts
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }
        
        log_entry = f"""
================================================================================
[{timestamp}] SELF-REFLECTION
================================================================================
"""
        
        for note in reflection_notes:
            emoji = severity_emoji.get(note.get("severity", "info"), "📝")
            log_entry += f"""
{emoji} [{note.get('category', 'unknown').upper()}] ({note.get('severity', 'info')})
   {note.get('content', 'No content')}
"""
            if note.get("action_required") and note.get("suggested_action"):
                log_entry += f"   → Action: {note['suggested_action']}\n"
        
        log_entry += "--------------------------------------------------------------------------------\n"
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def log_verification(
        self,
        verification_results: list[dict[str, Any]],
    ) -> None:
        """
        Log verification results to the reasoning log.

        Args:
            verification_results: List of verification result dicts
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        passed = sum(1 for v in verification_results if v.get("is_consistent"))
        failed = len(verification_results) - passed
        
        log_entry = f"""
================================================================================
[{timestamp}] VERIFICATION RESULTS
================================================================================
✅ Passed: {passed}
❌ Failed: {failed}
"""
        
        for v in verification_results:
            status = "✅" if v.get("is_consistent") else "❌"
            log_entry += f"""
{status} [{v.get('item_type', 'unknown')}] #{v.get('item_id', '?')}
   Notes: {v.get('verification_notes', 'No notes')}
"""
            if v.get("issues_found"):
                for issue in v["issues_found"]:
                    log_entry += f"   ⚠️ Issue: {issue}\n"
        
        log_entry += "--------------------------------------------------------------------------------\n"
        
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def write_final_report(self, state: dict[str, Any]) -> tuple[str, str]:
        """
        Write the final SITREP intelligence report to a Markdown file.

        Args:
            state: The final agent state containing the SITREP report

        Returns:
            Tuple of (report_content, file_path) - content for terminal display and path to file
        """
        # Generate SITREP header
        dtg = datetime.now().strftime("%d %b %Y").upper()
        task = state.get("task", "Unknown Query")
        classification = state.get("classification", "OSINT / PUBLIC")
        sources = state.get("intelligence_sources_used", [])
        
        report_lines = [
            "🔴 **COMPREHENSIVE OSINT SITREP – PROJECT OVERWATCH**",
            "",
            f"**CLASSIFICATION:** {classification}",
            f"**DTG:** {dtg} / **QUERY:** \"{task}\"",
            f"**INTELLIGENCE SOURCES:** {', '.join(sources) if sources else 'Multiple OSINT Sources'}",
            "",
            "–––",
            "",
        ]
        
        # Section I - Executive Intelligence Summary
        section_i = state.get("section_i", {})
        if section_i or state.get("executive_summary"):
            report_lines.extend([
                "## SECTION I – EXECUTIVE INTELLIGENCE SUMMARY",
                "",
                "### A. DIRECT RESPONSE TO QUERY",
                "",
                section_i.get("direct_response", state.get("executive_summary", "*No response available.*")),
                "",
                "### B. KEY INTELLIGENCE HIGHLIGHTS",
                "",
            ])
            
            highlights = section_i.get("key_highlights", [])
            if highlights:
                for h in highlights:
                    report_lines.append(f"• {h}")
            else:
                # Fallback to key_insights
                for insight in state.get("key_insights", [])[:5]:
                    if isinstance(insight, dict):
                        insight = insight.get("description", str(insight))
                    report_lines.append(f"• {insight}")
            
            report_lines.append("")
            report_lines.extend([
                "### C. CONFIDENCE ASSESSMENT",
                "",
                f"**Overall Confidence:** {section_i.get('overall_confidence_percent', 75)}%",
                f"**Intelligence Quality:** {section_i.get('intelligence_quality', 'GOOD')}",
                f"**Query Complexity:** {section_i.get('query_complexity', 'MODERATE')}",
                "",
                "–––",
                "",
            ])
        
        # Section II - Detailed Analysis
        section_ii = state.get("section_ii", {})
        topics = section_ii.get("topics", [])
        if topics:
            report_lines.extend([
                "## SECTION II – DETAILED ANALYSIS",
                "",
            ])
            
            for i, topic in enumerate(topics, 1):
                report_lines.extend([
                    f"### {i}. {topic.get('title', f'Topic {i}')}",
                    "",
                    "**Current Situation:**",
                    topic.get("current_situation", "*No situation assessment available.*"),
                    "",
                ])
                
                if topic.get("key_developments"):
                    report_lines.append("**Key Developments:**")
                    for dev in topic["key_developments"]:
                        report_lines.append(f"• {dev}")
                    report_lines.append("")
                
                if topic.get("probability_assessments"):
                    report_lines.append("**Probability Forecasts:**")
                    for prob in topic["probability_assessments"]:
                        scenario = prob.get("scenario", "Unknown")
                        pct = prob.get("probability_percent", 50)
                        timeframe = prob.get("timeframe", "Near-term")
                        report_lines.append(f"• {scenario}: **{pct}%** ({timeframe})")
                    report_lines.append("")
                
                if topic.get("evidence_citations"):
                    report_lines.append("**Evidence:**")
                    for cite in topic["evidence_citations"][:3]:
                        report_lines.append(f"• {cite}")
                    report_lines.append("")
            
            if section_ii.get("cross_topic_connections"):
                report_lines.extend([
                    "**Cross-Topic Connections:**",
                    section_ii["cross_topic_connections"],
                    "",
                ])
            
            report_lines.extend(["–––", ""])
        
        # Section III - Supporting Intelligence Analysis
        section_iii = state.get("section_iii", {})
        report_lines.extend([
            "## SECTION III – SUPPORTING INTELLIGENCE ANALYSIS",
            "",
            "### A. SATELLITE INTELLIGENCE (NASA FIRMS)",
            section_iii.get("satellite_intel", "*No satellite data collected.*"),
            "",
            "### B. NEWS INTELLIGENCE (GDELT/RSS)",
            section_iii.get("news_intel", "*No news data collected.*"),
            "",
            "### C. CYBER INTELLIGENCE (IODA/OTX)",
            section_iii.get("cyber_intel", "*No cyber intelligence collected.*"),
            "",
            "### D. SOCIAL INTELLIGENCE (Telegram)",
            section_iii.get("social_intel", "*No social media data collected.*"),
            "",
        ])
        
        if section_iii.get("cross_source_validation"):
            report_lines.extend([
                "### E. CROSS-SOURCE VALIDATION",
                section_iii["cross_source_validation"],
                "",
            ])
        
        if section_iii.get("contradictions"):
            report_lines.append("**Contradictions Found:**")
            for c in section_iii["contradictions"]:
                report_lines.append(f"• {c}")
            report_lines.append("")
        
        if section_iii.get("intelligence_gaps"):
            report_lines.append("**Intelligence Gaps:**")
            for g in section_iii["intelligence_gaps"]:
                report_lines.append(f"• {g}")
            report_lines.append("")
        
        report_lines.extend(["–––", ""])
        
        # Section IV - Actionable Intelligence & Recommendations
        section_iv = state.get("section_iv", {})
        recommendations = state.get("recommendations", [])
        report_lines.extend([
            "## SECTION IV – ACTIONABLE INTELLIGENCE & RECOMMENDATIONS",
            "",
            "### A. IMMEDIATE ACTIONS",
        ])
        
        immediate = section_iv.get("immediate_actions", recommendations[:3] if recommendations else [])
        if immediate:
            for action in immediate:
                report_lines.append(f"• {action}")
        else:
            report_lines.append("• Continue monitoring situation")
        report_lines.append("")
        
        report_lines.append("### B. MONITORING INDICATORS")
        monitoring = section_iv.get("monitoring_indicators", [])
        if monitoring:
            for ind in monitoring:
                report_lines.append(f"• {ind}")
        else:
            report_lines.append("• No specific indicators identified")
        report_lines.append("")
        
        report_lines.append("### C. FOLLOW-UP COLLECTION")
        followup = section_iv.get("follow_up_collection", [])
        if followup:
            for f in followup:
                report_lines.append(f"• {f}")
        else:
            report_lines.append("• Forward collection required to address intelligence gaps")
        report_lines.extend(["", "–––", ""])
        
        # Section V - Intelligence Assessment Metadata
        section_v = state.get("section_v", {})
        report_lines.extend([
            "## SECTION V – INTELLIGENCE ASSESSMENT METADATA",
            "",
            "### A. SOURCE RELIABILITY MATRIX",
            "",
            "| Source | Reliability | Credibility | Timeliness | Grade |",
            "|--------|-------------|-------------|------------|-------|",
        ])
        
        # Build reliability matrix
        reliability_matrix = section_v.get("source_reliability_matrix", [])
        if reliability_matrix:
            for entry in reliability_matrix:
                name = entry.get("source_name", "Unknown")
                rel = entry.get("reliability", "B")
                cred = entry.get("credibility", "2")
                time = entry.get("timeliness", "Current")
                grade = entry.get("grade", f"{rel}-{cred}")
                report_lines.append(f"| {name} | {rel} | {cred} | {time} | {grade} |")
        else:
            # Generate default matrix from executed queries
            tool_sources = {}
            for q in state.get("executed_queries", []):
                tool = q.get("tool", "unknown")
                status = q.get("status", "unknown")
                if tool not in tool_sources:
                    tool_sources[tool] = {"success": 0, "failed": 0}
                if status == "success":
                    tool_sources[tool]["success"] += 1
                else:
                    tool_sources[tool]["failed"] += 1
            
            tool_to_source = {
                "search_news": ("GDELT News", "B", "2"),
                "fetch_rss_news": ("RSS Feeds", "B", "2"),
                "detect_thermal_anomalies": ("NASA FIRMS", "A", "2"),
                "check_connectivity": ("IODA", "B", "2"),
                "check_traffic_metrics": ("Cloudflare", "B", "2"),
                "search_telegram": ("Telegram", "C", "3"),
                "check_ioc": ("AlienVault OTX", "B", "2"),
                "search_threats": ("AlienVault OTX", "B", "2"),
            }
            
            seen_sources = set()
            for tool, counts in tool_sources.items():
                if tool in tool_to_source:
                    name, rel, cred = tool_to_source[tool]
                    if name not in seen_sources:
                        seen_sources.add(name)
                        grade = f"{rel}-{cred}"
                        report_lines.append(f"| {name} | {rel} | {cred} | Current | {grade} |")
        
        report_lines.extend([
            "",
            "### B. ANALYTICAL CONFIDENCE",
            section_v.get("analytical_confidence", state.get("confidence_assessment", "Moderate confidence based on available sources.")),
            "",
        ])
        
        if section_v.get("key_assumptions"):
            report_lines.append("**Key Assumptions:**")
            for a in section_v["key_assumptions"]:
                report_lines.append(f"• {a}")
            report_lines.append("")
        
        if section_v.get("alternative_scenarios"):
            report_lines.append("**Alternative Scenarios:**")
            for s in section_v["alternative_scenarios"]:
                report_lines.append(f"• {s}")
            report_lines.append("")
        
        report_lines.extend([
            "### C. INTELLIGENCE FRESHNESS",
            section_v.get("data_freshness", f"Data collected: {datetime.now().strftime('%Y-%m-%d')}"),
            "",
            "–––",
            "",
        ])
        
        # Section VI - Forward Intelligence Requirements
        section_vi = state.get("section_vi", {})
        report_lines.extend([
            "## SECTION VI – FORWARD INTELLIGENCE REQUIREMENTS",
            "",
            "### A. PRIORITY COLLECTION",
        ])
        
        priority = section_vi.get("priority_collection", [])
        if priority:
            for p in priority:
                report_lines.append(f"1. {p}")
        else:
            report_lines.append("• Collect additional temporal data to identify trends")
        report_lines.append("")
        
        report_lines.append("### B. MONITORING & EARLY WARNING")
        triggers = section_vi.get("early_warning_triggers", [])
        if triggers:
            for t in triggers:
                report_lines.append(f"• {t}")
        else:
            report_lines.append("• Monitor for significant changes in assessed situation")
        report_lines.extend(["", "–––", ""])
        
        # Footer
        report_lines.extend([
            f"**CLASSIFICATION:** {classification}",
            f"**ANALYST:** Project Overwatch OSINT Intelligence System",
            f"**SESSION:** {self.session_id}",
            "",
            "〔END SITREP〕",
        ])
        
        # Write the report
        report_content = "\n".join(report_lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return report_content, str(self.report_path)

    def save_state(self, state: dict[str, Any]) -> str:
        """
        Save the complete state as JSON for debugging/analysis.

        Args:
            state: The agent state to save

        Returns:
            Path to the saved state file
        """
        # Remove messages from state (they contain non-serializable objects)
        serializable_state = {
            k: v for k, v in state.items()
            if k != "messages"
        }
        
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(serializable_state, f, indent=2, default=str)
        
        return str(self.state_path)

    def finalize(self, state: dict[str, Any]) -> dict[str, str]:
        """
        Finalize the session by writing all output files.

        Args:
            state: The final agent state

        Returns:
            Dict with paths to all output files
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log reasoning trace summary
        reasoning_trace = state.get("reasoning_trace", [])
        if reasoning_trace:
            trace_entry = f"""
================================================================================
REASONING TRACE SUMMARY
================================================================================
Total Reasoning Steps: {len(reasoning_trace)}
Reasoning Depth: {state.get('reasoning_depth', 0)}
Reflection Iterations: {state.get('reflection_iterations', 0)}
================================================================================

"""
            with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
                f.write(trace_entry)
            
            # Log each reasoning step
            for step in reasoning_trace:
                self.log_reasoning_step(step)
        
        # Log hypotheses final state
        if state.get("hypotheses"):
            hyp_entry = f"""
================================================================================
FINAL HYPOTHESIS STATUS
================================================================================
"""
            with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
                f.write(hyp_entry)
            
            for h in state["hypotheses"]:
                self.log_hypothesis_update(h, "Final status")
        
        # Log reflection notes
        if state.get("reflection_notes"):
            self.log_reflection(state["reflection_notes"])
        
        # Log verification results
        if state.get("verification_results"):
            self.log_verification(state["verification_results"])
        
        # Log completion
        completion_entry = f"""
================================================================================
RESEARCH COMPLETED
================================================================================
Timestamp: {timestamp}
Total Iterations: {state.get('iteration', 0)}
Total Findings: {len(state.get('findings', []))}
Task Complexity: {state.get('task_complexity', 'N/A')}
Sub-tasks: {len(state.get('sub_tasks', []))}
Hypotheses: {len(state.get('hypotheses', []))}
Verified Insights: {len(state.get('verified_insights', []))}
Verified Correlations: {len(state.get('verified_correlations', []))}
================================================================================
"""
        with open(self.reasoning_log_path, "a", encoding="utf-8") as f:
            f.write(completion_entry)
        
        # Write final report
        report_content, report_path = self.write_final_report(state)
        
        # Save state JSON
        state_path = self.save_state(state)
        
        return {
            "report": report_path,
            "report_content": report_content,
            "reasoning_log": str(self.reasoning_log_path),
            "state": state_path,
            "session_dir": str(self.session_dir),
        }


# Global output writer instance (initialized per session)
_current_writer: OutputWriter | None = None


def get_output_writer(session_id: str | None = None, query: str | None = None) -> OutputWriter:
    """
    Get or create the output writer for the current session.

    Args:
        session_id: Optional session ID (creates new writer if provided)
        query: Optional research query (used to name the output folder)

    Returns:
        OutputWriter instance
    """
    global _current_writer
    
    if session_id is not None or query is not None or _current_writer is None:
        _current_writer = OutputWriter(session_id=session_id, query=query)
    
    return _current_writer


def reset_output_writer() -> None:
    """Reset the global output writer (for new sessions)."""
    global _current_writer
    _current_writer = None

