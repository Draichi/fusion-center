"""
Output Writer Module for Project Overwatch.

Handles writing research outputs:
- Final report as Markdown (.md)
- Reasoning steps as a separate log file
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.shared.config import settings


class OutputWriter:
    """Handles writing agent outputs to files."""

    def __init__(self, output_dir: str | None = None, session_id: str | None = None):
        """
        Initialize the output writer.

        Args:
            output_dir: Directory for output files (default from settings)
            session_id: Unique session identifier (auto-generated if not provided)
        """
        self.output_dir = Path(output_dir or settings.output_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
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

    def write_final_report(self, state: dict[str, Any]) -> str:
        """
        Write the final research report to a Markdown file.

        Args:
            state: The final agent state containing the report

        Returns:
            Path to the written report file
        """
        task = state.get("task", "Unknown Task")
        executive_summary = state.get("executive_summary", "No summary available.")
        detailed_report = state.get("detailed_report", "No detailed report available.")
        recommendations = state.get("recommendations", [])
        confidence_assessment = state.get("confidence_assessment", "")
        key_insights = state.get("key_insights", [])
        correlations = state.get("correlations", [])
        uncertainties = state.get("uncertainties", [])
        
        # Build the markdown report
        report_lines = [
            "# Intelligence Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Session ID:** {self.session_id}",
            "",
            "---",
            "",
            "## Research Task",
            "",
            f"> {task}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            executive_summary or "*No executive summary available.*",
            "",
        ]
        
        # Key Insights
        if key_insights:
            report_lines.extend([
                "---",
                "",
                "## Key Insights",
                "",
            ])
            for i, insight in enumerate(key_insights, 1):
                if insight:
                    if isinstance(insight, dict):
                        insight_text = insight.get("description", str(insight))
                    else:
                        insight_text = str(insight)
                    report_lines.append(f"{i}. {insight_text}")
            report_lines.append("")
        
        # Detailed Report
        if detailed_report:
            report_lines.extend([
                "---",
                "",
                "## Detailed Analysis",
                "",
                detailed_report,
                "",
            ])
        
        # Correlations
        if correlations:
            report_lines.extend([
                "---",
                "",
                "## Correlations Found",
                "",
            ])
            for corr in correlations:
                corr_type = corr.get("correlation_type", "unknown")
                description = corr.get("description", "No description")
                confidence = corr.get("confidence", "unknown")
                implications = corr.get("implications", [])
                
                report_lines.extend([
                    f"### {corr_type.title()} Correlation",
                    "",
                    f"**Confidence:** {confidence}",
                    "",
                    description,
                    "",
                ])
                
                if implications:
                    report_lines.append("**Implications:**")
                    for imp in implications:
                        report_lines.append(f"- {imp}")
                    report_lines.append("")
        
        # Recommendations
        if recommendations:
            report_lines.extend([
                "---",
                "",
                "## Recommendations",
                "",
            ])
            for i, rec in enumerate(recommendations, 1):
                report_lines.append(f"{i}. {rec}")
            report_lines.append("")
        
        # Confidence Assessment
        if confidence_assessment:
            report_lines.extend([
                "---",
                "",
                "## Confidence Assessment",
                "",
                confidence_assessment,
                "",
            ])
        
        # Uncertainties
        if uncertainties:
            report_lines.extend([
                "---",
                "",
                "## Uncertainties & Information Gaps",
                "",
            ])
            for unc in uncertainties:
                report_lines.append(f"- {unc}")
            report_lines.append("")
        
        # Multi-step Reasoning section
        if state.get("hypotheses") or state.get("reasoning_trace") or state.get("reflection_notes"):
            report_lines.extend([
                "---",
                "",
                "## Multi-step Reasoning Process",
                "",
            ])
            
            # Hypotheses
            if state.get("hypotheses"):
                report_lines.extend([
                    "### Hypotheses Tested",
                    "",
                ])
                for h in state["hypotheses"]:
                    status_emoji = {
                        "supported": "✅",
                        "refuted": "❌",
                        "investigating": "🔍",
                        "inconclusive": "❓",
                        "proposed": "📝",
                    }.get(h.get("status", "proposed"), "📝")
                    conf = h.get("confidence", 0)
                    report_lines.append(f"- {status_emoji} **{h.get('id', 'H?')}**: {h.get('statement', 'No statement')} (confidence: {conf:.0%})")
                report_lines.append("")
            
            # Reflection summary
            if state.get("reflection_notes"):
                critical = [n for n in state["reflection_notes"] if n.get("severity") == "critical"]
                warnings = [n for n in state["reflection_notes"] if n.get("severity") == "warning"]
                if critical or warnings:
                    report_lines.extend([
                        "### Self-Reflection Notes",
                        "",
                    ])
                    for note in critical[:3]:
                        report_lines.append(f"- 🚨 **Critical**: {note.get('content', 'No content')}")
                    for note in warnings[:3]:
                        report_lines.append(f"- ⚠️ **Warning**: {note.get('content', 'No content')}")
                    report_lines.append("")
            
            # Reasoning depth
            report_lines.extend([
                "### Reasoning Statistics",
                "",
                f"- **Reasoning Depth:** {state.get('reasoning_depth', 0)} steps",
                f"- **Reflection Iterations:** {state.get('reflection_iterations', 0)}",
                f"- **Verified Items:** {len(state.get('verification_results', []))}",
                "",
            ])
        
        # Sources Used section
        executed_queries = state.get("executed_queries", [])
        findings = state.get("findings", [])
        
        if executed_queries or findings:
            report_lines.extend([
                "---",
                "",
                "## Sources Used",
                "",
            ])
            
            # Group queries by tool/source
            sources_summary = {}
            for query in executed_queries:
                tool = query.get("tool", "unknown")
                status = query.get("status", "unknown")
                
                if tool not in sources_summary:
                    sources_summary[tool] = {"success": 0, "failed": 0, "queries": []}
                
                if status == "success":
                    sources_summary[tool]["success"] += 1
                else:
                    sources_summary[tool]["failed"] += 1
                
                sources_summary[tool]["queries"].append(query)
            
            # Map tool names to human-readable source names
            tool_to_source = {
                "search_news": "GDELT News Database",
                "detect_thermal_anomalies": "NASA FIRMS Satellite Data",
                "check_connectivity": "IODA Internet Monitoring",
                "check_traffic_metrics": "Cloudflare Radar",
                "check_ioc": "AlienVault OTX IoC Lookup",
                "search_threats": "AlienVault OTX Threat Pulses",
                "get_threat_pulse": "AlienVault OTX Pulse Details",
            }
            
            for tool, info in sources_summary.items():
                source_name = tool_to_source.get(tool, tool)
                success_count = info["success"]
                failed_count = info["failed"]
                
                status_icon = "✅" if failed_count == 0 else "⚠️" if success_count > 0 else "❌"
                
                report_lines.append(f"### {status_icon} {source_name}")
                report_lines.append("")
                report_lines.append(f"- **Queries:** {success_count + failed_count} total ({success_count} successful, {failed_count} failed)")
                
                # Show query details
                for q in info["queries"][:5]:  # Limit to first 5
                    args = q.get("args", {})
                    status = "✓" if q.get("status") == "success" else "✗"
                    
                    # Format query args nicely
                    if tool == "search_news":
                        keywords = args.get("keywords", "N/A")
                        country = args.get("source_country", "Global")
                        report_lines.append(f"  - {status} Keywords: `{keywords[:50]}...` | Source: {country}")
                    elif tool == "detect_thermal_anomalies":
                        lat = args.get("latitude", "?")
                        lon = args.get("longitude", "?")
                        radius = args.get("radius_km", "?")
                        report_lines.append(f"  - {status} Location: ({lat}, {lon}) | Radius: {radius}km")
                    elif tool == "check_connectivity":
                        region = args.get("region_name") or args.get("country_code", "N/A")
                        report_lines.append(f"  - {status} Region: {region}")
                    elif tool == "check_traffic_metrics":
                        country = args.get("country_code", "N/A")
                        metric = args.get("metric", "traffic")
                        report_lines.append(f"  - {status} Country: {country} | Metric: {metric}")
                    else:
                        report_lines.append(f"  - {status} {args}")
                
                if len(info["queries"]) > 5:
                    report_lines.append(f"  - ... and {len(info['queries']) - 5} more queries")
                
                report_lines.append("")
            
            # Add news article sources if available
            news_sources = set()
            for finding in findings:
                if finding.get("source_type") == "news":
                    domain = finding.get("source") or finding.get("content", {}).get("domain")
                    if domain:
                        news_sources.add(domain)
            
            if news_sources:
                report_lines.extend([
                    "### 📰 News Sources Cited",
                    "",
                ])
                for source in sorted(news_sources)[:20]:  # Limit to 20 sources
                    report_lines.append(f"- {source}")
                if len(news_sources) > 20:
                    report_lines.append(f"- ... and {len(news_sources) - 20} more sources")
                report_lines.append("")
        
        # Metadata footer
        report_lines.extend([
            "---",
            "",
            "## Research Metadata",
            "",
            f"- **Iterations:** {state.get('iteration', 0)}",
            f"- **Findings Collected:** {len(state.get('findings', []))}",
            f"- **Queries Executed:** {len(state.get('executed_queries', []))}",
            f"- **Task Complexity:** {state.get('task_complexity', 'N/A')}",
            f"- **Sub-tasks:** {len(state.get('sub_tasks', []))}",
            f"- **Started:** {state.get('started_at', 'N/A')}",
            f"- **Completed:** {state.get('last_updated', 'N/A')}",
            "",
            "---",
            "",
            f"*Report generated by Project Overwatch with Multi-step Reasoning*",
            f"*Reasoning log available at: `{self.reasoning_log_path.name}`*",
        ])
        
        # Write the report
        report_content = "\n".join(report_lines)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        return str(self.report_path)

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
        report_path = self.write_final_report(state)
        
        # Save state JSON
        state_path = self.save_state(state)
        
        return {
            "report": report_path,
            "reasoning_log": str(self.reasoning_log_path),
            "state": state_path,
            "session_dir": str(self.session_dir),
        }


# Global output writer instance (initialized per session)
_current_writer: OutputWriter | None = None


def get_output_writer(session_id: str | None = None) -> OutputWriter:
    """
    Get or create the output writer for the current session.

    Args:
        session_id: Optional session ID (creates new writer if provided)

    Returns:
        OutputWriter instance
    """
    global _current_writer
    
    if session_id is not None or _current_writer is None:
        _current_writer = OutputWriter(session_id=session_id)
    
    return _current_writer


def reset_output_writer() -> None:
    """Reset the global output writer (for new sessions)."""
    global _current_writer
    _current_writer = None

