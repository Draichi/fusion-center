"""
Gathering Node - Executes queries to gather intelligence data.
"""

from datetime import datetime
from typing import Any

from src.agent.state import AgentState, ResearchPhase, IntelligenceType
from src.agent.tools import MCPToolExecutor, VALID_TOOL_NAMES, TOOL_NAME_ALIASES


def _resolve_tool_name(tool_name: str) -> str:
    """
    Resolve a tool name to its canonical form, handling aliases.
    
    Args:
        tool_name: The tool name, possibly an alias
        
    Returns:
        The resolved canonical tool name
    """
    if tool_name in VALID_TOOL_NAMES:
        return tool_name
    if tool_name in TOOL_NAME_ALIASES:
        return TOOL_NAME_ALIASES[tool_name]
    return tool_name
from src.shared.logger import get_logger, log_agent_step
from src.shared.output_writer import get_output_writer

logger = get_logger()


def _log_tool_result_summary(tool_name: str, result: dict[str, Any]) -> None:
    """Log a brief summary of tool results."""
    summary_lines = []
    
    if tool_name == "search_news":
        articles = result.get("articles", [])
        if articles:
            summary_lines.append(f"[bold]Found {len(articles)} articles:[/bold]")
            for article in articles:
                title = article.get("title", "No title")
                source = article.get("domain", "unknown")
                summary_lines.append(f"  • [{source}] {title}")
    
    elif tool_name == "detect_thermal_anomalies":
        count = result.get("anomaly_count", 0)
        if count > 0:
            summary_lines.append(f"[bold]Detected {count} thermal anomalies[/bold]")
            anomalies = result.get("anomalies", [])
            for a in anomalies:
                summary_lines.append(
                    f"  • ({a.get('latitude'):.2f}, {a.get('longitude'):.2f}) "
                    f"brightness={a.get('brightness')} conf={a.get('confidence')}"
                )
    
    elif tool_name == "check_connectivity":
        status = result.get("current_status", {})
        if status:
            summary_lines.append(f"[bold]Connectivity Status:[/bold] {status.get('status', 'unknown')}")
            if status.get("bgp_visibility"):
                summary_lines.append(f"  • BGP Visibility: {status['bgp_visibility']:.1f}%")
        outages = result.get("recent_outages", [])
        if outages:
            summary_lines.append(f"  • Recent outages: {len(outages)}")
    
    elif tool_name == "check_ioc":
        indicator_info = result.get("indicator_info", {})
        pulse_count = indicator_info.get("pulse_count", 0)
        if pulse_count > 0:
            summary_lines.append(f"[bold]Found {pulse_count} threat pulses for indicator[/bold]")
            malware = indicator_info.get("malware_families", [])
            if malware:
                summary_lines.append(f"  • Malware families: {', '.join(malware[:5])}")
        else:
            summary_lines.append("[bold]Indicator not found in threat database (clean)[/bold]")
    
    elif tool_name == "search_threats":
        pulses = result.get("pulses", [])
        if pulses:
            summary_lines.append(f"[bold]Found {len(pulses)} threat pulses[/bold]")
            for p in pulses[:5]:
                summary_lines.append(f"  • {p.get('name', 'Unknown')}")
    
    elif tool_name == "check_traffic_metrics":
        metrics = result.get("metrics", {})
        annotations = metrics.get("meta", {}).get("confidenceInfo", {}).get("annotations", [])
        summary_lines.append(f"[bold]Cloudflare Radar metrics for {result.get('country', 'unknown')}[/bold]")
        summary = metrics.get("summary_0", {})
        if summary:
            summary_lines.append(f"  • Desktop: {summary.get('desktop', 'N/A')}%, Mobile: {summary.get('mobile', 'N/A')}%")
        if annotations:
            outage_count = sum(1 for a in annotations if a.get("eventType") == "OUTAGE")
            if outage_count:
                summary_lines.append(f"  • [bold red]⚠ {outage_count} outage event(s) detected[/bold red]")
                for a in annotations[:2]:
                    if a.get("eventType") == "OUTAGE":
                        summary_lines.append(f"    → {a.get('description', 'No description')[:80]}...")
    
    elif tool_name == "search_telegram":
        messages = result.get("messages", [])
        if messages:
            summary_lines.append(f"[bold]Found {len(messages)} Telegram messages[/bold]")
            for msg in messages[:3]:
                channel = msg.get("channel_username", "unknown")
                text_preview = msg.get("text", "")[:60]
                summary_lines.append(f"  • @{channel}: {text_preview}...")
    
    if summary_lines:
        logger.panel("\n".join(summary_lines), title=f"📥 {tool_name} Results", style="dim cyan")


def _extract_findings(
    tool_name: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract structured findings from tool results."""
    findings = []
    timestamp = datetime.utcnow().isoformat()
    
    if tool_name == "search_news":
        articles = result.get("articles", [])
        for article in articles:
            findings.append({
                "source": article.get("domain", "unknown"),
                "source_type": IntelligenceType.NEWS.value,
                "timestamp": timestamp,
                "content": {
                    "title": article.get("title"),
                    "url": article.get("url"),
                    "date": article.get("seendate"),
                    "country": article.get("sourcecountry"),
                },
                "relevance_score": 0.8,
                "confidence": "medium",
            })
    
    elif tool_name == "fetch_rss_news":
        articles = result.get("articles", [])
        source_name = result.get("source", "RSS")
        for article in articles:
            findings.append({
                "source": source_name,
                "source_type": IntelligenceType.NEWS.value,
                "timestamp": timestamp,
                "content": {
                    "title": article.get("title"),
                    "url": article.get("link"),
                    "date": article.get("published"),
                    "description": article.get("description"),
                },
                "relevance_score": 0.8,
                "confidence": "medium",
            })
    
    elif tool_name == "detect_thermal_anomalies":
        anomalies = result.get("anomalies", [])
        for anomaly in anomalies:
            findings.append({
                "source": "NASA FIRMS",
                "source_type": IntelligenceType.SATELLITE.value,
                "timestamp": timestamp,
                "content": {
                    "brightness": anomaly.get("brightness"),
                    "confidence": anomaly.get("confidence"),
                    "acq_date": anomaly.get("acq_date"),
                    "acq_time": anomaly.get("acq_time"),
                    "frp": anomaly.get("frp"),
                },
                "location": {
                    "lat": anomaly.get("latitude"),
                    "lon": anomaly.get("longitude"),
                },
                "relevance_score": 0.95,
                "confidence": anomaly.get("confidence", "nominal"),
            })
    
    elif tool_name == "check_connectivity":
        status = result.get("current_status", {})
        outages = result.get("recent_outages", [])
        
        if status:
            findings.append({
                "source": "IODA",
                "source_type": IntelligenceType.CYBER.value,
                "timestamp": timestamp,
                "content": {
                    "region": status.get("region"),
                    "status": status.get("status"),
                    "bgp_visibility": status.get("bgp_visibility"),
                },
                "relevance_score": 0.85,
                "confidence": "high",
            })
        
        for outage in outages:
            metrics = outage.get("metrics", {})
            findings.append({
                "source": "IODA",
                "source_type": IntelligenceType.CYBER.value,
                "timestamp": timestamp,
                "content": {
                    "region": outage.get("region"),
                    "severity": outage.get("severity"),
                    "start_time": outage.get("start_time"),
                    "data_source": outage.get("data_source"),
                    "degradation_ratio": metrics.get("ratio"),
                },
                "relevance_score": 0.9,
                "confidence": "high",
            })
    
    elif tool_name == "check_ioc":
        indicator_info = result.get("indicator_info", {})
        pulses = result.get("pulses", [])
        
        if indicator_info.get("pulse_count", 0) > 0:
            findings.append({
                "source": "AlienVault OTX",
                "source_type": IntelligenceType.THREAT_INTEL.value,
                "timestamp": timestamp,
                "content": {
                    "indicator": indicator_info.get("indicator"),
                    "indicator_type": indicator_info.get("indicator_type"),
                    "pulse_count": indicator_info.get("pulse_count"),
                    "malware_families": indicator_info.get("malware_families", []),
                    "country": indicator_info.get("country"),
                },
                "relevance_score": min(1.0, 0.5 + (indicator_info.get("pulse_count", 0) * 0.1)),
                "confidence": "high" if indicator_info.get("pulse_count", 0) > 5 else "medium",
            })
        
        for pulse in pulses[:5]:
            findings.append({
                "source": "AlienVault OTX",
                "source_type": IntelligenceType.THREAT_INTEL.value,
                "timestamp": timestamp,
                "content": {
                    "pulse_name": pulse.get("name"),
                    "pulse_id": pulse.get("pulse_id"),
                    "description": pulse.get("description", "")[:200],
                    "tags": pulse.get("tags", []),
                    "malware_families": pulse.get("malware_families", []),
                },
                "relevance_score": 0.85,
                "confidence": "medium",
            })
    
    elif tool_name == "search_threats":
        pulses = result.get("pulses", [])
        for pulse in pulses:
            findings.append({
                "source": "AlienVault OTX",
                "source_type": IntelligenceType.THREAT_INTEL.value,
                "timestamp": timestamp,
                "content": {
                    "pulse_name": pulse.get("name"),
                    "pulse_id": pulse.get("pulse_id"),
                    "description": pulse.get("description", "")[:300],
                    "author": pulse.get("author_name"),
                    "indicator_count": pulse.get("indicator_count", 0),
                    "tags": pulse.get("tags", []),
                    "targeted_countries": pulse.get("targeted_countries", []),
                    "malware_families": pulse.get("malware_families", []),
                },
                "relevance_score": 0.8,
                "confidence": "medium",
            })
    
    elif tool_name == "check_traffic_metrics":
        # Cloudflare Radar data
        metrics = result.get("metrics", {})
        meta = metrics.get("meta", {})
        annotations = meta.get("confidenceInfo", {}).get("annotations", [])
        
        # Extract traffic summary
        summary = metrics.get("summary_0", {})
        if summary:
            findings.append({
                "source": "Cloudflare Radar",
                "source_type": IntelligenceType.CYBER.value,
                "timestamp": timestamp,
                "content": {
                    "country": result.get("country"),
                    "metric_type": result.get("metric_type"),
                    "desktop_traffic": summary.get("desktop"),
                    "mobile_traffic": summary.get("mobile"),
                },
                "relevance_score": 0.7,
                "confidence": "high",
            })
        
        # Extract outage annotations (very valuable!)
        for annotation in annotations:
            if annotation.get("eventType") == "OUTAGE":
                findings.append({
                    "source": "Cloudflare Radar",
                    "source_type": IntelligenceType.CYBER.value,
                    "timestamp": timestamp,
                    "content": {
                        "event_type": annotation.get("eventType"),
                        "description": annotation.get("description"),
                        "start_date": annotation.get("startDate"),
                        "end_date": annotation.get("endDate"),
                        "linked_url": annotation.get("linkedUrl"),
                    },
                    "relevance_score": 0.95,
                    "confidence": "high",
                })
    
    elif tool_name == "search_telegram":
        # Telegram OSINT messages
        messages = result.get("messages", [])
        for msg in messages:
            # Calculate relevance based on views (more views = more relevant)
            views = msg.get("views") or 0
            if views > 10000:
                relevance = 0.95
                confidence = "high"
            elif views > 1000:
                relevance = 0.85
                confidence = "medium"
            else:
                relevance = 0.7
                confidence = "low"
            
            findings.append({
                "source": f"Telegram @{msg.get('channel_username', 'unknown')}",
                "source_type": IntelligenceType.SOCIAL.value,
                "timestamp": timestamp,
                "content": {
                    "channel": msg.get("channel_name"),
                    "channel_username": msg.get("channel_username"),
                    "text": msg.get("text", "")[:500],  # Truncate long messages
                    "message_date": msg.get("date"),
                    "views": views,
                    "url": msg.get("url"),
                    "has_media": msg.get("has_media", False),
                },
                "relevance_score": relevance,
                "confidence": confidence,
            })
    
    return findings


async def gather_intelligence(
    state: AgentState,
    tool_executor: MCPToolExecutor,
) -> dict[str, Any]:
    """
    Execute pending queries to gather intelligence.
    
    This is a special node that doesn't use LLM, but executes tools directly.
    """
    log_agent_step(state["iteration"] + 1, "Gathering intelligence data")
    
    writer = get_output_writer()
    writer.log_reasoning(
        phase="gathering",
        step=state["iteration"] + 1,
        action="Executing intelligence queries",
        details={"pending_queries_count": len(state.get("pending_queries", []))},
    )
    
    pending = state.get("pending_queries", [])
    if not pending:
        logger.info("No pending queries to execute")
        return {
            "current_phase": ResearchPhase.ANALYZING.value,
            "iteration": state["iteration"] + 1,
            "last_updated": datetime.utcnow().isoformat(),
        }
    
    new_findings = []
    executed = []
    
    # Execute up to 5 queries per iteration to avoid overload
    queries_to_run = pending[:5]
    remaining = pending[5:]
    
    # Define required parameters for each tool
    REQUIRED_PARAMS = {
        "search_news": ["keywords"],
        "fetch_rss_news": ["source"],
        "detect_thermal_anomalies": ["latitude", "longitude"],
        "check_connectivity": ["country_code"],
        "check_traffic_metrics": ["country_code"],
        "check_ioc": ["indicator"],
        "search_threats": ["query"],
        "get_threat_pulse": ["pulse_id"],
        "search_telegram": ["keywords"],
    }
    
    for query in queries_to_run:
        tool_name = query.get("tool")
        args = query.get("args", {})
        
        # Resolve tool name to handle aliases
        resolved_tool_name = _resolve_tool_name(tool_name)
        
        # Validate required parameters before execution
        required = REQUIRED_PARAMS.get(resolved_tool_name, [])
        missing = [p for p in required if p not in args or not args[p]]
        if missing:
            logger.warning(f"Skipping {tool_name}: missing required parameters {missing}")
            writer.log_tool_execution(
                tool_name=tool_name,
                args=args,
                result_summary=f"Skipped: Missing required parameters: {missing}",
                success=False,
            )
            continue
        
        logger.info(f"Executing query: [bold]{tool_name}[/bold]")
        
        result = await tool_executor.execute(tool_name, args)
        
        executed.append({
            "tool": tool_name,
            "args": args,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        # Process results into findings
        if result.get("status") == "success" or result.get("status") == "stub":
            findings = _extract_findings(resolved_tool_name, args, result)
            new_findings.extend(findings)
            logger.success(f"Extracted {len(findings)} findings from {tool_name}")
            
            writer.log_tool_execution(
                tool_name=tool_name,
                args=args,
                result_summary=f"Extracted {len(findings)} findings",
                success=True,
            )
            
            _log_tool_result_summary(resolved_tool_name, result)
        else:
            error_msg = result.get("error_message") or result.get("error", "Unknown error")
            logger.warning(f"Query returned error: {error_msg}")
            
            writer.log_tool_execution(
                tool_name=tool_name,
                args=args,
                result_summary=f"Error: {error_msg}",
                success=False,
            )
    
    all_findings = state.get("findings", []) + new_findings
    all_executed = state.get("executed_queries", []) + executed
    
    # Determine next phase
    next_phase = ResearchPhase.GATHERING.value if remaining else ResearchPhase.ANALYZING.value
    
    # Reset analysis-reflection cycle counter when we finish gathering and go to analysis
    # This indicates we're breaking out of the analysis-reflection loop with new data
    reset_cycles = 0 if not remaining else state.get("analysis_reflection_cycles", 0)
    
    return {
        "findings": all_findings,
        "executed_queries": all_executed,
        "pending_queries": remaining,
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "analysis_reflection_cycles": reset_cycles,
        "last_updated": datetime.utcnow().isoformat(),
    }

