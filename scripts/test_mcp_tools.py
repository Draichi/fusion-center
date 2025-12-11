#!/usr/bin/env python3
"""
MCP Tools Test Script

This script tests all MCP tools to identify which ones are working,
returning errors, or returning empty responses.

Usage:
    uv run python scripts/test_mcp_tools.py
    
    # Test specific tools only
    uv run python scripts/test_mcp_tools.py --tools gdelt,ioda
    
    # Skip tools that require API keys
    uv run python scripts/test_mcp_tools.py --skip-api-required
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path for imports
sys.path.insert(0, str(__file__).replace("/scripts/test_mcp_tools.py", ""))

from src.shared.config import settings


class TestStatus(Enum):
    """Test result status."""
    SUCCESS = "✅ Success"
    ERROR = "❌ Error"
    EMPTY = "⚠️ Empty"
    SKIPPED = "⏭️ Skipped"
    STUB = "📝 Stub"


@dataclass
class TestResult:
    """Result of a tool test."""
    tool_name: str
    status: TestStatus
    message: str
    data_count: int = 0
    response_time_ms: float = 0
    raw_response: dict[str, Any] | None = None


console = Console()


# =============================================================================
# Tool Test Functions
# =============================================================================


async def test_gdelt_news() -> TestResult:
    """Test GDELT news search tool."""
    from src.mcp_server.tools.news import query_gdelt_events
    
    start = datetime.now()
    try:
        result = await query_gdelt_events(
            keywords="Ukraine",
            max_records=5,
            timespan="1d"
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="GDELT News (search_news)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        article_count = result.get("article_count", 0)
        if article_count == 0:
            return TestResult(
                tool_name="GDELT News (search_news)",
                status=TestStatus.EMPTY,
                message="No articles found (API working but no results)",
                response_time_ms=elapsed,
                raw_response=result
            )
        
        return TestResult(
            tool_name="GDELT News (search_news)",
            status=TestStatus.SUCCESS,
            message=f"Found {article_count} articles",
            data_count=article_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="GDELT News (search_news)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_nasa_firms() -> TestResult:
    """Test NASA FIRMS thermal anomaly detection."""
    from src.mcp_server.tools.geo import check_nasa_firms
    
    # Check if API key is configured
    if not settings.has_nasa_key:
        return TestResult(
            tool_name="NASA FIRMS (detect_thermal_anomalies)",
            status=TestStatus.SKIPPED,
            message="NASA_FIRMS_API_KEY not configured"
        )
    
    start = datetime.now()
    try:
        # Test with Kyiv coordinates
        result = await check_nasa_firms(
            latitude=50.4501,
            longitude=30.5234,
            day_range=3,
            radius_km=25
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="NASA FIRMS (detect_thermal_anomalies)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        anomaly_count = result.get("anomaly_count", 0)
        return TestResult(
            tool_name="NASA FIRMS (detect_thermal_anomalies)",
            status=TestStatus.SUCCESS if anomaly_count > 0 else TestStatus.EMPTY,
            message=f"Found {anomaly_count} thermal anomalies" if anomaly_count > 0 else "No anomalies found (API working)",
            data_count=anomaly_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="NASA FIRMS (detect_thermal_anomalies)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_ioda_connectivity() -> TestResult:
    """Test IODA internet connectivity monitoring."""
    from src.mcp_server.tools.cyber import check_internet_outages
    
    start = datetime.now()
    try:
        result = await check_internet_outages(
            country_code="UA",
            hours_back=24
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="IODA Connectivity (check_connectivity)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        current_status = result.get("current_status", {})
        outage_count = len(result.get("recent_outages", []))
        status_level = current_status.get("status", "unknown")
        
        return TestResult(
            tool_name="IODA Connectivity (check_connectivity)",
            status=TestStatus.SUCCESS,
            message=f"Status: {status_level}, {outage_count} recent outages",
            data_count=outage_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="IODA Connectivity (check_connectivity)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_cloudflare_radar() -> TestResult:
    """Test Cloudflare Radar traffic metrics."""
    from src.mcp_server.tools.cyber import check_cloudflare_radar
    
    start = datetime.now()
    try:
        result = await check_cloudflare_radar(
            country_code="UA",
            metric="traffic"
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="Cloudflare Radar (check_traffic_metrics)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        metrics = result.get("metrics", {})
        metric_count = len(metrics)
        
        return TestResult(
            tool_name="Cloudflare Radar (check_traffic_metrics)",
            status=TestStatus.SUCCESS if metric_count > 0 else TestStatus.EMPTY,
            message=f"Retrieved {metric_count} metrics" if metric_count > 0 else "No metrics available",
            data_count=metric_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="Cloudflare Radar (check_traffic_metrics)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_ioda_outages() -> TestResult:
    """Test IODA outage events detection."""
    from src.mcp_server.tools.cyber import get_ioda_outages
    
    start = datetime.now()
    try:
        # Query global outages (more likely to have data)
        result = await get_ioda_outages(
            entity_type="country",
            entity_code=None,  # Global
            days_back=7,
            limit=10
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="IODA Outages (get_outages)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        outage_count = result.get("outage_count", 0)
        
        return TestResult(
            tool_name="IODA Outages (get_outages)",
            status=TestStatus.SUCCESS if outage_count > 0 else TestStatus.EMPTY,
            message=f"Found {outage_count} outage events" if outage_count > 0 else "No outages detected (API working)",
            data_count=outage_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="IODA Outages (get_outages)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_telegram_search() -> TestResult:
    """Test Telegram channel search."""
    from src.mcp_server.tools.telegram import search_telegram_channels
    
    # Check if credentials are configured
    if not settings.has_telegram_credentials:
        return TestResult(
            tool_name="Telegram Search (search_telegram)",
            status=TestStatus.SKIPPED,
            message="TELEGRAM_API_ID/HASH not configured"
        )
    
    start = datetime.now()
    try:
        result = await search_telegram_channels(
            keywords="Ukraine",
            category="news",
            hours_back=24,
            max_messages=5
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="Telegram Search (search_telegram)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        message_count = result.get("message_count", 0)
        return TestResult(
            tool_name="Telegram Search (search_telegram)",
            status=TestStatus.SUCCESS if message_count > 0 else TestStatus.EMPTY,
            message=f"Found {message_count} messages" if message_count > 0 else "No messages found (API working)",
            data_count=message_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="Telegram Search (search_telegram)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_telegram_channel_info() -> TestResult:
    """Test Telegram channel info retrieval."""
    from src.mcp_server.tools.telegram import get_telegram_channel_info
    
    # Check if credentials are configured
    if not settings.has_telegram_credentials:
        return TestResult(
            tool_name="Telegram Channel Info (get_channel_info)",
            status=TestStatus.SKIPPED,
            message="TELEGRAM_API_ID/HASH not configured"
        )
    
    start = datetime.now()
    try:
        result = await get_telegram_channel_info(
            channel_username="meduzalive"
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="Telegram Channel Info (get_channel_info)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        channel_info = result.get("channel_info", {})
        title = channel_info.get("title", "Unknown")
        
        return TestResult(
            tool_name="Telegram Channel Info (get_channel_info)",
            status=TestStatus.SUCCESS,
            message=f"Retrieved info for '{title}'",
            data_count=1,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="Telegram Channel Info (get_channel_info)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


def test_list_osint_channels() -> TestResult:
    """Test listing curated OSINT channels."""
    from src.mcp_server.tools.telegram import list_curated_channels
    
    start = datetime.now()
    try:
        result = list_curated_channels()
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        total_channels = result.get("total_channels", 0)
        categories = result.get("categories", {})
        
        return TestResult(
            tool_name="List OSINT Channels (list_osint_channels)",
            status=TestStatus.SUCCESS if total_channels > 0 else TestStatus.EMPTY,
            message=f"Listed {total_channels} channels in {len(categories)} categories",
            data_count=total_channels,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="List OSINT Channels (list_osint_channels)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_otx_ioc_lookup() -> TestResult:
    """Test AlienVault OTX IoC lookup."""
    from src.mcp_server.tools.threat_intel import lookup_indicator
    
    # Check if API key is configured
    if not settings.has_otx_key:
        return TestResult(
            tool_name="OTX IoC Lookup (check_ioc)",
            status=TestStatus.SKIPPED,
            message="OTX_API_KEY not configured"
        )
    
    start = datetime.now()
    try:
        # Test with Google's DNS (should be clean)
        result = await lookup_indicator(
            indicator="8.8.8.8",
            indicator_type="IPv4"
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="OTX IoC Lookup (check_ioc)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        indicator_info = result.get("indicator_info", {})
        pulse_count = indicator_info.get("pulse_count", 0)
        
        return TestResult(
            tool_name="OTX IoC Lookup (check_ioc)",
            status=TestStatus.SUCCESS,
            message=f"Found {pulse_count} pulses for indicator",
            data_count=pulse_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="OTX IoC Lookup (check_ioc)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


async def test_otx_threat_search() -> TestResult:
    """Test AlienVault OTX threat pulse search."""
    from src.mcp_server.tools.threat_intel import search_pulses
    
    # Check if API key is configured
    if not settings.has_otx_key:
        return TestResult(
            tool_name="OTX Threat Search (search_threats)",
            status=TestStatus.SKIPPED,
            message="OTX_API_KEY not configured"
        )
    
    start = datetime.now()
    try:
        result = await search_pulses(
            query="ransomware",
            limit=5
        )
        elapsed = (datetime.now() - start).total_seconds() * 1000
        
        if result.get("status") == "error":
            return TestResult(
                tool_name="OTX Threat Search (search_threats)",
                status=TestStatus.ERROR,
                message=result.get("error_message", "Unknown error"),
                response_time_ms=elapsed,
                raw_response=result
            )
        
        pulse_count = result.get("pulse_count", 0)
        
        return TestResult(
            tool_name="OTX Threat Search (search_threats)",
            status=TestStatus.SUCCESS if pulse_count > 0 else TestStatus.EMPTY,
            message=f"Found {pulse_count} threat pulses" if pulse_count > 0 else "No pulses found (API working)",
            data_count=pulse_count,
            response_time_ms=elapsed,
            raw_response=result
        )
    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds() * 1000
        return TestResult(
            tool_name="OTX Threat Search (search_threats)",
            status=TestStatus.ERROR,
            message=f"Exception: {type(e).__name__}: {str(e)}",
            response_time_ms=elapsed
        )


# =============================================================================
# Main Test Runner
# =============================================================================


TOOL_MAP = {
    "gdelt": test_gdelt_news,
    "nasa": test_nasa_firms,
    "ioda": test_ioda_connectivity,
    "ioda_outages": test_ioda_outages,
    "cloudflare": test_cloudflare_radar,
    "telegram_search": test_telegram_search,
    "telegram_info": test_telegram_channel_info,
    "telegram_list": test_list_osint_channels,
    "otx_ioc": test_otx_ioc_lookup,
    "otx_search": test_otx_threat_search,
}

# Tools that require API keys
API_REQUIRED_TOOLS = {"nasa", "telegram_search", "telegram_info", "otx_ioc", "otx_search"}


def print_config_status():
    """Print current configuration status."""
    table = Table(title="🔧 Configuration Status", show_header=True)
    table.add_column("Configuration", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Note", style="dim")
    
    # NASA FIRMS
    nasa_status = "✅ Configured" if settings.has_nasa_key else "❌ Not Set"
    table.add_row(
        "NASA_FIRMS_API_KEY",
        nasa_status,
        "Required for thermal anomaly detection"
    )
    
    # Telegram
    telegram_status = "✅ Configured" if settings.has_telegram_credentials else "❌ Not Set"
    table.add_row(
        "TELEGRAM_API_ID/HASH",
        telegram_status,
        "Required for Telegram monitoring"
    )
    
    # AlienVault OTX
    otx_status = "✅ Configured" if settings.has_otx_key else "❌ Not Set"
    table.add_row(
        "OTX_API_KEY",
        otx_status,
        "Required for AlienVault OTX threat intel"
    )
    
    # Cloudflare (optional)
    cf_status = "✅ Configured" if settings.cloudflare_api_token else "⚪ Optional"
    table.add_row(
        "CLOUDFLARE_API_TOKEN",
        cf_status,
        "Optional for enhanced metrics"
    )
    
    console.print()
    console.print(table)
    console.print()


async def run_tests(
    tools: list[str] | None = None,
    skip_api_required: bool = False,
    verbose: bool = False
) -> list[TestResult]:
    """Run all or selected tool tests."""
    
    # Determine which tools to test
    if tools:
        test_tools = {k: v for k, v in TOOL_MAP.items() if k in tools}
    else:
        test_tools = TOOL_MAP.copy()
    
    # Skip API-required tools if requested
    if skip_api_required:
        test_tools = {k: v for k, v in test_tools.items() if k not in API_REQUIRED_TOOLS}
    
    results: list[TestResult] = []
    
    console.print(Panel.fit(
        "[bold blue]🧪 MCP Tools Test Suite[/bold blue]\n"
        f"Testing {len(test_tools)} tools...",
        border_style="blue"
    ))
    console.print()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for tool_key, test_func in test_tools.items():
            task = progress.add_task(f"Testing {tool_key}...", total=1)
            
            # Run test (handle both async and sync functions)
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            results.append(result)
            progress.update(task, completed=1)
    
    return results


def print_results(results: list[TestResult], verbose: bool = False):
    """Print test results in a formatted table."""
    
    # Results table
    table = Table(title="📊 Test Results", show_header=True)
    table.add_column("Tool", style="cyan", min_width=40)
    table.add_column("Status", justify="center", min_width=12)
    table.add_column("Message", min_width=30)
    table.add_column("Time (ms)", justify="right", style="dim")
    
    for result in results:
        status_style = {
            TestStatus.SUCCESS: "green",
            TestStatus.ERROR: "red",
            TestStatus.EMPTY: "yellow",
            TestStatus.SKIPPED: "dim",
            TestStatus.STUB: "blue",
        }.get(result.status, "white")
        
        table.add_row(
            result.tool_name,
            f"[{status_style}]{result.status.value}[/{status_style}]",
            result.message[:50] + "..." if len(result.message) > 50 else result.message,
            f"{result.response_time_ms:.0f}" if result.response_time_ms > 0 else "-"
        )
    
    console.print()
    console.print(table)
    
    # Summary
    summary = {
        TestStatus.SUCCESS: 0,
        TestStatus.ERROR: 0,
        TestStatus.EMPTY: 0,
        TestStatus.SKIPPED: 0,
        TestStatus.STUB: 0,
    }
    for result in results:
        summary[result.status] += 1
    
    console.print()
    console.print(Panel.fit(
        f"[green]✅ Success: {summary[TestStatus.SUCCESS]}[/green]  "
        f"[red]❌ Error: {summary[TestStatus.ERROR]}[/red]  "
        f"[yellow]⚠️ Empty: {summary[TestStatus.EMPTY]}[/yellow]  "
        f"[dim]⏭️ Skipped: {summary[TestStatus.SKIPPED]}[/dim]  "
        f"[blue]📝 Stub: {summary[TestStatus.STUB]}[/blue]",
        title="Summary",
        border_style="cyan"
    ))
    
    # Print verbose details for errors
    if verbose:
        error_results = [r for r in results if r.status == TestStatus.ERROR]
        if error_results:
            console.print()
            console.print("[bold red]Error Details:[/bold red]")
            for result in error_results:
                console.print(f"\n[cyan]{result.tool_name}[/cyan]")
                console.print(f"  Message: {result.message}")
                if result.raw_response:
                    console.print(f"  Raw response: {result.raw_response}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test MCP tools for errors and empty responses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available tools:
  gdelt           - GDELT news search
  nasa            - NASA FIRMS thermal anomalies (requires API key)
  ioda            - IODA internet connectivity
  ioda_outages    - IODA detected outage events
  cloudflare      - Cloudflare Radar traffic metrics
  telegram_search - Telegram channel search (requires API credentials)
  telegram_info   - Telegram channel info (requires API credentials)
  telegram_list   - List curated OSINT channels
  otx_ioc         - AlienVault OTX IoC lookup (requires API key)
  otx_search      - AlienVault OTX threat pulse search (requires API key)

Examples:
  # Test all tools
  uv run python scripts/test_mcp_tools.py

  # Test specific tools
  uv run python scripts/test_mcp_tools.py --tools gdelt,ioda,cloudflare

  # Test OTX tools
  uv run python scripts/test_mcp_tools.py --tools otx_ioc,otx_search

  # Skip tools that require API keys
  uv run python scripts/test_mcp_tools.py --skip-api-required

  # Verbose output with error details
  uv run python scripts/test_mcp_tools.py -v
        """
    )
    parser.add_argument(
        "--tools", "-t",
        type=str,
        help="Comma-separated list of tools to test (default: all)"
    )
    parser.add_argument(
        "--skip-api-required", "-s",
        action="store_true",
        help="Skip tools that require API keys"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed error information"
    )
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse tools list
    tools = None
    if args.tools:
        tools = [t.strip() for t in args.tools.split(",")]
        invalid_tools = set(tools) - set(TOOL_MAP.keys())
        if invalid_tools:
            console.print(f"[red]Unknown tools: {invalid_tools}[/red]")
            console.print(f"Available: {list(TOOL_MAP.keys())}")
            sys.exit(1)
    
    # Print configuration status
    print_config_status()
    
    # Run tests
    results = await run_tests(
        tools=tools,
        skip_api_required=args.skip_api_required,
        verbose=args.verbose
    )
    
    # Print results
    print_results(results, verbose=args.verbose)
    
    # Exit with error code if any tests failed
    error_count = sum(1 for r in results if r.status == TestStatus.ERROR)
    if error_count > 0:
        console.print()
        console.print(f"[red]⚠️ {error_count} tool(s) returned errors![/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
