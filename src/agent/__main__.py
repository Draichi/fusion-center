"""
Project Overwatch - Deep Research Agent CLI.

Usage:
    python -m src.agent "Analyze military activity in Ukraine"
    python -m src.agent --provider gemini "Check internet status in Iran"
    python -m src.agent --provider ollama --model llama3.2 "Local analysis"
    python -m src.agent --provider docker "Local analysis with Docker Model Runner"
"""

import argparse
import asyncio
import json
import sys

from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Project Overwatch - Deep Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Providers:
  gemini    - Google Gemini (requires GOOGLE_API_KEY)
  grok      - xAI Grok (requires XAI_API_KEY)
  ollama    - Local Ollama (no key required)
  docker    - Docker Model Runner (no key required)

Examples:
  # Use default provider (from .env)
  python -m src.agent "Analyze military activity in Ukraine"

  # Use Gemini
  python -m src.agent --provider gemini "Monitor Iran situation"

  # Use local Ollama
  python -m src.agent --provider ollama --model llama3.2 "Analyze conflict"

  # Use Docker Model Runner
  python -m src.agent --provider docker "Local analysis with Docker"

  # Use Grok
  python -m src.agent --provider grok "Deep dive into Gaza conflict"

  # Output as JSON
  python -m src.agent --json "Search for news about protests"
        """,
    )
    parser.add_argument(
        "task",
        nargs="?",  # Makes task optional
        default=None,
        help="Research task or question to investigate",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "grok", "ollama", "docker"],
        default=None,
        help=f"LLM provider (default: {settings.agent_provider})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name override",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="MCP server SSE endpoint URL",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum research iterations (default: 10)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Additional context as JSON string",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit",
    )
    return parser.parse_args()


def print_providers() -> None:
    """Print available LLM providers."""
    logger.divider("Available LLM Providers")
    
    providers = [
        ("gemini", settings.has_google_key, settings.agent_model_gemini, "GOOGLE_API_KEY"),
        ("grok", settings.has_xai_key, settings.agent_model_grok, "XAI_API_KEY"),
        ("ollama", True, settings.agent_model_ollama, "(local - no key)"),
        ("docker", True, settings.agent_model_docker, "(Docker Model Runner - no key)"),
    ]
    
    for name, configured, model, key_name in providers:
        status = "✅" if configured else "❌"
        logger.markdown(f"**{name}** {status}")
        logger.markdown(f"  - Model: `{model}`")
        logger.markdown(f"  - Key: `{key_name}`")
        logger.markdown("")
    
    logger.info(f"Default provider: [bold]{settings.agent_provider}[/bold]")


def print_report(state: dict) -> None:
    """Print a formatted research report."""
    logger.divider("Research Complete")
    
    # Executive Summary
    if state.get("executive_summary"):
        logger.panel(
            state["executive_summary"],
            title="📋 Executive Summary",
            style="green",
        )
    
    # Statistics
    stats = f"""
**Iterations:** {state.get('iteration', 0)}
**Findings Collected:** {len(state.get('findings', []))}
**Queries Executed:** {len(state.get('executed_queries', []))}
**Correlations Found:** {len(state.get('correlations', []))}
**Key Insights:** {len(state.get('key_insights', []))}
"""
    logger.markdown(stats)
    
    # Key Insights
    if state.get("key_insights"):
        logger.divider("Key Insights")
        for i, insight in enumerate(state["key_insights"], 1):
            if insight:
                logger.markdown(f"**{i}.** {insight}")
    
    # Correlations
    if state.get("correlations"):
        logger.divider("Correlations")
        for corr in state["correlations"]:
            logger.markdown(f"""
- **Type:** {corr.get('correlation_type', 'unknown')}
- **Description:** {corr.get('description', 'N/A')}
- **Confidence:** {corr.get('confidence', 'unknown')}
""")
    
    # Detailed Report
    if state.get("detailed_report"):
        logger.divider("Detailed Report")
        logger.markdown(state["detailed_report"])
    
    # Recommendations
    if state.get("recommendations"):
        logger.divider("Recommendations")
        for i, rec in enumerate(state["recommendations"], 1):
            logger.markdown(f"**{i}.** {rec}")
    
    # Confidence Assessment
    if state.get("confidence_assessment"):
        logger.panel(
            state["confidence_assessment"],
            title="🎯 Confidence Assessment",
            style="cyan",
        )
    
    # Uncertainties
    if state.get("uncertainties"):
        logger.divider("Uncertainties & Gaps")
        for unc in state["uncertainties"]:
            logger.markdown(f"- {unc}")
    
    # Error if any
    if state.get("error"):
        logger.error(f"Research encountered an error: {state['error']}")


async def main() -> None:
    """Run the Deep Research Agent."""
    args = parse_args()
    
    # Handle --list-providers
    if args.list_providers:
        print_providers()
        return
    
    # Check if task was provided
    if not args.task:
        logger.error("Please provide a research task. Example:")
        logger.markdown('  `python -m src.agent "Analyze military activity in Ukraine"`')
        logger.info("Use --help for more options")
        sys.exit(1)
    
    # Parse optional context
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            logger.error("Invalid JSON context provided")
            sys.exit(1)
    
    # Determine provider
    provider = args.provider or settings.agent_provider
    
    # Check if provider is available
    if provider == "gemini" and not settings.has_google_key:
        logger.error("GOOGLE_API_KEY not set. Get your key at: https://aistudio.google.com/app/apikey")
        sys.exit(1)
    elif provider == "grok" and not settings.has_xai_key:
        logger.error("XAI_API_KEY not set. Get your key at: https://console.x.ai/")
        sys.exit(1)
    # ollama and docker are always available (local)
    
    logger.info(f"Starting research: [bold]{args.task}[/bold]")
    logger.info(f"Provider: [bold]{provider}[/bold]")
    logger.info(f"Max iterations: [bold]{args.max_iter}[/bold]")
    
    try:
        from src.agent.graph import DeepResearchAgent
        
        agent = DeepResearchAgent(
            mcp_server_url=args.server,
            provider=provider,
            model=args.model,
            max_iterations=args.max_iter,
        )
        
        result = await agent.research(
            task=args.task,
            context=context,
        )
        
        if args.json:
            output = {
                k: v for k, v in result.items()
                if k != "messages"
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print_report(result)
            
    except ImportError as e:
        logger.error(f"Missing dependencies. Run: uv pip install -e '.[agent]'")
        logger.error(f"Details: {e}")
        sys.exit(1)
    except ConnectionError:
        logger.error(f"Could not connect to MCP server. Is it running?")
        logger.error(f"Start with: python -m src.mcp_server.server --transport sse")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Research failed: {e}")
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
