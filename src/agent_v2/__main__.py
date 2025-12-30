"""
Project Overwatch - Agent v2 CLI.

Usage:
    python -m src.agent_v2 "Analyze military activity in Ukraine"
    python -m src.agent_v2 --thinking-model openai:gpt-4 "Complex analysis"
"""

import argparse
import asyncio
import json
import sys

from rich.console import Console
from rich.markdown import Markdown

from src.shared.config import settings
from src.shared.logger import get_logger, log_startup_banner

logger = get_logger()
console = Console()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Build default model specs from settings (Ollama only)
    structured_default = settings.structured_llm_model
    thinking_default = settings.thinking_llm_model
    
    parser = argparse.ArgumentParser(
        description="Project Overwatch - Agent v2 (PydanticAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default models
  python -m src.agent_v2 "Analyze military activity in Ukraine"

  # Custom models (Ollama)
  python -m src.agent_v2 --structured-model llama3.2 "Quick analysis"
  python -m src.agent_v2 --thinking-model qwen2.5:7b "Deep analysis"

  # Output as JSON
  python -m src.agent_v2 --json "Search for news about protests"
        """,
    )
    parser.add_argument(
        "task",
        nargs="?",
        default=None,
        help="Research task or question to investigate",
    )
    parser.add_argument(
        "--structured-model",
        default=structured_default,
        help=f"Model for structured outputs (default from .env: {structured_default})",
    )
    parser.add_argument(
        "--thinking-model",
        default=thinking_default,
        help=f"Model for complex reasoning (default from .env: {thinking_default})",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="MCP server SSE endpoint URL",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    return parser.parse_args()


def format_sitrep_markdown(sitrep) -> str:
    """Convert SITREP output to markdown string."""
    sections = []
    
    # Header
    sections.append(f"# SITREP Intelligence Report")
    sections.append(f"**Classification:** {sitrep.classification}")
    sections.append(f"**Query:** {sitrep.query_summary}")
    sections.append(f"**Sources Used:** {', '.join(sitrep.intelligence_sources_used)}")
    sections.append("")
    
    # Section I
    s1 = sitrep.section_i
    sections.append("## I. Executive Intelligence Summary")
    sections.append(f"\n{s1.direct_response}\n")
    sections.append("### Key Highlights")
    for h in s1.key_highlights:
        sections.append(f"- {h}")
    sections.append(f"\n**Confidence:** {s1.overall_confidence_percent}% | **Quality:** {s1.intelligence_quality}")
    sections.append("")
    
    # Section II
    s2 = sitrep.section_ii
    sections.append("## II. Detailed Analysis")
    for topic in s2.topics:
        sections.append(f"\n### {topic.title}")
        sections.append(f"\n{topic.current_situation}\n")
        if topic.key_developments:
            sections.append("**Key Developments:**")
            for dev in topic.key_developments:
                sections.append(f"- {dev}")
        if topic.probability_assessments:
            sections.append("\n**Probability Assessments:**")
            for pa in topic.probability_assessments:
                sections.append(f"- {pa.scenario}: {pa.probability_percent}% ({pa.timeframe})")
    if s2.cross_topic_connections:
        sections.append(f"\n**Cross-Topic Connections:** {s2.cross_topic_connections}")
    sections.append("")
    
    # Section III
    s3 = sitrep.section_iii
    sections.append("## III. Supporting Intelligence Analysis")
    sections.append(f"\n**Satellite Intel:** {s3.satellite_intel}")
    sections.append(f"\n**News Intel:** {s3.news_intel}")
    sections.append(f"\n**Cyber Intel:** {s3.cyber_intel}")
    sections.append(f"\n**Social Intel:** {s3.social_intel}")
    if s3.cross_source_validation:
        sections.append(f"\n**Cross-Source Validation:** {s3.cross_source_validation}")
    if s3.intelligence_gaps:
        sections.append("\n**Intelligence Gaps:**")
        for gap in s3.intelligence_gaps:
            sections.append(f"- {gap}")
    sections.append("")
    
    # Section IV
    s4 = sitrep.section_iv
    sections.append("## IV. Actionable Intelligence")
    if s4.immediate_actions:
        sections.append("\n**Immediate Actions:**")
        for action in s4.immediate_actions:
            sections.append(f"- {action}")
    if s4.monitoring_indicators:
        sections.append("\n**Monitoring Indicators:**")
        for ind in s4.monitoring_indicators:
            sections.append(f"- {ind}")
    sections.append("")
    
    # Section V
    s5 = sitrep.section_v
    sections.append("## V. Intelligence Assessment Metadata")
    if s5.source_reliability_matrix:
        sections.append("\n**Source Reliability Matrix:**")
        for src in s5.source_reliability_matrix:
            grade = src.grade or f"{src.reliability}-{src.credibility}"
            sections.append(f"- {src.source_name}: {grade}")
    if s5.analytical_confidence:
        sections.append(f"\n**Analytical Confidence:** {s5.analytical_confidence}")
    if s5.data_freshness:
        sections.append(f"\n**Data Freshness:** {s5.data_freshness}")
    sections.append("")
    
    # Section VI
    s6 = sitrep.section_vi
    sections.append("## VI. Forward Intelligence Requirements")
    if s6.priority_collection:
        sections.append("\n**Priority Collection:**")
        for pc in s6.priority_collection:
            sections.append(f"- {pc}")
    if s6.early_warning_triggers:
        sections.append("\n**Early Warning Triggers:**")
        for ewt in s6.early_warning_triggers:
            sections.append(f"- {ewt}")
    
    return "\n".join(sections)


async def main() -> None:
    """Run the Agent v2."""
    args = parse_args()
    
    if not args.task:
        logger.error("Please provide a research task. Example:")
        logger.markdown('  `python -m src.agent_v2 "Analyze military activity in Ukraine"`')
        logger.info("Use --help for more options")
        sys.exit(1)
    
    log_startup_banner("agent_v2")
    
    logger.info(f"Starting Agent v2 research: [bold]{args.task}[/bold]")
    logger.info(f"Structured Model: [bold]{args.structured_model}[/bold]")
    logger.info(f"Thinking Model: [bold]{args.thinking_model}[/bold]")
    logger.info(f"Max iterations: [bold]{args.max_iter}[/bold]")
    
    try:
        from src.agent_v2.agent import ResearchAgent, DualLLMConfig
        
        config = DualLLMConfig(
            structured_model=args.structured_model,
            thinking_model=args.thinking_model,
        )
        
        agent = ResearchAgent(
            mcp_url=args.server,
            llm_config=config,
        )
        
        sitrep = await agent.research(
            task=args.task,
            max_iterations=args.max_iter,
        )
        
        if args.json:
            print(json.dumps(sitrep.model_dump(), indent=2, default=str))
        else:
            # Print as formatted markdown
            md_content = format_sitrep_markdown(sitrep)
            console.print()
            console.print(Markdown(md_content))
            console.print()
            
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
        import traceback
        traceback.print_exc()
        if args.json:
            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
