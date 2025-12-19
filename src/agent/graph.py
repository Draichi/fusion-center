"""
LangGraph Definition for the Deep Research Agent.

This module defines the graph structure that orchestrates the research process.
Supports multiple LLM providers: Gemini, Grok (xAI), Ollama, and Docker Model Runner.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from mcp import ClientSession
from mcp.client.sse import sse_client
from alive_progress import alive_bar

from src.agent.state import AgentState, ResearchPhase, create_initial_state
from src.shared.config import settings
from src.agent.nodes import (
    PlanningNode,
    DecompositionNode,
    HypothesisGenerationNode,
    HypothesisUpdateNode,
    AnalysisNode,
    ReflectionNode,
    CorrelationNode,
    VerificationNode,
    SynthesisNode,
    route_next_step,
    gather_intelligence,
)
from src.agent.tools import MCPToolExecutor, VALID_TOOL_NAMES
from src.shared.config import settings
from src.shared.logger import get_logger, log_startup_banner, log_agent_result, log_tools_table
from src.shared.output_writer import get_output_writer, reset_output_writer

logger = get_logger()


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """
    Get the appropriate LLM based on provider configuration.
    
    Supported providers:
    - gemini: Google Gemini (requires GOOGLE_API_KEY)
    - grok: xAI Grok (requires XAI_API_KEY)
    - ollama: Local Ollama (no key required)
    - docker: Docker Model Runner (no key required)
    
    Args:
        provider: Provider name override (default from settings)
        model: Model name override
        temperature: Temperature for generation
        
    Returns:
        Configured LLM instance
    """
    provider = provider or settings.agent_provider
    temp = temperature if temperature is not None else settings.agent_temperature
    
    if provider == "gemini":
        if not settings.has_google_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. "
                "Get your key at: https://aistudio.google.com/app/apikey"
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        model_name = model or settings.agent_model_gemini
        logger.info(f"Using Gemini: [bold]{model_name}[/bold]")
        
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temp,
            google_api_key=settings.google_api_key,
        )
    
    elif provider == "grok":
        if not settings.has_xai_key:
            raise ValueError(
                "XAI_API_KEY not set. "
                "Get your key at: https://console.x.ai/"
            )
        # xAI Grok uses OpenAI-compatible API
        # We use langchain-openai as a client only (not OpenAI service)
        from langchain_openai import ChatOpenAI
        
        model_name = model or settings.agent_model_grok
        logger.info(f"Using Grok (xAI): [bold]{model_name}[/bold]")
        
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            api_key=settings.xai_api_key,
            base_url="https://api.x.ai/v1",
        )
    
    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        
        model_name = model or settings.agent_model_ollama
        logger.info(f"Using Ollama (local): [bold]{model_name}[/bold]")
        
        return ChatOllama(
            model=model_name,
            temperature=temp,
            base_url=settings.ollama_base_url,
        )
    
    elif provider == "docker":
        # Docker Model Runner uses OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        
        model_name = model or settings.agent_model_docker
        logger.info(f"Using Docker Model Runner: [bold]{model_name}[/bold]")
        
        return ChatOpenAI(
            model=model_name,
            temperature=temp,
            api_key="docker-model-runner",  # Not required, but LangChain needs something
            base_url=settings.docker_model_base_url,
        )
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: gemini, grok, ollama, docker"
        )


def get_dual_llms() -> tuple[BaseChatModel, BaseChatModel]:
    """
    Get both thinking and structured LLMs for dual-LLM architecture.
    
    Returns:
        Tuple of (thinking_llm, structured_llm)
    """
    logger.info("Initializing dual-LLM architecture...")
    
    # Initialize thinking LLM
    thinking_llm = get_llm(
        provider=settings.thinking_llm_provider,
        model=settings.thinking_llm_model,
        temperature=settings.thinking_llm_temperature,
    )
    logger.info(f"  Thinking LLM: [bold]{settings.thinking_llm_provider}/{settings.thinking_llm_model}[/bold] (temp={settings.thinking_llm_temperature})")
    
    # Initialize structured LLM
    structured_llm = get_llm(
        provider=settings.structured_llm_provider,
        model=settings.structured_llm_model,
        temperature=settings.structured_llm_temperature,
    )
    logger.info(f"  Structured LLM: [bold]{settings.structured_llm_provider}/{settings.structured_llm_model}[/bold] (temp={settings.structured_llm_temperature})")
    
    return thinking_llm, structured_llm


def build_research_graph(
    thinking_llm: BaseChatModel,
    structured_llm: BaseChatModel,
    checkpointer: MemorySaver | None = None,
    enable_multi_step_reasoning: bool = True,
) -> StateGraph:
    """
    Build the LangGraph for deep research with multi-step reasoning and dual-LLM architecture.
    
    The graph follows this enhanced flow with multi-step reasoning:
    
    START → plan → decompose → hypothesize → gather ⟷ analyze → reflect → correlate → verify → synthesize → END
                                               ↑___________|        ↑_______|
    
    Multi-step reasoning adds:
    - Task decomposition for complex problems
    - Hypothesis generation and testing
    - Self-reflection and bias checking
    - Verification before synthesis
    
    Dual-LLM architecture:
    - Thinking LLM: Complex reasoning (reflection, verification, synthesis)
    - Structured LLM: Structured outputs (planning, analysis, correlation)
    
    Args:
        thinking_llm: Language model for complex reasoning tasks
        structured_llm: Language model for structured JSON outputs
        checkpointer: Optional checkpointer for persistence
        enable_multi_step_reasoning: Whether to use enhanced reasoning (default True)
        
    Returns:
        Compiled StateGraph
    """
    
    # Create the graph with our state schema
    graph = StateGraph(AgentState)
    
    # === Initialize Node Strategies with Dual LLMs ===
    planning_node = PlanningNode(thinking_llm, structured_llm)
    decomposition_node = DecompositionNode(thinking_llm, structured_llm)
    hypothesis_node = HypothesisGenerationNode(thinking_llm, structured_llm)
    analysis_node = AnalysisNode(thinking_llm, structured_llm)
    reflection_node = ReflectionNode(thinking_llm, structured_llm)
    correlation_node = CorrelationNode(thinking_llm, structured_llm)
    verification_node = VerificationNode(thinking_llm, structured_llm)
    synthesis_node = SynthesisNode(thinking_llm, structured_llm)
    
    # Gather node - placeholder that will be replaced per-run (needs tool_executor)
    async def gather_node(state: AgentState) -> dict[str, Any]:
        return {
            "current_phase": ResearchPhase.ANALYZING.value,
            "iteration": state["iteration"] + 1,
        }
    
    # === Add Nodes ===
    graph.add_node("plan", planning_node.execute)
    graph.add_node("decompose", decomposition_node.execute)
    graph.add_node("hypothesize", hypothesis_node.execute)
    graph.add_node("gather", gather_node)
    graph.add_node("analyze", analysis_node.execute)
    graph.add_node("reflect", reflection_node.execute)
    graph.add_node("correlate", correlation_node.execute)
    graph.add_node("verify", verification_node.execute)
    graph.add_node("synthesize", synthesis_node.execute)
    
    # === Add Edges ===
    
    # Start with planning
    graph.set_entry_point("plan")
    
    # All routing options for multi-step reasoning
    full_routing = {
        "decompose": "decompose",
        "hypothesize": "hypothesize",
        "gather": "gather",
        "analyze": "analyze",
        "reflect": "reflect",
        "correlate": "correlate",
        "verify": "verify",
        "synthesize": "synthesize",
        "end": END,
    }
    
    # After planning, can go to decomposition, hypothesize, or gathering
    graph.add_conditional_edges(
        "plan",
        route_next_step,
        full_routing,
    )
    
    # After decomposition, go to hypothesis generation
    graph.add_conditional_edges(
        "decompose",
        route_next_step,
        full_routing,
    )
    
    # After hypothesis generation, go to gathering
    graph.add_conditional_edges(
        "hypothesize",
        route_next_step,
        full_routing,
    )
    
    # After gathering, route based on state
    graph.add_conditional_edges(
        "gather",
        route_next_step,
        full_routing,
    )
    
    # After analysis, route to reflection or back to gathering
    graph.add_conditional_edges(
        "analyze",
        route_next_step,
        full_routing,
    )
    
    # After reflection, route to correlation or back to gathering
    graph.add_conditional_edges(
        "reflect",
        route_next_step,
        full_routing,
    )
    
    # After correlation, go to verification
    graph.add_conditional_edges(
        "correlate",
        route_next_step,
        full_routing,
    )
    
    # After verification, go to synthesis or back to reflection
    graph.add_conditional_edges(
        "verify",
        route_next_step,
        full_routing,
    )
    
    # After synthesis, end
    graph.add_edge("synthesize", END)
    
    # Compile the graph
    return graph.compile(checkpointer=checkpointer)


class DeepResearchAgent:
    """
    Deep Research Agent using LangGraph.
    
    Supports multiple LLM providers:
    - Gemini (Google)
    - Grok (xAI)
    - Ollama (local)
    - Docker Model Runner (local)
    """
    
    def __init__(
        self,
        mcp_server_url: str | None = None,
        provider: str | None = None,  # Deprecated: ignored, uses dual-LLM from .env
        model: str | None = None,  # Deprecated: ignored, uses dual-LLM from .env
        temperature: float | None = None,  # Deprecated: ignored, uses dual-LLM from .env
        max_iterations: int | None = None,
        use_checkpointer: bool = True,
    ):
        """
        Initialize the Deep Research Agent with dual-LLM architecture.
        
        The agent always uses two separate LLMs configured in .env:
        - THINKING_LLM_* for complex reasoning (reflection, verification, synthesis)
        - STRUCTURED_LLM_* for structured JSON outputs (planning, analysis, correlation)
        
        Args:
            mcp_server_url: URL of the MCP server SSE endpoint
            provider: DEPRECATED - ignored, dual-LLM config from .env is used
            model: DEPRECATED - ignored, dual-LLM config from .env is used
            temperature: DEPRECATED - ignored, dual-LLM config from .env is used
            max_iterations: Maximum research iterations
            use_checkpointer: Whether to enable state persistence
        """
        self.mcp_server_url = mcp_server_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.max_iterations = max_iterations if max_iterations is not None else settings.agent_max_iterations
        
        # Always use dual-LLM architecture (ignore legacy parameters if passed)
        if provider or model or temperature is not None:
            logger.debug(f"Ignoring legacy parameters (provider={provider}, model={model}, temperature={temperature})")
            logger.debug("Using dual-LLM configuration from .env instead")
        
        # Initialize dual LLMs from .env configuration
        self.thinking_llm, self.structured_llm = get_dual_llms()
        
        # Initialize checkpointer for persistence
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        # Build the graph with dual LLMs
        self.graph = build_research_graph(self.thinking_llm, self.structured_llm, self.checkpointer)
        
        logger.info(f"Deep Research Agent initialized with dual-LLM architecture")
        logger.info(f"MCP Server: [bold]{self.mcp_server_url}[/bold]")
    
    async def research(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        thread_id: str = "default",
    ) -> dict[str, Any]:
        """
        Run a deep research investigation.
        
        Args:
            task: The research task/question
            context: Additional context (regions, coordinates, etc.)
            thread_id: Thread ID for state persistence
            
        Returns:
            Final research state with report
        """
        log_startup_banner("agent")
        logger.agent(f"Starting deep research: [bold]{task}[/bold]")
        
        # Initialize output writer for this session (with query for folder naming)
        reset_output_writer()
        output_writer = get_output_writer(query=task)
        
        # Create initial state
        initial_state = create_initial_state(
            task=task,
            context=context,
            max_iterations=self.max_iterations,
        )
        
        # Config for the run
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Connect to MCP server and run the research
            async with sse_client(self.mcp_server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize MCP connection
                    await session.initialize()
                    logger.success("Connected to MCP server")
                    
                    # List available tools from MCP server
                    try:
                        tools_result = await session.list_tools()
                        available_tools = []
                        tool_categories = {
                            "search_news": "📰 News",
                            "fetch_rss_news": "📰 News",
                            "detect_thermal_anomalies": "🛰️ Satellite",
                            "check_connectivity": "🌐 Cyber",
                            "get_outages": "🌐 Cyber",
                            "check_traffic_metrics": "🌐 Cyber",
                            "search_telegram": "📱 Telegram",
                            "get_channel_info": "📱 Telegram",
                            "list_osint_channels": "📱 Telegram",
                            "check_ioc": "🔍 Threat Intel",
                            "get_threat_pulse": "🔍 Threat Intel",
                            "search_threats": "🔍 Threat Intel",
                        }
                        
                        for tool in tools_result.tools:
                            category = tool_categories.get(tool.name, "🔧 Other")
                            description = tool.description or "No description"
                            # Truncate long descriptions
                            if len(description) > 60:
                                description = description[:57] + "..."
                            available_tools.append((tool.name, category, description))
                        
                    except Exception as e:
                        logger.warning(f"Could not list tools from MCP server: {e}")
                        logger.info(f"Using {len(VALID_TOOL_NAMES)} tools from VALID_TOOL_NAMES")
                    
                    # Create tool executor with active session
                    tool_executor = MCPToolExecutor(session)
                    
                    # Run the graph with MCP integration
                    final_state = await self._run_with_mcp(
                        initial_state,
                        tool_executor,
                        config,
                    )
                    
                    log_agent_result(
                        success=final_state.get("error") is None,
                        summary=final_state.get("executive_summary", "Research completed"),
                    )
                    
                    # Save output files (report.md and reasoning.log)
                    if settings.save_report or settings.save_reasoning_log:
                        output_paths = output_writer.finalize(final_state)
                        logger.success(f"Report saved to: [bold]{output_paths['report']}[/bold]")
                        final_state["output_paths"] = output_paths
                        # Add report_content to state for terminal display
                        final_state["report_content"] = output_paths.get("report_content", "")
                    
                    return final_state
                    
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                **initial_state,
                "error": str(e),
                "current_phase": ResearchPhase.COMPLETE.value,
            }
    
    async def _execute_node_with_error_check(self, node, state, node_name):
        """Helper to execute node and check for errors."""
        try:
            updates = await node.execute(state)
            
            # Check if updates is None
            if updates is None:
                logger.error(f"❌ Node {node_name} returned None instead of dict")
                error_state = {
                    **state,
                    "error": f"{node_name} returned None",
                    "current_phase": ResearchPhase.COMPLETE.value,
                    "iteration": state["iteration"] + 1,
                }
                return error_state, True
            
            # Check if node returned an error
            if "error" in updates:
                logger.error(f"❌ Error in {node_name} phase: {updates['error']}")
                # Force completion with error
                return {**state, **updates, "current_phase": ResearchPhase.COMPLETE.value}, True
            
            # Get next phase
            next_phase = node.get_next_phase(state, updates)
            updates["current_phase"] = next_phase
            
            return {**state, **updates}, False
            
        except Exception as e:
            logger.error(f"❌ Unhandled exception in {node_name}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            error_state = {
                **state,
                "error": f"{type(e).__name__}: {str(e)}",
                "current_phase": ResearchPhase.COMPLETE.value,
                "iteration": state["iteration"] + 1,
            }
            return error_state, True

    async def _run_with_mcp(
        self,
        initial_state: AgentState,
        tool_executor: MCPToolExecutor,
        config: dict[str, Any],
    ) -> AgentState:
        """
        Run the research graph with MCP tool execution.
        
        Supports multi-step reasoning phases:
        - DECOMPOSING: Break complex tasks into sub-tasks
        - HYPOTHESIZING: Generate and test hypotheses
        - REFLECTING: Self-critique and bias checking
        - VERIFYING: Verify conclusions before synthesis
        """
        state = initial_state
        
        # Initialize node strategies with dual LLMs
        planning_node = PlanningNode(self.thinking_llm, self.structured_llm)
        decomposition_node = DecompositionNode(self.thinking_llm, self.structured_llm)
        hypothesis_node = HypothesisGenerationNode(self.thinking_llm, self.structured_llm)
        hypothesis_update_node = HypothesisUpdateNode(self.thinking_llm, self.structured_llm)
        analysis_node = AnalysisNode(self.thinking_llm, self.structured_llm)
        reflection_node = ReflectionNode(self.thinking_llm, self.structured_llm)
        correlation_node = CorrelationNode(self.thinking_llm, self.structured_llm)
        verification_node = VerificationNode(self.thinking_llm, self.structured_llm)
        synthesis_node = SynthesisNode(self.thinking_llm, self.structured_llm)
        
        total_iterations = state["max_iterations"]
        with alive_bar(total_iterations, manual=True, title="Deep Researching", bar="smooth", spinner=None, enrich_print=False) as bar:
            logger.set_progress_bar(bar)
            try:
                while state["current_phase"] != ResearchPhase.COMPLETE.value:
                    # Update bar with current phase
                    current_phase_display = state["current_phase"].replace("_", " ").title()
                    bar.text(f"Phase: {current_phase_display}")
                    
                    if total_iterations > 0:
                        bar(min(1.0, state["iteration"] / total_iterations))
            
                    # Check iteration limit
                    if state["iteration"] >= state["max_iterations"]:
                        logger.warning("Max iterations reached, forcing synthesis")
                        state["current_phase"] = ResearchPhase.SYNTHESIZING.value
                    
                    phase = state["current_phase"]
                    
                    if phase == ResearchPhase.PLANNING.value:
                        state, has_error = await self._execute_node_with_error_check(planning_node, state, "planning")
                        if has_error:
                            break
                    
                    # Multi-step Reasoning: Task Decomposition
                    elif phase == ResearchPhase.DECOMPOSING.value:
                        state, has_error = await self._execute_node_with_error_check(decomposition_node, state, "decomposition")
                        if has_error:
                            break
                    
                    # Multi-step Reasoning: Hypothesis Generation
                    elif phase == ResearchPhase.HYPOTHESIZING.value:
                        state, has_error = await self._execute_node_with_error_check(hypothesis_node, state, "hypothesis")
                        if has_error:
                            break
                        
                    elif phase == ResearchPhase.GATHERING.value:
                        updates = await gather_intelligence(state, tool_executor)
                        state = {**state, **updates}
                        # After gathering, update hypotheses if we have any
                        if state.get("hypotheses"):
                            state, has_error = await self._execute_node_with_error_check(hypothesis_update_node, state, "hypothesis_update")
                            if has_error:
                                break
                        
                    elif phase == ResearchPhase.ANALYZING.value:
                        updates = await analysis_node.execute(state)
                        updates["current_phase"] = analysis_node.get_next_phase(state, updates)
                        state = {**state, **updates}
                    
                    # Multi-step Reasoning: Self-Reflection
                    elif phase == ResearchPhase.REFLECTING.value:
                        updates = await reflection_node.execute(state)
                        updates["current_phase"] = reflection_node.get_next_phase(state, updates)
                        state = {**state, **updates}
                        
                    elif phase == ResearchPhase.CORRELATING.value:
                        updates = await correlation_node.execute(state)
                        updates["current_phase"] = correlation_node.get_next_phase(state, updates)
                        state = {**state, **updates}
                    
                    # Multi-step Reasoning: Verification
                    elif phase == ResearchPhase.VERIFYING.value:
                        updates = await verification_node.execute(state)
                        updates["current_phase"] = verification_node.get_next_phase(state, updates)
                        state = {**state, **updates}
                        
                    elif phase == ResearchPhase.SYNTHESIZING.value:
                        updates = await synthesis_node.execute(state)
                        updates["current_phase"] = synthesis_node.get_next_phase(state, updates)
                        state = {**state, **updates}
                        
                    else:
                        break
            
                # Ensure bar completes
                bar(1.0)
            
            finally:
                logger.set_progress_bar(None)
        
        return state
    
    async def continue_research(
        self,
        thread_id: str,
        additional_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Continue a previous research investigation.
        
        Args:
            thread_id: Thread ID of the previous research
            additional_context: New context to add
            
        Returns:
            Updated research state
        """
        if not self.checkpointer:
            raise ValueError("Checkpointer not enabled")
        
        config = {"configurable": {"thread_id": thread_id}}
        saved_state = self.checkpointer.get(config)
        
        if not saved_state:
            raise ValueError(f"No saved state found for thread: {thread_id}")
        
        if additional_context:
            saved_state["context"] = {
                **saved_state.get("context", {}),
                **additional_context,
            }
        
        return await self.research(
            task=saved_state["task"],
            context=saved_state["context"],
            thread_id=thread_id,
        )


async def run_deep_research(
    task: str,
    context: dict[str, Any] | None = None,
    provider: str | None = None,
    model: str | None = None,
    mcp_server_url: str | None = None,
) -> dict[str, Any]:
    """
    Convenience function to run deep research.
    
    Args:
        task: Research task/question
        context: Additional context
        provider: LLM provider (gemini, grok, ollama, docker)
        model: Model name override
        mcp_server_url: MCP server URL
        
    Returns:
        Research results
    """
    agent = DeepResearchAgent(
        mcp_server_url=mcp_server_url,
        provider=provider,
        model=model,
    )
    return await agent.research(task, context)
