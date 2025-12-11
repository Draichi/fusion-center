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

from src.agent.state import AgentState, ResearchPhase, create_initial_state
from src.shared.config import settings
from src.agent.nodes import (
    # Traditional nodes
    plan_research,
    gather_intelligence,
    analyze_findings,
    correlate_findings,
    synthesize_report,
    route_next_step,
    # Multi-step Reasoning nodes
    decompose_task,
    generate_hypotheses,
    update_hypotheses,
    reflect_on_analysis,
    verify_conclusions,
)
from src.agent.tools import MCPToolExecutor
from src.shared.config import settings
from src.shared.logger import get_logger, log_startup_banner, log_agent_result
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


def build_research_graph(
    llm: BaseChatModel,
    checkpointer: MemorySaver | None = None,
    enable_multi_step_reasoning: bool = True,
) -> StateGraph:
    """
    Build the LangGraph for deep research with multi-step reasoning.
    
    The graph follows this enhanced flow with multi-step reasoning:
    
    START → plan → decompose → hypothesize → gather ⟷ analyze → reflect → correlate → verify → synthesize → END
                                               ↑___________|        ↑_______|
    
    Multi-step reasoning adds:
    - Task decomposition for complex problems
    - Hypothesis generation and testing
    - Self-reflection and bias checking
    - Verification before synthesis
    
    Args:
        llm: Language model to use for reasoning
        checkpointer: Optional checkpointer for persistence
        enable_multi_step_reasoning: Whether to use enhanced reasoning (default True)
        
    Returns:
        Compiled StateGraph
    """
    
    # Create the graph with our state schema
    graph = StateGraph(AgentState)
    
    # === Add Nodes ===
    
    # Planning node - creates the research plan
    async def plan_node(state: AgentState) -> dict[str, Any]:
        return await plan_research(state, llm)
    
    # Decomposition node - breaks complex tasks into sub-tasks
    async def decompose_node(state: AgentState) -> dict[str, Any]:
        return await decompose_task(state, llm)
    
    # Hypothesis generation node - forms testable hypotheses
    async def hypothesize_node(state: AgentState) -> dict[str, Any]:
        return await generate_hypotheses(state, llm)
    
    # Gather node - placeholder that will be replaced per-run
    async def gather_node(state: AgentState) -> dict[str, Any]:
        return {
            "current_phase": ResearchPhase.ANALYZING.value,
            "iteration": state["iteration"] + 1,
        }
    
    # Analysis node
    async def analyze_node(state: AgentState) -> dict[str, Any]:
        return await analyze_findings(state, llm)
    
    # Reflection node - self-critique and bias check
    async def reflect_node(state: AgentState) -> dict[str, Any]:
        return await reflect_on_analysis(state, llm)
    
    # Correlation node
    async def correlate_node(state: AgentState) -> dict[str, Any]:
        return await correlate_findings(state, llm)
    
    # Verification node - verify conclusions before synthesis
    async def verify_node(state: AgentState) -> dict[str, Any]:
        return await verify_conclusions(state, llm)
    
    # Synthesis node
    async def synthesize_node(state: AgentState) -> dict[str, Any]:
        return await synthesize_report(state, llm)
    
    # Add all nodes (including multi-step reasoning nodes)
    graph.add_node("plan", plan_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("hypothesize", hypothesize_node)
    graph.add_node("gather", gather_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("correlate", correlate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("synthesize", synthesize_node)
    
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
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        use_checkpointer: bool = True,
    ):
        """
        Initialize the Deep Research Agent.
        
        Args:
            mcp_server_url: URL of the MCP server SSE endpoint
            provider: LLM provider (gemini, grok, ollama, docker)
            model: Model name override
            temperature: LLM temperature
            max_iterations: Maximum research iterations
            use_checkpointer: Whether to enable state persistence
        """
        self.mcp_server_url = mcp_server_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.provider = provider or settings.agent_provider
        self.model = model
        self.temperature = temperature
        self.max_iterations = max_iterations if max_iterations is not None else settings.agent_max_iterations
        
        # Initialize LLM
        self.llm = get_llm(
            provider=self.provider,
            model=self.model,
            temperature=self.temperature,
        )
        
        # Initialize checkpointer for persistence
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        # Build the graph
        self.graph = build_research_graph(self.llm, self.checkpointer)
        
        logger.info(f"Deep Research Agent initialized")
        logger.info(f"Provider: [bold]{self.provider}[/bold]")
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
        
        # Initialize output writer for this session
        reset_output_writer()
        output_writer = get_output_writer()
        
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
                        logger.success(f"Reasoning log saved to: [bold]{output_paths['reasoning_log']}[/bold]")
                        final_state["output_paths"] = output_paths
                    
                    return final_state
                    
        except Exception as e:
            logger.error(f"Research failed: {e}")
            return {
                **initial_state,
                "error": str(e),
                "current_phase": ResearchPhase.COMPLETE.value,
            }
    
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
        
        while state["current_phase"] != ResearchPhase.COMPLETE.value:
            # Check iteration limit
            if state["iteration"] >= state["max_iterations"]:
                logger.warning("Max iterations reached, forcing synthesis")
                state["current_phase"] = ResearchPhase.SYNTHESIZING.value
            
            phase = state["current_phase"]
            
            if phase == ResearchPhase.PLANNING.value:
                updates = await plan_research(state, self.llm)
                state = {**state, **updates}
            
            # Multi-step Reasoning: Task Decomposition
            elif phase == ResearchPhase.DECOMPOSING.value:
                updates = await decompose_task(state, self.llm)
                state = {**state, **updates}
            
            # Multi-step Reasoning: Hypothesis Generation
            elif phase == ResearchPhase.HYPOTHESIZING.value:
                updates = await generate_hypotheses(state, self.llm)
                state = {**state, **updates}
                
            elif phase == ResearchPhase.GATHERING.value:
                updates = await gather_intelligence(state, tool_executor)
                state = {**state, **updates}
                # After gathering, update hypotheses if we have any
                if state.get("hypotheses"):
                    updates = await update_hypotheses(state, self.llm)
                    state = {**state, **updates}
                
            elif phase == ResearchPhase.ANALYZING.value:
                updates = await analyze_findings(state, self.llm)
                state = {**state, **updates}
            
            # Multi-step Reasoning: Self-Reflection
            elif phase == ResearchPhase.REFLECTING.value:
                updates = await reflect_on_analysis(state, self.llm)
                state = {**state, **updates}
                
            elif phase == ResearchPhase.CORRELATING.value:
                updates = await correlate_findings(state, self.llm)
                state = {**state, **updates}
            
            # Multi-step Reasoning: Verification
            elif phase == ResearchPhase.VERIFYING.value:
                updates = await verify_conclusions(state, self.llm)
                state = {**state, **updates}
                
            elif phase == ResearchPhase.SYNTHESIZING.value:
                updates = await synthesize_report(state, self.llm)
                state = {**state, **updates}
                
            else:
                break
        
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
