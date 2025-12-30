"""
Main Research Agent using PydanticAI.

This is the core agent implementation with:
- Dual-LLM architecture (thinking + structured)
- Multi-step reasoning (decompose, hypothesize, reflect, verify)
- MCP tool integration via MCPServerSSE
- Explicit Python control flow for visibility
"""

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

from src.agent_v2.config import DualLLMConfig, create_model
from src.agent_v2.state import ResearchContext, Hypothesis, SubTask, ReflectionNote
from src.agent_v2.schemas import (
    ResearchPlanOutput,
    TaskDecompositionOutput,
    HypothesisGenerationOutput,
    AnalysisOutput,
    ReflectionOutput,
    VerificationOutput,
    SITREPOutput,
)
from src.agent_v2.prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    DECOMPOSITION_PROMPT,
    HYPOTHESIS_PROMPT,
    GATHERER_PROMPT,
    ANALYST_PROMPT,
    REFLECTION_PROMPT,
    VERIFICATION_PROMPT,
    SITREP_PROMPT,
)
from src.agent_v2.phases import (
    build_gather_prompt,
    build_analysis_prompt,
    build_reflection_prompt,
    build_verification_prompt,
    build_synthesis_prompt,
)
from src.agent_v2.debug import log_prompt, DEBUG_PROMPTS
from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


class ResearchAgent:
    """
    PydanticAI-based research agent with multi-step reasoning.
    
    Features:
    - Dual-LLM architecture (structured + thinking)
    - Multi-step reasoning phases
    - MCP tool integration
    - Explicit control flow with logging
    """
    
    def __init__(
        self,
        mcp_url: str | None = None,
        llm_config: DualLLMConfig | None = None,
    ):
        """
        Initialize the research agent.
        
        Args:
            mcp_url: URL of the MCP server SSE endpoint
            llm_config: Configuration for dual-LLM architecture
        """
        self.mcp_url = mcp_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.config = llm_config or DualLLMConfig()
        
        # Create model instances
        self.structured_model = create_model(self.config.structured_model)
        self.thinking_model = create_model(self.config.thinking_model)
        
        # MCP server connection
        self.mcp_server = MCPServerSSE(self.mcp_url)
        
        # Initialize specialized agents
        self._init_agents()
        
        logger.info(f"ResearchAgent initialized")
        logger.info(f"  Structured LLM: {self.config.structured_model}")
        logger.info(f"  Thinking LLM: {self.config.thinking_model}")
        logger.info(f"  MCP Server: {self.mcp_url}")
    
    def _init_agents(self) -> None:
        """Initialize specialized agents for each research phase."""
        # Structured LLM agents (JSON output)
        self.planner = Agent(
            self.structured_model,
            output_type=ResearchPlanOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + PLANNER_PROMPT,
        )
        
        self.decomposer = Agent(
            self.structured_model,
            output_type=TaskDecompositionOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + DECOMPOSITION_PROMPT,
        )
        
        self.hypothesizer = Agent(
            self.structured_model,
            output_type=HypothesisGenerationOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + HYPOTHESIS_PROMPT,
        )
        
        self.analyzer = Agent(
            self.structured_model,
            output_type=AnalysisOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + ANALYST_PROMPT,
        )
        
        # Thinking LLM agents (complex reasoning)
        self.reflector = Agent(
            self.thinking_model,
            output_type=ReflectionOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + REFLECTION_PROMPT,
        )
        
        self.verifier = Agent(
            self.thinking_model,
            output_type=VerificationOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + VERIFICATION_PROMPT,
        )
        
        self.synthesizer = Agent(
            self.thinking_model,
            output_type=SITREPOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + SITREP_PROMPT,
        )
        
        # Gatherer with MCP tools - uses thinking model because tool calling 
        # requires better model support (local models like qwen2.5:7b struggle with tool results)
        # No output_type - the gatherer just needs to call tools, not produce structured output
        self.gatherer = Agent(
            self.thinking_model,
            system_prompt=SYSTEM_PROMPT + "\n\n" + GATHERER_PROMPT,
            toolsets=[self.mcp_server],
            end_strategy="early",  # Stop after tools are done
        )
    
    async def research(
        self,
        task: str,
        max_iterations: int = 5,
    ) -> SITREPOutput:
        """
        Run the full research pipeline.
        
        Args:
            task: The research question/task
            max_iterations: Maximum gather-analyze-reflect iterations
            
        Returns:
            Complete SITREP report
        """
        logger.info(f"🔍 Starting research: {task}")
        
        if DEBUG_PROMPTS:
            logger.info("📝 DEBUG_PROMPTS enabled - prompts will be logged to logs/ directory")
        
        ctx = ResearchContext(task=task, max_iterations=max_iterations)
        
        async with self.mcp_server:
            logger.success("Connected to MCP server")
            
            # Phase 1: Planning
            await self._run_planning(ctx, task)
            
            # Phase 2: Decomposition
            await self._run_decomposition(ctx, task)
            
            # Phase 3: Hypothesis Generation
            await self._run_hypothesis_generation(ctx, task)
            
            # Phase 4-6: Gather → Analyze → Reflect Loop
            for iteration in range(max_iterations):
                logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                ctx.iteration = iteration + 1
                
                await self._run_gather(ctx)
                await self._run_analysis(ctx)
                
                should_continue = await self._run_reflection(ctx)
                if not should_continue:
                    break
            
            # Phase 7: Verification
            await self._run_verification(ctx)
            
            # Phase 8: Synthesis
            return await self._run_synthesis(ctx)
    
    async def _run_planning(self, ctx: ResearchContext, task: str) -> None:
        """Phase 1: Create research plan."""
        logger.info("📋 Phase 1: Planning research...")
        plan_prompt = f"Create a research plan for: {task}"
        log_prompt("01_planning", plan_prompt)
        
        result = await self.planner.run(plan_prompt)
        ctx.research_plan = result.output.model_dump()
        
        logger.info(f"  → {len(result.output.objectives)} objectives:")
        for i, obj in enumerate(result.output.objectives, 1):
            logger.info(f"     {i}. {obj}")
        
        if result.output.regions_of_interest:
            logger.info(f"  → Regions: {', '.join(result.output.regions_of_interest)}")
        
        if result.output.keywords:
            logger.info(f"  → Keywords: {', '.join(result.output.keywords)}")
        
        logger.info(f"  → {len(result.output.initial_queries)} initial queries planned")
    
    async def _run_decomposition(self, ctx: ResearchContext, task: str) -> None:
        """Phase 2: Decompose task into sub-tasks."""
        logger.info("🔍 Phase 2: Decomposing task...")
        decomp_prompt = f"Task: {task}\nPlan objectives: {ctx.research_plan.get('objectives', [])}"
        log_prompt("02_decomposition", decomp_prompt)
        
        result = await self.decomposer.run(decomp_prompt)
        ctx.task_complexity = result.output.task_complexity
        
        for st in result.output.sub_tasks:
            ctx.sub_tasks.append(SubTask(
                id=st.id,
                description=st.description,
                focus_area=st.focus_area,
            ))
        
        logger.info(f"  → Complexity: {ctx.task_complexity}")
        logger.info(f"  → {len(ctx.sub_tasks)} sub-tasks:")
        for st in result.output.sub_tasks:
            logger.info(f"     [{st.id}] {st.description} ({st.focus_area})")
    
    async def _run_hypothesis_generation(self, ctx: ResearchContext, task: str) -> None:
        """Phase 3: Generate testable hypotheses."""
        logger.info("💡 Phase 3: Generating hypotheses...")
        hypo_prompt = f"Task: {task}\nSub-tasks: {[st.description for st in ctx.sub_tasks]}"
        log_prompt("03_hypothesis", hypo_prompt)
        
        result = await self.hypothesizer.run(hypo_prompt)
        
        for h in result.output.hypotheses:
            ctx.hypotheses.append(Hypothesis(
                id=h.id,
                statement=h.statement,
                confidence=h.initial_confidence,
            ))
        
        logger.info(f"  → {len(ctx.hypotheses)} hypotheses generated:")
        for h in result.output.hypotheses:
            confidence_pct = int(h.initial_confidence * 100)
            logger.info(f"     [{h.id}] {h.statement} (confidence: {confidence_pct}%)")
    
    async def _run_gather(self, ctx: ResearchContext) -> None:
        """Phase 4: Gather intelligence via MCP tools."""
        logger.info("  📡 Gathering intelligence...")
        gather_prompt = build_gather_prompt(ctx)
        log_prompt("04_gathering", gather_prompt, iteration=ctx.iteration)
        
        result = await self.gatherer.run(gather_prompt)
        
        # Count tool calls from the message history
        tool_calls = [msg for msg in result.all_messages() if hasattr(msg, 'parts') and any(hasattr(p, 'tool_name') for p in getattr(msg, 'parts', []))]
        logger.info(f"  → {len(tool_calls)} tool interactions")
        logger.info(f"  → Gathering complete")
    
    async def _run_analysis(self, ctx: ResearchContext) -> None:
        """Phase 5: Analyze gathered findings."""
        logger.info("  🧠 Analyzing findings...")
        analysis_prompt = build_analysis_prompt(ctx)
        log_prompt("05_analysis", analysis_prompt, iteration=ctx.iteration)
        
        try:
            result = await self.analyzer.run(analysis_prompt)
            ctx.key_insights = result.output.key_insights
            ctx.uncertainties = result.output.uncertainties
            
            # Store correlations as strings now
            ctx.correlations.extend(result.output.correlations)
            
            # Log thinking process if available
            if result.output.thinking:
                thinking_preview = result.output.thinking[:150] + "..." if len(result.output.thinking) > 150 else result.output.thinking
                logger.info(f"  → Thinking: {thinking_preview}")
            
            logger.info(f"  → {len(ctx.key_insights)} insights found:")
            for i, insight in enumerate(ctx.key_insights, 1):
                # Truncate long insights for readability
                display = insight[:100] + "..." if len(insight) > 100 else insight
                logger.info(f"     {i}. {display}")
            
            if ctx.uncertainties:
                logger.info(f"  → {len(ctx.uncertainties)} uncertainties identified")
        except Exception as e:
            logger.warning(f"  ⚠️ Analysis failed, continuing with existing insights: {e}")
            # Add a note about the failure
            ctx.uncertainties.append(f"Analysis phase failed on iteration {ctx.iteration}")
    
    async def _run_reflection(self, ctx: ResearchContext) -> bool:
        """Phase 6: Reflect on analysis. Returns True if more investigation needed."""
        logger.info("  🪞 Reflecting on analysis...")
        reflection_prompt = build_reflection_prompt(ctx)
        log_prompt("06_reflection", reflection_prompt, iteration=ctx.iteration)
        
        try:
            result = await self.reflector.run(reflection_prompt)
            
            for note in result.output.reflection_notes:
                ctx.reflection_notes.append(ReflectionNote(
                    category=note.category,
                    content=note.content,
                    severity=note.severity,
                ))
            
            # Log reflection summary
            if result.output.summary:
                logger.info(f"  → Summary: {result.output.summary[:100]}...")
            
            # Log reflection notes by severity
            critical_notes = [n for n in result.output.reflection_notes if n.severity == "critical"]
            warning_notes = [n for n in result.output.reflection_notes if n.severity == "warning"]
            
            if critical_notes:
                logger.warning(f"  → {len(critical_notes)} critical issues found")
                for note in critical_notes:
                    display = note.content[:80] + "..." if len(note.content) > 80 else note.content
                    logger.warning(f"     ⚠️  {display}")
            
            if warning_notes:
                logger.info(f"  → {len(warning_notes)} warnings noted")
            
            if result.output.needs_more_investigation:
                if result.output.next_steps:
                    logger.info(f"  → Next steps: {', '.join(result.output.next_steps[:3])}")
                logger.info("  → More investigation needed, continuing...")
                return True
            else:
                logger.info("  → Reflection complete, moving to verification")
                return False
        except Exception as e:
            logger.warning(f"  ⚠️ Reflection failed, moving to verification: {e}")
            return False  # Stop iterating and move to verification
    
    async def _run_verification(self, ctx: ResearchContext) -> None:
        """Phase 7: Verify conclusions."""
        logger.info("✅ Phase 7: Verifying conclusions...")
        verify_prompt = build_verification_prompt(ctx)
        log_prompt("07_verification", verify_prompt)
        
        try:
            result = await self.verifier.run(verify_prompt)
            
            passed = 0
            adjusted = 0
            failed = 0
            
            for iv in result.output.insight_verifications:
                # Normalize verdict to lowercase for comparison
                verdict = iv.verdict.lower() if iv.verdict else "fail"
                if verdict == "pass":
                    passed += 1
                    ctx.verified_insights.append(iv.insight)
                elif verdict == "adjust":
                    adjusted += 1
                    ctx.verified_insights.append(iv.insight)
                else:
                    failed += 1
            
            logger.info(f"  → Verification results: {passed} passed, {adjusted} adjusted, {failed} failed")
            logger.info(f"  → {len(ctx.verified_insights)} insights verified for final report")
        except Exception as e:
            logger.warning(f"  ⚠️ Verification failed, using unverified insights: {e}")
            # Use key insights as verified if verification fails
            ctx.verified_insights = ctx.key_insights.copy()
    
    async def _run_synthesis(self, ctx: ResearchContext) -> SITREPOutput:
        """Phase 8: Synthesize final SITREP report."""
        logger.info("📝 Phase 8: Synthesizing SITREP...")
        synthesis_prompt = build_synthesis_prompt(ctx)
        log_prompt("08_synthesis", synthesis_prompt)
        
        try:
            result = await self.synthesizer.run(synthesis_prompt)
            
            logger.success("Research complete!")
            logger.info(f"  → Intelligence quality: {result.output.section_i.intelligence_quality}")
            logger.info(f"  → Overall confidence: {result.output.section_i.overall_confidence_percent}%")
            logger.info(f"  → Sources used: {', '.join(result.output.intelligence_sources_used)}")
            
            return result.output
        except Exception as e:
            logger.warning(f"  ⚠️ Synthesis failed, returning basic report: {e}")
            # Return a minimal SITREP on failure
            from src.agent_v2.schemas import (
                SITREPSectionI, SITREPSectionII, SITREPSectionIII,
                SITREPSectionIV, SITREPSectionV, SITREPSectionVI
            )
            return SITREPOutput(
                query_summary=ctx.task,
                intelligence_sources_used=["Analysis incomplete due to error"],
                section_i=SITREPSectionI(
                    direct_response=f"Analysis incomplete. Key insights: {'; '.join(ctx.key_insights[:3]) if ctx.key_insights else 'None gathered'}",
                    key_highlights=ctx.key_insights[:5] if ctx.key_insights else ["No data collected"],
                    overall_confidence_percent=20,
                    intelligence_quality="POOR"
                ),
                section_ii=SITREPSectionII(),
                section_iii=SITREPSectionIII(intelligence_gaps=ctx.uncertainties),
                section_iv=SITREPSectionIV(),
                section_v=SITREPSectionV(),
                section_vi=SITREPSectionVI()
            )
