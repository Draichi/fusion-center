"""
Main Research Agent using PydanticAI.

This is the core agent implementation with:
- Dual-LLM architecture (thinking + structured)
- Multi-step reasoning (decompose, hypothesize, reflect, verify)
- MCP tool integration via MCPServerSSE
- Explicit Python control flow for visibility
- WebSocket integration for real-time dashboard updates
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    SynthesisThinkingOutput,
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
    SITREP_THINKING_PROMPT,
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

if TYPE_CHECKING:
    from src.agent_v2.websocket import WebSocketManager

logger = get_logger()


class ResearchAgent:
    """
    PydanticAI-based research agent with multi-step reasoning.

    Features:
    - Dual-LLM architecture (structured + thinking)
    - Multi-step reasoning phases
    - MCP tool integration
    - Explicit control flow with logging
    - WebSocket integration for real-time dashboard updates
    """

    def __init__(
        self,
        mcp_url: str | None = None,
        llm_config: DualLLMConfig | None = None,
        ws_manager: WebSocketManager | None = None,
    ):
        """
        Initialize the research agent.

        Args:
            mcp_url: URL of the MCP server SSE endpoint
            llm_config: Configuration for dual-LLM architecture
            ws_manager: Optional WebSocket manager for real-time updates
        """
        self.mcp_url = mcp_url or f"http://{settings.mcp_server_host}:{settings.mcp_server_port}/sse"
        self.config = llm_config or DualLLMConfig()
        self.ws_manager = ws_manager

        # Store context for external access (e.g., from API)
        self.ctx: ResearchContext | None = None

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
        if ws_manager:
            logger.info(f"  WebSocket: enabled")
    
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
        
        # Structured LLM agents for complex output schemas
        # These need JSON mode, so use structured_model (not thinking_model)
        self.reflector = Agent(
            self.structured_model,
            output_type=ReflectionOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + REFLECTION_PROMPT,
            retries=3,
        )

        self.verifier = Agent(
            self.structured_model,
            output_type=VerificationOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + VERIFICATION_PROMPT,
            retries=3,
        )

        # Two-stage synthesis: thinking model for deep analysis, structured model for JSON formatting
        self.synthesis_thinker = Agent(
            self.thinking_model,
            output_type=SynthesisThinkingOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + SITREP_THINKING_PROMPT,
            retries=3,
        )

        self.synthesizer = Agent(
            self.structured_model,
            output_type=SITREPOutput,
            system_prompt=SYSTEM_PROMPT + "\n\n" + SITREP_PROMPT,
            retries=5,  # SITREP schema is complex, needs more retries
        )

        # Gatherer with MCP tools - uses thinking model because tool calling
        # requires better reasoning about what tools to call and how to interpret results
        # No output_type - just needs to call tools, not produce structured output
        self.gatherer = Agent(
            self.thinking_model,
            system_prompt=SYSTEM_PROMPT + "\n\n" + GATHERER_PROMPT,
            toolsets=[self.mcp_server],
            end_strategy="early",  # Stop after tools are done
        )

    async def _broadcast_phase(self, phase: str, ctx: ResearchContext) -> None:
        """Broadcast phase change via WebSocket if manager is configured."""
        if self.ws_manager:
            try:
                await self.ws_manager.broadcast_phase_change(phase, ctx)
            except Exception as e:
                logger.warning(f"Failed to broadcast phase change: {e}")

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
        self.ctx = ctx  # Store for external access

        async with self.mcp_server:
            logger.success("Connected to MCP server")

            # Phase 1: Planning
            await self._run_planning(ctx, task)
            await self._broadcast_phase("planning_complete", ctx)

            # Phase 2: Decomposition
            await self._run_decomposition(ctx, task)
            await self._broadcast_phase("decomposition_complete", ctx)

            # Phase 3: Hypothesis Generation
            await self._run_hypothesis_generation(ctx, task)
            await self._broadcast_phase("hypothesis_complete", ctx)

            # Phase 4-6: Gather → Analyze → Reflect Loop
            for iteration in range(max_iterations):
                logger.info(f"🔄 Iteration {iteration + 1}/{max_iterations}")
                ctx.iteration = iteration + 1

                await self._broadcast_phase("gathering", ctx)
                await self._run_gather(ctx)
                await self._broadcast_phase("gathering_complete", ctx)

                await self._broadcast_phase("analysis", ctx)
                await self._run_analysis(ctx)
                await self._broadcast_phase("analysis_complete", ctx)

                await self._broadcast_phase("reflection", ctx)
                should_continue = await self._run_reflection(ctx)
                await self._broadcast_phase("reflection_complete", ctx)

                if not should_continue:
                    break

            # Phase 7: Verification
            await self._broadcast_phase("verification", ctx)
            await self._run_verification(ctx)
            await self._broadcast_phase("verification_complete", ctx)

            # Phase 8: Synthesis
            await self._broadcast_phase("synthesis", ctx)
            sitrep = await self._run_synthesis(ctx)
            await self._broadcast_phase("complete", ctx)

            return sitrep
    
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
        """Phase 8: Synthesize final SITREP report using two-stage approach."""
        logger.info("📝 Phase 8: Synthesizing SITREP...")
        synthesis_prompt = build_synthesis_prompt(ctx)

        # Stage 1: Deep thinking with thinking model
        logger.info("  🧠 Stage 1: Deep analysis (thinking model)...")
        log_prompt("08a_synthesis_thinking", synthesis_prompt)

        thinking_output = None
        try:
            thinking_result = await self.synthesis_thinker.run(synthesis_prompt)
            thinking_output = thinking_result.output
            logger.info(f"  → Analysis complete: {len(thinking_output.key_findings)} key findings")
            logger.info(f"  → Confidence: {thinking_output.confidence_assessment[:80]}...")
        except Exception as e:
            logger.warning(f"  ⚠️ Thinking stage failed: {e}")

        # Stage 2: Format into SITREP JSON with structured model
        logger.info("  📋 Stage 2: Formatting SITREP (structured model)...")

        if thinking_output:
            # Build formatting prompt with thinking output
            format_prompt = self._build_format_prompt(ctx, thinking_output)
        else:
            # Fallback: use original synthesis prompt
            format_prompt = synthesis_prompt

        log_prompt("08b_synthesis_format", format_prompt)

        try:
            result = await self.synthesizer.run(format_prompt)

            logger.success("Research complete!")
            logger.info(f"  → Intelligence quality: {result.output.section_i.intelligence_quality}")
            logger.info(f"  → Overall confidence: {result.output.section_i.overall_confidence_percent}%")
            logger.info(f"  → Sources used: {', '.join(result.output.intelligence_sources_used)}")

            return result.output
        except Exception as e:
            logger.warning(f"  ⚠️ Formatting failed, building from thinking output: {e}")
            # Build SITREP from thinking output if available
            return self._build_sitrep_from_thinking(ctx, thinking_output)

    def _build_format_prompt(self, ctx: ResearchContext, thinking: SynthesisThinkingOutput) -> str:
        """Build prompt for formatting stage with thinking output."""
        return f"""Format this intelligence analysis into the SITREP JSON structure.

## Original Query
{ctx.task}

## Analyst's Analysis

### Executive Summary
{thinking.executive_summary}

### Key Findings
{chr(10).join(f'- {f}' for f in thinking.key_findings)}

### Detailed Analysis
{thinking.detailed_analysis}

### Source Analysis
- Satellite: {thinking.satellite_analysis}
- News: {thinking.news_analysis}
- Cyber: {thinking.cyber_analysis}
- Social: {thinking.social_analysis}

### Cross-Source Insights
{thinking.cross_source_insights}

### Intelligence Gaps
{chr(10).join(f'- {g}' for g in thinking.intelligence_gaps) if thinking.intelligence_gaps else 'None identified'}

### Recommendations
{chr(10).join(f'- {r}' for r in thinking.recommendations) if thinking.recommendations else 'None'}

### Monitoring Priorities
{chr(10).join(f'- {m}' for m in thinking.monitoring_priorities) if thinking.monitoring_priorities else 'None'}

### Confidence Assessment
{thinking.confidence_assessment}

### Sources Used
{', '.join(thinking.sources_used) if thinking.sources_used else 'Various OSINT sources'}

Now format this into the complete SITREP JSON structure.
"""

    def _build_sitrep_from_thinking(
        self, ctx: ResearchContext, thinking: SynthesisThinkingOutput | None
    ) -> SITREPOutput:
        """Build SITREP directly from thinking output when formatting fails."""
        from src.agent_v2.schemas import (
            SITREPSectionI, SITREPSectionII, SITREPSectionIII,
            SITREPSectionIV, SITREPSectionV, SITREPSectionVI
        )

        if thinking:
            # Parse confidence from assessment text
            confidence_text = thinking.confidence_assessment.lower()
            if "high" in confidence_text or "strong" in confidence_text:
                confidence_pct = 75
                quality = "GOOD"
            elif "medium" in confidence_text or "moderate" in confidence_text:
                confidence_pct = 55
                quality = "FAIR"
            elif "low" in confidence_text or "limited" in confidence_text:
                confidence_pct = 35
                quality = "POOR"
            else:
                confidence_pct = 50
                quality = "FAIR"

            return SITREPOutput(
                query_summary=ctx.task,
                intelligence_sources_used=thinking.sources_used or ["OSINT sources"],
                section_i=SITREPSectionI(
                    direct_response=thinking.executive_summary,
                    key_highlights=thinking.key_findings[:5],
                    overall_confidence_percent=confidence_pct,
                    intelligence_quality=quality
                ),
                section_ii=SITREPSectionII(
                    cross_topic_connections=thinking.detailed_analysis[:500] if thinking.detailed_analysis else ""
                ),
                section_iii=SITREPSectionIII(
                    satellite_intel=thinking.satellite_analysis,
                    news_intel=thinking.news_analysis,
                    cyber_intel=thinking.cyber_analysis,
                    social_intel=thinking.social_analysis,
                    cross_source_validation=thinking.cross_source_insights,
                    intelligence_gaps=thinking.intelligence_gaps
                ),
                section_iv=SITREPSectionIV(
                    immediate_actions=thinking.recommendations,
                    monitoring_indicators=thinking.monitoring_priorities
                ),
                section_v=SITREPSectionV(
                    analytical_confidence=thinking.confidence_assessment
                ),
                section_vi=SITREPSectionVI()
            )
        else:
            # Complete fallback when both stages fail
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
