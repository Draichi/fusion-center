"""
Graph Nodes for the Deep Research Agent.

Each node represents a step in the research process.
Includes Multi-step Reasoning nodes for deeper analysis.

Uses Pydantic schemas with `with_structured_output()` for reliable JSON responses.
"""

import json
import uuid
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from src.agent.state import AgentState, IntelligenceType, ResearchPhase, HypothesisStatus
from src.shared.config import settings
from src.agent.tools import MCPToolExecutor, get_tool_definitions, TOOL_NAME_ALIASES, VALID_TOOL_NAMES
from src.agent.prompts import (
    SYSTEM_PROMPT,
    PLANNER_PROMPT,
    ANALYST_PROMPT,
    SYNTHESIZER_PROMPT,
    # Multi-step Reasoning prompts
    TASK_DECOMPOSITION_PROMPT,
    HYPOTHESIS_GENERATION_PROMPT,
    HYPOTHESIS_UPDATE_PROMPT,
    REFLECTION_PROMPT,
    VERIFICATION_PROMPT,
    ENHANCED_ANALYST_PROMPT,
    ENHANCED_SYNTHESIZER_PROMPT,
)
from src.agent.schemas import (
    ResearchPlanOutput,
    TaskDecompositionOutput,
    HypothesisGenerationOutput,
    HypothesisUpdateOutput,
    AnalysisOutput,
    ReflectionOutput,
    CorrelationOutput,
    VerificationOutput,
    SynthesisOutput,
)
from src.shared.logger import get_logger, log_agent_step
from src.shared.output_writer import get_output_writer

logger = get_logger()


def _resolve_tool_name(tool_name: str) -> str:
    """
    Resolve a tool name to its canonical form, handling aliases.
    
    This ensures that aliased tool names (like 'search_news_by_location')
    are resolved to their actual names (like 'search_news') before
    processing results.
    
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


# =============================================================================
# Helper Functions for Multi-step Reasoning
# =============================================================================


def _add_reasoning_step(
    state: AgentState,
    phase: str,
    thought: str,
    action: str,
    observation: str,
    conclusion: str,
    confidence: float,
) -> dict[str, Any]:
    """Add a reasoning step to the trace."""
    step = {
        "step_number": len(state.get("reasoning_trace", [])) + 1,
        "phase": phase,
        "thought": thought,
        "action": action,
        "observation": observation,
        "conclusion": conclusion,
        "confidence": confidence,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return step


def _extract_thinking_block(response_text: str) -> tuple[str, str]:
    """Extract thinking block from response if present."""
    thinking = ""
    main_response = response_text
    
    if "```thinking" in response_text:
        parts = response_text.split("```thinking")
        if len(parts) > 1:
            thinking_part = parts[1].split("```")[0]
            thinking = thinking_part.strip()
            # Get everything after the thinking block
            remaining = "```".join(parts[1].split("```")[1:])
            main_response = parts[0] + remaining
    
    return thinking, main_response.strip()


def _extract_json_with_logging(
    response_content: str,
    phase: str,
    context: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Extract JSON from LLM response with detailed logging.
    
    Args:
        response_content: Raw response from LLM
        phase: Current phase (for logging context)
        context: Additional context about what we were trying to parse
    
    Returns:
        Tuple of (parsed_dict, error_message). If parsing succeeds,
        error_message is None. If parsing fails, parsed_dict is None.
    """
    writer = get_output_writer()
    
    # Extract thinking block first
    thinking, main_response = _extract_thinking_block(response_content)
    
    # Log raw response for debugging
    response_preview = response_content[:500] + "..." if len(response_content) > 500 else response_content
    
    # Try to extract JSON from code blocks
    json_str = main_response
    if "```json" in main_response:
        json_str = main_response.split("```json")[1].split("```")[0]
    elif "```" in main_response:
        # Try to extract from generic code block
        parts = main_response.split("```")
        if len(parts) >= 2:
            json_str = parts[1].split("```")[0]
    
    json_str = json_str.strip()
    
    # Check if we have any content to parse
    if not json_str:
        error_msg = f"Empty response from LLM (phase: {phase})"
        logger.error(f"❌ {error_msg}")
        logger.error(f"   Raw response preview: {response_preview}")
        writer.log_reasoning(
            phase=phase,
            step=0,
            action="JSON parsing failed",
            details={
                "error": error_msg,
                "raw_response_preview": response_preview,
                "context": context,
            },
        )
        return None, error_msg
    
    try:
        result = json.loads(json_str)
        return result, None
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse JSON in {phase}: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"   JSON string attempted: {json_str[:300]}...")
        logger.error(f"   Raw response preview: {response_preview}")
        
        # Log to file for debugging
        writer.log_reasoning(
            phase=phase,
            step=0,
            action="JSON parsing failed",
            details={
                "error": str(e),
                "json_attempted": json_str[:500] if len(json_str) > 500 else json_str,
                "raw_response_preview": response_preview,
                "context": context,
            },
        )
        
        return None, error_msg


# =============================================================================
# Task Decomposition Node (Multi-step Reasoning)
# =============================================================================


async def decompose_task(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Decompose complex tasks into manageable sub-tasks.
    
    This node analyzes task complexity and breaks it down into
    focused sub-tasks that can be investigated independently.
    """
    log_agent_step(state["iteration"] + 1, "Decomposing task into sub-tasks")
    
    writer = get_output_writer()
    writer.log_reasoning(
        phase="decomposing",
        step=state["iteration"] + 1,
        action="Breaking down complex task into sub-tasks",
        details={"task": state["task"]},
    )
    
    decomposition_prompt = f"""
{TASK_DECOMPOSITION_PROMPT}

## Research Task
{state["task"]}

## Context
{json.dumps(state["context"], indent=2) if state["context"] else "No additional context provided."}

Analyze this task and decompose it into sub-tasks.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=decomposition_prompt),
    ]
    
    # Use structured output for reliable JSON
    thinking = ""
    try:
        structured_llm = llm.with_structured_output(TaskDecompositionOutput)
        output: TaskDecompositionOutput = await structured_llm.ainvoke(messages)
        result = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        result, error = _extract_json_with_logging(
            response.content,
            phase="decomposing",
            context=f"Task: {state['task'][:100]}...",
        )
        if error:
            return {
                "task_complexity": "simple",
                "current_phase": ResearchPhase.PLANNING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
        thinking, _ = _extract_thinking_block(response.content)
    
    # Build sub-tasks
    sub_tasks = []
    for st in result.get("sub_tasks", []):
        sub_tasks.append({
            "id": st.get("id", f"subtask_{len(sub_tasks) + 1}"),
            "description": st.get("description", ""),
            "parent_task": None,
            "status": "pending",
            "findings_ids": [],
            "dependencies": st.get("dependencies", []),
            "focus_area": st.get("focus_area", "thematic"),
            "complexity": st.get("complexity", "moderate"),
        })
    
    task_complexity = result.get("task_complexity", "moderate")
    
    logger.success(f"Task decomposed into {len(sub_tasks)} sub-tasks (complexity: {task_complexity})")
    
    # Log decomposition details
    writer.log_llm_response(
        phase="decomposing",
        prompt_summary=f"Decompose task: {state['task']}",
        response=json.dumps(result, indent=2),
    )
    
    if thinking:
        writer.log_reasoning(
            phase="decomposing",
            step=state["iteration"] + 1,
            action="Chain of thought",
            details={"thinking": thinking},
        )
    
    # Add reasoning step
    reasoning_step = _add_reasoning_step(
        state,
        phase="decomposing",
        thought=result.get("decomposition_reasoning", "Task needs to be broken down"),
        action=f"Decomposed into {len(sub_tasks)} sub-tasks",
        observation=f"Identified sub-tasks: {[st['description'] for st in sub_tasks]}",
        conclusion=f"Task complexity: {task_complexity}",
        confidence=0.85,
    )
    
    # Log summary
    summary_lines = [f"[bold]Task Complexity:[/bold] {task_complexity}"]
    for st in sub_tasks:
        dep_str = f" (deps: {st['dependencies']})" if st['dependencies'] else ""
        summary_lines.append(f"  • [{st['id']}] {st['description']}{dep_str}")
    
    logger.panel("\n".join(summary_lines), title="📋 Task Decomposition", style="blue")
    
    # Determine next phase based on complexity
    next_phase = ResearchPhase.HYPOTHESIZING.value if task_complexity in ["moderate", "complex"] else ResearchPhase.PLANNING.value
    
    return {
        "sub_tasks": sub_tasks,
        "current_sub_task_id": sub_tasks[0]["id"] if sub_tasks else None,
        "task_complexity": task_complexity,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        "chain_of_thought": state.get("chain_of_thought", []) + [thinking] if thinking else state.get("chain_of_thought", []),
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Hypothesis Generation Node (Multi-step Reasoning)
# =============================================================================


async def generate_hypotheses(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Generate testable hypotheses to guide research.
    
    This node forms hypotheses based on the task and existing findings,
    which will guide data gathering and analysis.
    """
    log_agent_step(state["iteration"] + 1, "Generating research hypotheses")
    
    writer = get_output_writer()
    writer.log_reasoning(
        phase="hypothesizing",
        step=state["iteration"] + 1,
        action="Forming testable hypotheses",
        details={
            "task": state["task"],
            "existing_findings": len(state.get("findings", [])),
        },
    )
    
    # Build context from existing findings if any
    findings_summary = ""
    if state.get("findings"):
        findings_summary = f"\n\n## Existing Findings ({len(state['findings'])} total)\n{json.dumps(state['findings'][:10], indent=2)}"
    
    hypothesis_prompt = f"""
{HYPOTHESIS_GENERATION_PROMPT}

## Research Task
{state["task"]}

## Context
{json.dumps(state["context"], indent=2) if state["context"] else "No additional context provided."}

## Sub-tasks (if decomposed)
{json.dumps(state.get("sub_tasks", []), indent=2) if state.get("sub_tasks") else "Task not decomposed."}
{findings_summary}

## Available Tools for Testing Hypotheses
{json.dumps(get_tool_definitions(), indent=2)}

Generate hypotheses that can be tested with these tools.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=hypothesis_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(HypothesisGenerationOutput)
        output: HypothesisGenerationOutput = await structured_llm.ainvoke(messages)
        result = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        result, error = _extract_json_with_logging(
            response.content,
            phase="hypothesizing",
            context=f"Task: {state['task'][:100]}...",
        )
        if error:
            return {
                "current_phase": ResearchPhase.PLANNING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    # Build hypotheses
    hypotheses = []
    test_queries = []
    
    for h in result.get("hypotheses", []):
        hypothesis = {
            "id": h.get("id", f"h{len(hypotheses) + 1}"),
            "statement": h.get("statement", ""),
            "status": HypothesisStatus.PROPOSED.value,
            "confidence": h.get("initial_confidence", 0.5),
            "supporting_evidence": [],
            "contradicting_evidence": [],
            "reasoning": "",
            "supporting_criteria": h.get("supporting_evidence_criteria", []),
            "refuting_criteria": h.get("refuting_evidence_criteria", []),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        hypotheses.append(hypothesis)
        
        # Collect test queries for this hypothesis
        for q in h.get("test_queries", []):
            q["hypothesis_id"] = hypothesis["id"]
            test_queries.append(q)
    
    logger.success(f"Generated {len(hypotheses)} hypotheses with {len(test_queries)} test queries")
    
    # Log response
    writer.log_llm_response(
        phase="hypothesizing",
        prompt_summary="Generate research hypotheses",
        response=json.dumps(result, indent=2),
    )
    
    # Log reasoning chain
    reasoning_chain = result.get("reasoning_chain", [])
    if reasoning_chain:
        writer.log_reasoning(
            phase="hypothesizing",
            step=state["iteration"] + 1,
            action="Reasoning chain",
            details={"chain": reasoning_chain},
        )
    
    # Add reasoning step
    reasoning_step = _add_reasoning_step(
        state,
        phase="hypothesizing",
        thought="\n".join(reasoning_chain) if reasoning_chain else "Forming hypotheses for investigation",
        action=f"Generated {len(hypotheses)} hypotheses",
        observation=f"Hypotheses: {[h['statement'] for h in hypotheses]}",
        conclusion=f"Ready to test with {len(test_queries)} queries",
        confidence=0.7,
    )
    
    # Log summary
    summary_lines = ["[bold]Hypotheses:[/bold]"]
    for h in hypotheses:
        conf_bar = "█" * int(h["confidence"] * 10) + "░" * (10 - int(h["confidence"] * 10))
        summary_lines.append(f"  [{h['id']}] {h['statement']}")
        summary_lines.append(f"      Confidence: [{conf_bar}] {h['confidence']:.0%}")
    
    logger.panel("\n".join(summary_lines), title="🔬 Research Hypotheses", style="cyan")
    
    # Use test queries if available, otherwise keep existing pending_queries from research plan
    queries_to_use = test_queries[:5] if test_queries else state.get("pending_queries", [])
    
    return {
        "hypotheses": state.get("hypotheses", []) + hypotheses,
        "active_hypothesis_id": hypotheses[0]["id"] if hypotheses else None,
        "pending_queries": queries_to_use,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        "chain_of_thought": state.get("chain_of_thought", []) + reasoning_chain,
        "current_phase": ResearchPhase.GATHERING.value,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Hypothesis Update Node (Multi-step Reasoning)
# =============================================================================


async def update_hypotheses(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Update hypothesis confidence based on new evidence.
    
    This node performs Bayesian-style updates to hypothesis
    confidence based on gathered findings.
    """
    log_agent_step(state["iteration"] + 1, "Updating hypotheses with new evidence")
    
    writer = get_output_writer()
    
    hypotheses = state.get("hypotheses", [])
    findings = state.get("findings", [])
    
    if not hypotheses or not findings:
        logger.info("No hypotheses or findings to update")
        return {
            "current_phase": ResearchPhase.ANALYZING.value,
            "iteration": state["iteration"] + 1,
        }
    
    writer.log_reasoning(
        phase="hypothesizing",
        step=state["iteration"] + 1,
        action="Updating hypotheses based on evidence",
        details={
            "num_hypotheses": len(hypotheses),
            "num_findings": len(findings),
        },
    )
    
    update_prompt = f"""
{HYPOTHESIS_UPDATE_PROMPT}

## Current Hypotheses
{json.dumps(hypotheses, indent=2)}

## New Findings
{json.dumps(findings[-20:], indent=2)}  # Last 20 findings

Review each finding and update hypothesis confidence accordingly.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=update_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(HypothesisUpdateOutput)
        output: HypothesisUpdateOutput = await structured_llm.ainvoke(messages)
        result = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        result, error = _extract_json_with_logging(
            response.content,
            phase="hypothesis_update",
            context=f"Updating {len(hypotheses)} hypotheses with {len(findings)} findings",
        )
        if error:
            return {
                "current_phase": ResearchPhase.ANALYZING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    # Update hypotheses
    updated_hypotheses = {h["id"]: h.copy() for h in hypotheses}
    
    for update in result.get("hypothesis_updates", []):
        h_id = update.get("hypothesis_id")
        if h_id in updated_hypotheses:
            h = updated_hypotheses[h_id]
            h["confidence"] = update.get("new_confidence", h["confidence"])
            h["status"] = update.get("new_status", h["status"])
            h["supporting_evidence"].extend(update.get("new_supporting_evidence", []))
            h["contradicting_evidence"].extend(update.get("new_contradicting_evidence", []))
            h["reasoning"] = update.get("confidence_change_reason", "")
            h["updated_at"] = datetime.utcnow().isoformat()
    
    updated_list = list(updated_hypotheses.values())
    
    # Log response
    writer.log_llm_response(
        phase="hypothesizing",
        prompt_summary="Update hypotheses with evidence",
        response=json.dumps(result, indent=2),
    )
    
    # Add reasoning step
    reasoning_chain = result.get("reasoning_chain", [])
    reasoning_step = _add_reasoning_step(
        state,
        phase="hypothesis_update",
        thought="\n".join(reasoning_chain) if reasoning_chain else "Evaluating evidence against hypotheses",
        action="Updated hypothesis confidence scores",
        observation=f"Updates: {len(result.get('hypothesis_updates', []))}",
        conclusion="Hypotheses updated based on evidence",
        confidence=0.8,
    )
    
    # Log summary
    summary_lines = ["[bold]Hypothesis Updates:[/bold]"]
    for h in updated_list:
        status_emoji = {
            "supported": "✅",
            "refuted": "❌",
            "investigating": "🔍",
            "inconclusive": "❓",
            "proposed": "📝",
        }.get(h["status"], "📝")
        conf_bar = "█" * int(h["confidence"] * 10) + "░" * (10 - int(h["confidence"] * 10))
        summary_lines.append(f"  {status_emoji} [{h['id']}] {h['statement']}")
        summary_lines.append(f"      Confidence: [{conf_bar}] {h['confidence']:.0%}")
    
    logger.panel("\n".join(summary_lines), title="📊 Hypothesis Status", style="yellow")
    
    return {
        "hypotheses": updated_list,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "chain_of_thought": state.get("chain_of_thought", []) + reasoning_chain,
        "current_phase": ResearchPhase.ANALYZING.value,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Reflection Node (Multi-step Reasoning)
# =============================================================================


async def reflect_on_analysis(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Perform critical self-reflection on the analysis.
    
    This node identifies biases, gaps, alternative explanations,
    and calibrates confidence levels.
    """
    log_agent_step(state["iteration"] + 1, "Self-reflection on analysis")
    
    writer = get_output_writer()
    writer.log_reasoning(
        phase="reflecting",
        step=state["iteration"] + 1,
        action="Critical self-examination of analysis",
        details={
            "insights_count": len(state.get("key_insights", [])),
            "hypotheses_count": len(state.get("hypotheses", [])),
            "reflection_iteration": state.get("reflection_iterations", 0) + 1,
        },
    )
    
    reflection_prompt = f"""
{REFLECTION_PROMPT}

## Original Task
{state["task"]}

## Current Hypotheses
{json.dumps(state.get("hypotheses", []), indent=2)}

## Key Insights So Far
{json.dumps(state.get("key_insights", []), indent=2)}

## Correlations Found
{json.dumps(state.get("correlations", []), indent=2)}

## Uncertainties Identified
{json.dumps(state.get("uncertainties", []), indent=2)}

## Available Tools for Additional Investigation
{json.dumps([t["name"] for t in get_tool_definitions()], indent=2)}

Critically reflect on this analysis. Be thorough but constructive.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=reflection_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(ReflectionOutput)
        output: ReflectionOutput = await structured_llm.ainvoke(messages)
        result = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        result, error = _extract_json_with_logging(
            response.content,
            phase="reflecting",
            context=f"Reflecting on {len(state.get('key_insights', []))} insights",
        )
        if error:
            return {
                "current_phase": ResearchPhase.CORRELATING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    # Build reflection notes
    reflection_notes = []
    for note in result.get("reflection_notes", []):
        reflection_notes.append({
            "category": note.get("category", "gap_analysis"),
            "content": note.get("content", ""),
            "severity": note.get("severity", "info"),
            "action_required": note.get("action_required", False),
            "suggested_action": note.get("suggested_action"),
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    needs_investigation = result.get("needs_more_investigation", False)
    investigation_queries = result.get("investigation_suggestions", [])
    
    # Log response
    writer.log_llm_response(
        phase="reflecting",
        prompt_summary="Self-reflection on analysis",
        response=json.dumps(result, indent=2),
    )
    
    # Add reasoning step
    reflection_chain = result.get("reflection_chain", [])
    reasoning_step = _add_reasoning_step(
        state,
        phase="reflecting",
        thought="\n".join(reflection_chain) if reflection_chain else "Examining analysis for biases and gaps",
        action=f"Generated {len(reflection_notes)} reflection notes",
        observation=f"Critical issues: {sum(1 for n in reflection_notes if n['severity'] == 'critical')}",
        conclusion="Needs more investigation" if needs_investigation else "Analysis is robust",
        confidence=0.75,
    )
    
    # Log summary with severity-based formatting
    summary_lines = ["[bold]Reflection Results:[/bold]"]
    severity_counts = {"info": 0, "warning": 0, "critical": 0}
    
    for note in reflection_notes:
        severity_counts[note["severity"]] = severity_counts.get(note["severity"], 0) + 1
        emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(note["severity"], "📝")
        summary_lines.append(f"  {emoji} [{note['category']}] {note['content']}")
        if note["action_required"] and note["suggested_action"]:
            summary_lines.append(f"      → Action: {note['suggested_action']}")
    
    summary_lines.append(f"\n[bold]Summary:[/bold] {severity_counts['critical']} critical, {severity_counts['warning']} warnings, {severity_counts['info']} info")
    
    style = "red" if severity_counts["critical"] > 0 else "yellow" if severity_counts["warning"] > 0 else "green"
    logger.panel("\n".join(summary_lines), title="🪞 Self-Reflection", style=style)
    
    # Determine next phase
    if needs_investigation and investigation_queries and state.get("reflection_iterations", 0) < 2:
        next_phase = ResearchPhase.GATHERING.value
        pending = investigation_queries[:3]
    elif any(n["severity"] == "critical" and n["action_required"] for n in reflection_notes):
        next_phase = ResearchPhase.ANALYZING.value
        pending = state.get("pending_queries", [])
    else:
        next_phase = ResearchPhase.CORRELATING.value
        pending = state.get("pending_queries", [])
    
    return {
        "reflection_notes": state.get("reflection_notes", []) + reflection_notes,
        "reflection_iterations": state.get("reflection_iterations", 0) + 1,
        "needs_more_reflection": needs_investigation,
        "pending_queries": pending,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "chain_of_thought": state.get("chain_of_thought", []) + reflection_chain,
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Verification Node (Multi-step Reasoning)
# =============================================================================


async def verify_conclusions(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Verify consistency and validity of conclusions.
    
    This node performs final checks before synthesis to ensure
    conclusions are supported by evidence and internally consistent.
    """
    log_agent_step(state["iteration"] + 1, "Verifying conclusions")
    
    writer = get_output_writer()
    writer.log_reasoning(
        phase="verifying",
        step=state["iteration"] + 1,
        action="Verifying consistency of conclusions",
        details={
            "insights_to_verify": len(state.get("key_insights", [])),
            "correlations_to_verify": len(state.get("correlations", [])),
        },
    )
    
    verification_prompt = f"""
{VERIFICATION_PROMPT}

## Original Task
{state["task"]}

## Findings (Evidence Base)
{json.dumps(state.get("findings", [])[:30], indent=2)}

## Hypotheses and Their Status
{json.dumps(state.get("hypotheses", []), indent=2)}

## Key Insights to Verify
{json.dumps(state.get("key_insights", []), indent=2)}

## Correlations to Verify
{json.dumps(state.get("correlations", []), indent=2)}

Verify each conclusion against the evidence.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=verification_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(VerificationOutput)
        output: VerificationOutput = await structured_llm.ainvoke(messages)
        result = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        result, error = _extract_json_with_logging(
            response.content,
            phase="verifying",
            context=f"Verifying {len(state.get('key_insights', []))} insights and {len(state.get('correlations', []))} correlations",
        )
        if error:
            return {
                "verified_insights": state.get("key_insights", []),
                "verified_correlations": state.get("correlations", []),
                "current_phase": ResearchPhase.SYNTHESIZING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    # Process verification results
    verification_results = []
    verified_insights = []
    verified_correlations = []
    
    # Process insight verifications
    for iv in result.get("insight_verifications", []):
        verification_results.append({
            "item_type": "insight",
            "item_id": iv.get("insight_index", 0),
            "is_consistent": iv.get("verdict") in ["pass", "adjust"],
            "issues_found": iv.get("issues", []),
            "confidence_adjustment": 0.0 if iv.get("verdict") == "pass" else -0.1,
            "verification_notes": iv.get("evidence_check", ""),
        })
        
        if iv.get("verdict") in ["pass", "adjust"]:
            # Use revised insight if available, otherwise original
            insight_text = iv.get("revised_insight") or iv.get("original_insight", "")
            if insight_text:
                verified_insights.append(insight_text)
    
    # Process correlation verifications
    for cv in result.get("correlation_verifications", []):
        verification_results.append({
            "item_type": "correlation",
            "item_id": cv.get("correlation_index", 0),
            "is_consistent": cv.get("verdict") in ["pass", "adjust"],
            "issues_found": cv.get("issues", []),
            "confidence_adjustment": 0.0,
            "verification_notes": f"Temporal: {cv.get('temporal_check', 'N/A')}, Spatial: {cv.get('spatial_check', 'N/A')}",
        })
        
        if cv.get("verdict") in ["pass", "adjust"]:
            # Get the original correlation and update its confidence
            corr_idx = cv.get("correlation_index", 0)
            correlations = state.get("correlations", [])
            if corr_idx < len(correlations):
                verified_corr = correlations[corr_idx].copy()
                verified_corr["confidence"] = cv.get("suggested_confidence", verified_corr.get("confidence", "medium"))
                verified_correlations.append(verified_corr)
    
    # Log response
    writer.log_llm_response(
        phase="verifying",
        prompt_summary="Verify conclusions before synthesis",
        response=json.dumps(result, indent=2),
    )
    
    # Add reasoning step
    verification_chain = result.get("verification_chain", [])
    overall = result.get("overall_assessment", {})
    
    reasoning_step = _add_reasoning_step(
        state,
        phase="verifying",
        thought="\n".join(verification_chain) if verification_chain else "Verifying all conclusions",
        action=f"Verified {len(verification_results)} items",
        observation=f"Passed: {sum(1 for v in verification_results if v['is_consistent'])}, Failed: {sum(1 for v in verification_results if not v['is_consistent'])}",
        conclusion="Ready for synthesis" if overall.get("ready_for_synthesis") else "Issues need addressing",
        confidence=0.85,
    )
    
    # Log summary
    passed = sum(1 for v in verification_results if v["is_consistent"])
    failed = len(verification_results) - passed
    
    summary_lines = [
        f"[bold]Verification Results:[/bold]",
        f"  ✅ Passed: {passed}",
        f"  ❌ Failed/Flagged: {failed}",
        f"  📝 Verified Insights: {len(verified_insights)}",
        f"  🔗 Verified Correlations: {len(verified_correlations)}",
    ]
    
    if overall.get("critical_issues"):
        summary_lines.append("\n[bold]Critical Issues:[/bold]")
        for issue in overall["critical_issues"]:
            summary_lines.append(f"  🚨 {issue}")
    
    style = "green" if overall.get("ready_for_synthesis") else "yellow"
    logger.panel("\n".join(summary_lines), title="✓ Verification Complete", style=style)
    
    # Determine next phase
    next_phase = ResearchPhase.SYNTHESIZING.value if overall.get("ready_for_synthesis", True) else ResearchPhase.REFLECTING.value
    
    return {
        "verification_results": state.get("verification_results", []) + verification_results,
        "verified_insights": verified_insights if verified_insights else state.get("key_insights", []),
        "verified_correlations": verified_correlations if verified_correlations else state.get("correlations", []),
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "chain_of_thought": state.get("chain_of_thought", []) + verification_chain,
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Planning Node
# =============================================================================


async def plan_research(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Plan the research approach based on the task.
    
    This node analyzes the task and creates a structured research plan
    including regions, keywords, and data sources to query.
    """
    log_agent_step(state["iteration"] + 1, "Planning research approach")
    
    # Log to reasoning file
    writer = get_output_writer()
    writer.log_reasoning(
        phase="planning",
        step=state["iteration"] + 1,
        action="Creating research plan",
        details={"task": state["task"], "context": state.get("context")},
    )
    
    planning_prompt = f"""
{PLANNER_PROMPT}

## Task
{state["task"]}

## Context
{json.dumps(state["context"], indent=2) if state["context"] else "No additional context provided."}

## Available Tools
{json.dumps(get_tool_definitions(), indent=2)}

Create a comprehensive research plan based on the task and available tools.
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=planning_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(ResearchPlanOutput)
        result: ResearchPlanOutput = await structured_llm.ainvoke(messages)
        plan = result.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        # Fallback to manual JSON extraction
        response = await llm.ainvoke(messages)
        plan, error = _extract_json_with_logging(
            response.content,
            phase="planning",
            context=f"Task: {state['task'][:100]}...",
        )
        if error:
            return {
                "error": error,
                "current_phase": ResearchPhase.COMPLETE.value,
            }
    
    # Log the research plan details
    logger.success(f"Research plan created with {len(plan.get('initial_queries', []))} initial queries")
    
    # Log LLM response to reasoning file
    writer.log_llm_response(
        phase="planning",
        prompt_summary=f"Create research plan for: {state['task']}",
        response=json.dumps(plan, indent=2),
    )
    
    # Show plan details
    plan_summary = []
    if plan.get("objectives"):
        plan_summary.append(f"[bold]Objectives:[/bold] {len(plan['objectives'])}")
        for obj in plan["objectives"]:
            plan_summary.append(f"  • {obj}")
    if plan.get("regions_of_interest"):
        plan_summary.append(f"[bold]Regions:[/bold] {', '.join(plan['regions_of_interest'])}")
    if plan.get("keywords"):
        plan_summary.append(f"[bold]Keywords:[/bold] {', '.join(plan['keywords'])}")
    if plan.get("priority_sources"):
        plan_summary.append(f"[bold]Data Sources:[/bold] {', '.join(plan['priority_sources'])}")
    if plan.get("initial_queries"):
        plan_summary.append(f"[bold]Planned Queries:[/bold]")
        for q in plan["initial_queries"]:
            plan_summary.append(f"  • {q.get('tool')}({', '.join(f'{k}={v}' for k,v in q.get('args', {}).items())})")
    
    logger.panel("\n".join(plan_summary), title="📋 Research Plan", style="cyan")
    
    # Add reasoning step for planning
    reasoning_step = _add_reasoning_step(
        state,
        phase="planning",
        thought=f"Planning research approach for: {state['task']}",
        action="Created comprehensive research plan",
        observation=f"Planned {len(plan.get('initial_queries', []))} initial queries across {len(plan.get('priority_sources', []))} sources",
        conclusion="Ready to proceed with task decomposition",
        confidence=0.8,
    )
    
    # Start with task decomposition for multi-step reasoning
    # This allows the agent to break down complex tasks before gathering
    return {
        "research_plan": plan,
        "pending_queries": plan.get("initial_queries", []),
        "current_phase": ResearchPhase.DECOMPOSING.value,  # Start with decomposition
        "iteration": state["iteration"] + 1,
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        "last_updated": datetime.utcnow().isoformat(),
        "messages": [
            HumanMessage(content=state["task"]),
            AIMessage(content=f"Research plan created: {json.dumps(plan, indent=2)}"),
        ],
    }


# =============================================================================
# Data Gathering Node
# =============================================================================


async def gather_intelligence(
    state: AgentState,
    tool_executor: MCPToolExecutor,
) -> dict[str, Any]:
    """
    Execute pending queries to gather intelligence.
    
    This node executes the planned queries against the MCP server
    and collects the results as findings.
    """
    log_agent_step(state["iteration"] + 1, "Gathering intelligence data")
    
    # Log to reasoning file
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
        "detect_thermal_anomalies": ["latitude", "longitude"],
        "check_connectivity": ["country_code"],
        "check_traffic_metrics": ["country_code"],
        "search_sanctions": ["query"],
        "search_telegram": ["keywords"],
        "screen_entity": ["name"],
    }
    
    for query in queries_to_run:
        tool_name = query.get("tool")
        args = query.get("args", {})
        
        # Resolve tool name to handle aliases (e.g., search_news_by_location -> search_news)
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
        
        # Process results into findings using resolved name for proper extraction
        if result.get("status") == "success" or result.get("status") == "stub":
            findings = _extract_findings(resolved_tool_name, args, result)
            new_findings.extend(findings)
            logger.success(f"Extracted {len(findings)} findings from {tool_name}")
            
            # Log to reasoning file
            writer.log_tool_execution(
                tool_name=tool_name,
                args=args,
                result_summary=f"Extracted {len(findings)} findings",
                success=True,
            )
            
            # Log brief result summary using resolved name
            _log_tool_result_summary(resolved_tool_name, result)
        else:
            error_msg = result.get("error_message") or result.get("error", "Unknown error")
            logger.warning(f"Query returned error: {error_msg}")
            
            # Log failed tool execution
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
    
    return {
        "findings": all_findings,
        "executed_queries": all_executed,
        "pending_queries": remaining,
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


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
    
    elif tool_name in ["search_sanctions", "screen_entity"]:
        entities = result.get("entities", [])
        if entities:
            summary_lines.append(f"[bold]Found {len(entities)} sanctioned entities[/bold]")
            for e in entities:
                summary_lines.append(f"  • {e.get('name')} ({e.get('entity_type')})")
    
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
            findings.append({
                "source": "IODA",
                "source_type": IntelligenceType.CYBER.value,
                "timestamp": timestamp,
                "content": {
                    "region": outage.get("region"),
                    "severity": outage.get("severity"),
                    "start_time": outage.get("start_time"),
                },
                "relevance_score": 0.9,
                "confidence": "high",
            })
    
    elif tool_name == "search_sanctions":
        entities = result.get("entities", [])
        for entity in entities:
            findings.append({
                "source": "OpenSanctions",
                "source_type": IntelligenceType.SANCTIONS.value,
                "timestamp": timestamp,
                "content": {
                    "name": entity.get("name"),
                    "entity_type": entity.get("entity_type"),
                    "countries": entity.get("countries"),
                    "sanctions_lists": entity.get("sanctions_lists"),
                },
                "relevance_score": entity.get("match_score", 0.8),
                "confidence": "high" if entity.get("match_score", 0) > 0.9 else "medium",
            })
    
    return findings


# =============================================================================
# Analysis Node
# =============================================================================


async def analyze_findings(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Analyze collected findings and extract insights.
    
    This node uses the LLM to analyze the gathered intelligence
    and identify key patterns and insights. Uses Chain-of-Thought
    reasoning for deeper analysis.
    """
    log_agent_step(state["iteration"] + 1, "Analyzing findings")
    
    # Log to reasoning file
    writer = get_output_writer()
    writer.log_reasoning(
        phase="analyzing",
        step=state["iteration"] + 1,
        action="Analyzing collected findings with multi-step reasoning",
        details={
            "findings_count": len(state.get("findings", [])),
            "hypotheses_count": len(state.get("hypotheses", [])),
        },
    )
    
    findings = state.get("findings", [])
    if not findings:
        logger.warning("No findings to analyze")
        return {
            "current_phase": ResearchPhase.SYNTHESIZING.value,
            "iteration": state["iteration"] + 1,
        }
    
    # Get available tools for the prompt
    available_tools = get_tool_definitions()
    tool_names = [t["name"] for t in available_tools]
    
    # Use enhanced analyst prompt for multi-step reasoning
    analysis_prompt = f"""
{ENHANCED_ANALYST_PROMPT}

## Original Task
{state["task"]}

## Current Hypotheses
{json.dumps(state.get("hypotheses", []), indent=2) if state.get("hypotheses") else "No hypotheses generated yet."}

## Collected Findings ({len(findings)} total)
{json.dumps(findings[:30], indent=2)}  # Limit to 30 for context window

## Available Tools for Follow-up Queries
You can ONLY use these tool names for follow_up_queries:
{json.dumps(tool_names, indent=2)}

Tool details:
{json.dumps(available_tools, indent=2)}

## Instructions
Think step by step:
1. First, survey all the evidence - what types of data do we have?
2. What patterns emerge from individual sources?
3. How does this evidence relate to our hypotheses?
4. What areas still need investigation?

Then provide your analysis with:
1. Key patterns observed (with evidence citations)
2. Notable events or anomalies
3. Preliminary correlations between different data sources
4. How findings support/refute each hypothesis
5. Confidence assessment for each insight
6. Areas requiring further investigation

IMPORTANT: For follow_up_queries, you MUST use one of the exact tool names listed above.
Do NOT use data source names like "GDELT" or "IODA" - use the actual tool names like "search_news" or "check_connectivity".

Respond as JSON:
{{
    "thinking": "Your step-by-step reasoning process...",
    "key_insights": ["insight 1", "insight 2"],
    "hypothesis_implications": [
        {{"hypothesis_id": "h1", "evidence_type": "supporting|contradicting|neutral", "explanation": "..."}}
    ],
    "correlations": [
        {{
            "finding_ids": [0, 5],
            "correlation_type": "temporal",
            "description": "...",
            "confidence": "high",
            "implications": ["..."]
        }}
    ],
    "uncertainties": ["uncertainty 1"],
    "follow_up_queries": [
        {{"tool": "search_news", "args": {{"keywords": "...", "country_code": "..."}}, "reason": "..."}}
    ]
}}
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=analysis_prompt),
    ]
    
    # Use structured output for reliable JSON
    thinking = ""
    try:
        structured_llm = llm.with_structured_output(AnalysisOutput)
        output: AnalysisOutput = await structured_llm.ainvoke(messages)
        analysis = output.model_dump()
        thinking = analysis.get("thinking", "")
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        thinking, _ = _extract_thinking_block(response.content)
        analysis, error = _extract_json_with_logging(
            response.content,
            phase="analyzing",
            context=f"Analyzing {len(findings)} findings",
        )
        if error:
            return {
                "current_phase": ResearchPhase.SYNTHESIZING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
        analysis_thinking = analysis.get("thinking", "")
        if not thinking and analysis_thinking:
            thinking = analysis_thinking
    
    # Check if follow-up queries are needed
    follow_up = analysis.get("follow_up_queries", [])
    # Use REFLECTING phase for multi-step reasoning instead of going directly to CORRELATING
    next_phase = ResearchPhase.GATHERING.value if follow_up else ResearchPhase.REFLECTING.value
    
    logger.success(f"Analysis complete: {len(analysis.get('key_insights', []))} insights")
    
    # Log LLM response to reasoning file
    writer.log_llm_response(
        phase="analyzing",
        prompt_summary=f"Analyze {len(state.get('findings', []))} findings",
        response=json.dumps(analysis, indent=2),
    )
    
    # Log thinking/reasoning chain
    if thinking:
        writer.log_reasoning(
            phase="analyzing",
            step=state["iteration"] + 1,
            action="Chain of thought reasoning",
            details={"thinking": thinking},
        )
    
    # Add reasoning step
    reasoning_step = _add_reasoning_step(
        state,
        phase="analyzing",
        thought=thinking if thinking else "Analyzing patterns in collected evidence",
        action=f"Identified {len(analysis.get('key_insights', []))} insights",
        observation=f"Found {len(analysis.get('correlations', []))} correlations, {len(analysis.get('uncertainties', []))} uncertainties",
        conclusion="Analysis complete, proceeding to reflection" if not follow_up else f"Need {len(follow_up)} more queries",
        confidence=0.75,
    )
    
    # Log analysis details
    analysis_summary = []
    if analysis.get("key_insights"):
        analysis_summary.append("[bold]Key Insights:[/bold]")
        for insight in analysis["key_insights"]:
            if isinstance(insight, dict):
                analysis_summary.append(f"  • {insight.get('description', str(insight))}")
            else:
                analysis_summary.append(f"  • {insight}")
    
    # Show hypothesis implications if present
    if analysis.get("hypothesis_implications"):
        analysis_summary.append(f"[bold]Hypothesis Implications:[/bold]")
        for hi in analysis["hypothesis_implications"]:
            emoji = {"supporting": "✅", "contradicting": "❌", "neutral": "➖"}.get(hi.get("evidence_type", "neutral"), "➖")
            analysis_summary.append(f"  {emoji} [{hi.get('hypothesis_id')}] {hi.get('explanation', '')}")
    
    if analysis.get("correlations"):
        analysis_summary.append(f"[bold]Correlations Found:[/bold] {len(analysis['correlations'])}")
    if analysis.get("uncertainties"):
        analysis_summary.append(f"[bold]Uncertainties:[/bold] {len(analysis['uncertainties'])}")
    if follow_up:
        analysis_summary.append(f"[bold]Follow-up Queries:[/bold] {len(follow_up)}")
        for q in follow_up:
            analysis_summary.append(f"  • {q.get('tool')}: {q.get('reason', 'No reason provided')}")
    
    if analysis_summary:
        logger.panel("\n".join(analysis_summary), title="🔍 Analysis Results", style="yellow")
    
    return {
        "key_insights": state.get("key_insights", []) + analysis.get("key_insights", []),
        "correlations": state.get("correlations", []) + analysis.get("correlations", []),
        "uncertainties": state.get("uncertainties", []) + analysis.get("uncertainties", []),
        "pending_queries": follow_up[:3],  # Limit follow-ups
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        "chain_of_thought": state.get("chain_of_thought", []) + ([thinking] if thinking else []),
        "current_phase": next_phase,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
        "messages": state["messages"] + [
            AIMessage(content=f"Analysis: {json.dumps(analysis, indent=2)}"),
        ],
    }


# =============================================================================
# Correlation Node  
# =============================================================================


async def correlate_findings(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Find correlations between findings from different sources.
    
    This node specifically looks for connections between news,
    satellite data, and connectivity information.
    """
    log_agent_step(state["iteration"] + 1, "Correlating multi-source data")
    
    # Log to reasoning file
    writer = get_output_writer()
    writer.log_reasoning(
        phase="correlating",
        step=state["iteration"] + 1,
        action="Finding correlations between multi-source data",
        details={"findings_count": len(state.get("findings", []))},
    )
    
    findings = state.get("findings", [])
    
    # Group findings by type
    by_type = {}
    for i, f in enumerate(findings):
        ftype = f.get("source_type", "unknown")
        if ftype not in by_type:
            by_type[ftype] = []
        by_type[ftype].append({"index": i, **f})
    
    correlation_prompt = f"""
You are correlating intelligence from multiple sources to find connections.

## Findings by Source Type
{json.dumps(by_type, indent=2)}

## Task Context
{state["task"]}

Look for:
1. **Temporal correlations**: Events happening around the same time
2. **Geospatial correlations**: Events in the same location from different sources
3. **Causal correlations**: One event potentially causing another
4. **Pattern correlations**: Similar patterns across different data types

For each correlation found, explain the connection and its implications.

Respond as JSON:
{{
    "correlations": [
        {{
            "finding_ids": [1, 15, 22],
            "correlation_type": "geospatial-temporal",
            "description": "Thermal anomalies detected at X coincide with news reports of Y",
            "confidence": "high",
            "implications": ["Military activity likely", "Infrastructure damage probable"]
        }}
    ],
    "synthesis_notes": "Overall pattern suggests..."
}}
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=correlation_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(CorrelationOutput)
        output: CorrelationOutput = await structured_llm.ainvoke(messages)
        correlations = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        correlations, error = _extract_json_with_logging(
            response.content,
            phase="correlating",
            context=f"Correlating {len(findings)} findings by type: {list(by_type.keys())}",
        )
        if error:
            return {
                "current_phase": ResearchPhase.VERIFYING.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    logger.success(f"Found {len(correlations.get('correlations', []))} correlations")
    
    # Log LLM response to reasoning file
    writer.log_llm_response(
        phase="correlating",
        prompt_summary="Find correlations between multi-source intelligence",
        response=json.dumps(correlations, indent=2),
    )
    
    # Log correlation details
    corr_summary = []
    for corr in correlations.get("correlations", []):
        corr_type = corr.get("correlation_type", "unknown")
        desc = corr.get("description", "No description")
        confidence = corr.get("confidence", "unknown")
        corr_summary.append(f"[bold]{corr_type}[/bold] ({confidence})")
        corr_summary.append(f"  {desc}")
    
    if correlations.get("synthesis_notes"):
        corr_summary.append(f"\n[bold]Synthesis:[/bold] {correlations['synthesis_notes']}")
    
    if corr_summary:
        logger.panel("\n".join(corr_summary), title="🔗 Correlations Found", style="magenta")
    
    # Add reasoning step for correlation
    reasoning_step = _add_reasoning_step(
        state,
        phase="correlating",
        thought="Looking for connections across different data sources",
        action=f"Found {len(correlations.get('correlations', []))} correlations",
        observation=correlations.get("synthesis_notes", "No synthesis notes"),
        conclusion="Ready for verification before final synthesis",
        confidence=0.8,
    )
    
    return {
        "correlations": state.get("correlations", []) + correlations.get("correlations", []),
        "key_insights": state.get("key_insights", []) + [correlations.get("synthesis_notes", "")],
        "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
        "reasoning_depth": state.get("reasoning_depth", 0) + 1,
        "current_phase": ResearchPhase.VERIFYING.value,  # Go to verification before synthesis
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Synthesis Node
# =============================================================================


async def synthesize_report(
    state: AgentState,
    llm: BaseChatModel,
) -> dict[str, Any]:
    """
    Synthesize all findings into a final intelligence report.
    
    This node creates the executive summary, detailed report,
    and recommendations. Uses verified insights and correlations
    from the multi-step reasoning process.
    """
    log_agent_step(state["iteration"] + 1, "Synthesizing final report")
    
    # Log to reasoning file
    writer = get_output_writer()
    
    # Use verified data if available, otherwise fall back to original
    insights_to_use = state.get("verified_insights") or state.get("key_insights", [])
    correlations_to_use = state.get("verified_correlations") or state.get("correlations", [])
    
    writer.log_reasoning(
        phase="synthesizing",
        step=state["iteration"] + 1,
        action="Creating final intelligence report from verified findings",
        details={
            "verified_insights_count": len(state.get("verified_insights", [])),
            "verified_correlations_count": len(state.get("verified_correlations", [])),
            "hypotheses_count": len(state.get("hypotheses", [])),
            "reasoning_depth": state.get("reasoning_depth", 0),
            "reflection_iterations": state.get("reflection_iterations", 0),
        },
    )
    
    # Build hypothesis summary
    hypotheses_summary = ""
    if state.get("hypotheses"):
        hyp_lines = []
        for h in state["hypotheses"]:
            status_emoji = {
                "supported": "✅",
                "refuted": "❌",
                "investigating": "🔍",
                "inconclusive": "❓",
                "proposed": "📝",
            }.get(h.get("status", "proposed"), "📝")
            hyp_lines.append(f"- {status_emoji} **{h['id']}**: {h['statement']} (confidence: {h.get('confidence', 0):.0%})")
        hypotheses_summary = "\n## Hypothesis Results\n" + "\n".join(hyp_lines)
    
    # Build reflection summary
    reflection_summary = ""
    if state.get("reflection_notes"):
        critical_notes = [n for n in state["reflection_notes"] if n.get("severity") == "critical"]
        if critical_notes:
            reflection_summary = "\n## Critical Reflection Notes\n" + "\n".join([f"- {n['content']}" for n in critical_notes])
    
    # Prepare findings with source information for citation
    findings_for_citation = []
    for i, finding in enumerate(state.get("findings", [])):
        findings_for_citation.append({
            "finding_id": i + 1,
            "source": finding.get("source"),
            "source_type": finding.get("source_type"),
            "timestamp": finding.get("timestamp"),
            "content": finding.get("content"),
            "location": finding.get("location"),
        })
    
    # Prepare executed queries for reference
    executed_queries_summary = []
    for q in state.get("executed_queries", []):
        executed_queries_summary.append({
            "tool": q.get("tool"),
            "args": q.get("args"),
            "timestamp": q.get("timestamp"),
        })
    
    synthesis_prompt = f"""
{ENHANCED_SYNTHESIZER_PROMPT}

## Original Task
{state["task"]}

## Research Plan
{json.dumps(state.get("research_plan", {}), indent=2)}
{hypotheses_summary}

## Verified Key Insights
{json.dumps(insights_to_use, indent=2)}

## Verified Correlations
{json.dumps(correlations_to_use, indent=2)}

## Uncertainties
{json.dumps(state.get("uncertainties", []), indent=2)}
{reflection_summary}

## Reasoning Process Summary
- Reasoning depth: {state.get("reasoning_depth", 0)} steps
- Reflection iterations: {state.get("reflection_iterations", 0)}
- Verification results: {len(state.get("verification_results", []))} items verified

## Raw Findings (Sources for Citation)
These are the actual data points collected from tools. CITE THESE in your report.
{json.dumps(findings_for_citation, indent=2)}

## Queries Executed
{json.dumps(executed_queries_summary, indent=2)}

Create a comprehensive intelligence report as JSON. 
IMPORTANT: The detailed_report MUST include inline citations for every claim and a "## Sources" section at the end.

{{
    "executive_summary": "2-3 sentence summary of key findings",
    "detailed_report": "Multi-paragraph detailed analysis in markdown format WITH INLINE CITATIONS and a ## Sources section at the end",
    "recommendations": ["recommendation 1", "recommendation 2"],
    "confidence_assessment": "Overall confidence level and explanation",
    "methodology_note": "Brief note on the multi-step reasoning process used"
}}
"""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=synthesis_prompt),
    ]
    
    # Use structured output for reliable JSON
    try:
        structured_llm = llm.with_structured_output(SynthesisOutput)
        output: SynthesisOutput = await structured_llm.ainvoke(messages)
        report = output.model_dump()
    except Exception as e:
        logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
        response = await llm.ainvoke(messages)
        report, error = _extract_json_with_logging(
            response.content,
            phase="synthesizing",
            context=f"Synthesizing report with {len(insights_to_use)} insights and {len(correlations_to_use)} correlations",
        )
        if error:
            logger.warning(f"⚠️ JSON parsing failed, using raw response as report")
            return {
                "detailed_report": response.content,
                "current_phase": ResearchPhase.COMPLETE.value,
                "iteration": state["iteration"] + 1,
                "error": error,
            }
    
    logger.success("Final report synthesized")
    
    # Log LLM response to reasoning file
    writer.log_llm_response(
        phase="synthesizing",
        prompt_summary="Synthesize final intelligence report",
        response=json.dumps(report, indent=2),
    )
    
    # Log the executive summary
    if report.get("executive_summary"):
        logger.panel(
            report["executive_summary"],
            title="📊 Executive Summary",
            style="green"
        )
    
    # Log recommendations
    if report.get("recommendations"):
        rec_lines = ["[bold]Recommendations:[/bold]"]
        for rec in report["recommendations"]:
            rec_lines.append(f"  • {rec}")
        logger.panel("\n".join(rec_lines), title="💡 Recommendations", style="cyan")
    
    # Log confidence assessment
    if report.get("confidence_assessment"):
        logger.thinking(f"Confidence: {report['confidence_assessment']}")
    
    return {
        "executive_summary": report.get("executive_summary"),
        "detailed_report": report.get("detailed_report"),
        "recommendations": report.get("recommendations", []),
        "confidence_assessment": report.get("confidence_assessment"),
        "current_phase": ResearchPhase.COMPLETE.value,
        "iteration": state["iteration"] + 1,
        "last_updated": datetime.utcnow().isoformat(),
    }


# =============================================================================
# Router Function
# =============================================================================


def route_next_step(state: AgentState) -> Literal[
    "decompose", "hypothesize", "gather", "analyze", "reflect", "correlate", "verify", "synthesize", "end"
]:
    """
    Determine the next node based on current state.
    
    This is used as a conditional edge in the graph.
    Supports both traditional and multi-step reasoning phases.
    """
    phase = state.get("current_phase", ResearchPhase.PLANNING.value)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", settings.agent_max_iterations)
    
    # Safety check for max iterations
    if iteration >= max_iter:
        logger.warning(f"Max iterations ({max_iter}) reached, forcing synthesis")
        return "synthesize"
    
    # Check for errors
    if state.get("error"):
        return "end"
    
    # Route based on phase (including new multi-step reasoning phases)
    if phase == ResearchPhase.DECOMPOSING.value:
        return "decompose"
    elif phase == ResearchPhase.HYPOTHESIZING.value:
        return "hypothesize"
    elif phase == ResearchPhase.GATHERING.value:
        return "gather"
    elif phase == ResearchPhase.ANALYZING.value:
        return "analyze"
    elif phase == ResearchPhase.REFLECTING.value:
        return "reflect"
    elif phase == ResearchPhase.CORRELATING.value:
        return "correlate"
    elif phase == ResearchPhase.VERIFYING.value:
        return "verify"
    elif phase == ResearchPhase.SYNTHESIZING.value:
        return "synthesize"
    elif phase == ResearchPhase.COMPLETE.value:
        return "end"
    else:
        return "end"
