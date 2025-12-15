"""
Planning Node - Creates research plan based on task.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import ResearchPlanOutput
from src.agent.prompts import PLANNER_PROMPT
from src.agent.tools import get_tool_definitions
from src.shared.logger import get_logger

logger = get_logger()


class PlanningNode(BaseNode):
    """Node that plans the research approach."""
    
    def get_phase_name(self) -> str:
        return "planning"
    
    def get_prompt(self, state: AgentState) -> str:
        return f"""
{PLANNER_PROMPT}

## Task
{state["task"]}

## Context
{json.dumps(state["context"], indent=2) if state["context"] else "No additional context provided."}

## Available Tools
{json.dumps(get_tool_definitions(), indent=2)}

Create a comprehensive research plan based on the task and available tools.
"""
    
    def get_output_schema(self) -> type:
        return ResearchPlanOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        plan = output
        
        # Log the research plan details
        logger.success(f"Research plan created with {len(plan.get('initial_queries', []))} initial queries")
        
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
        
        # Add reasoning step
        reasoning_step = self._add_reasoning_step(
            state,
            phase="planning",
            thought=f"Planning research approach for: {state['task']}",
            action="Created comprehensive research plan",
            observation=f"Planned {len(plan.get('initial_queries', []))} initial queries across {len(plan.get('priority_sources', []))} sources",
            conclusion="Ready to proceed with task decomposition",
            confidence=0.8,
        )
        
        return {
            "research_plan": plan,
            "pending_queries": plan.get("initial_queries", []),
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "reasoning_depth": state.get("reasoning_depth", 0) + 1,
            "messages": [
                HumanMessage(content=state["task"]),
                AIMessage(content=f"Research plan created: {json.dumps(plan, indent=2)}"),
            ],
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        # Start with task decomposition for multi-step reasoning
        return ResearchPhase.DECOMPOSING.value

