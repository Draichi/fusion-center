"""
Decomposition Node - Breaks complex tasks into sub-tasks.
"""

import json
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import TaskDecompositionOutput
from src.agent.prompts import TASK_DECOMPOSITION_PROMPT
from src.shared.logger import get_logger

logger = get_logger()


class DecompositionNode(BaseNode):
    """Node that decomposes complex tasks into manageable sub-tasks."""
    
    def get_phase_name(self) -> str:
        return "decomposing"
    
    def get_node_type(self) -> str:
        return "structured"
    
    def get_prompt(self, state: AgentState) -> str:
        return f"""
{TASK_DECOMPOSITION_PROMPT}

## Research Task
{state["task"]}

## Context
{json.dumps(state["context"], indent=2) if state["context"] else "No additional context provided."}

Analyze this task and decompose it into sub-tasks.
"""
    
    def get_output_schema(self) -> type:
        return TaskDecompositionOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        result = output
        
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
        
        # Add reasoning step
        reasoning_step = self._add_reasoning_step(
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
        
        return {
            "sub_tasks": sub_tasks,
            "current_sub_task_id": sub_tasks[0]["id"] if sub_tasks else None,
            "task_complexity": task_complexity,
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "reasoning_depth": state.get("reasoning_depth", 0) + 1,
            "chain_of_thought": state.get("chain_of_thought", []) + ([thinking] if thinking else []),
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        task_complexity = output.get("task_complexity", "moderate")
        if task_complexity in ["moderate", "complex"]:
            return ResearchPhase.HYPOTHESIZING.value
        return ResearchPhase.PLANNING.value

