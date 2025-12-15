"""
Base Node Class with Template Method Pattern.

Provides common functionality for all research nodes:
- JSON parsing with fallback
- Logging and reasoning trace
- Structured output handling
- Error handling
"""

import json
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agent.state import AgentState
from src.agent.prompts import SYSTEM_PROMPT
from src.shared.logger import get_logger, log_agent_step
from src.shared.output_writer import get_output_writer

logger = get_logger()


class BaseNode(ABC):
    """
    Base class for all research nodes using Template Method pattern.
    
    Each node implements:
    - `get_prompt()`: Build the prompt for this node
    - `get_output_schema()`: Return the Pydantic schema for structured output
    - `process_output()`: Process the parsed output into state updates
    - `get_next_phase()`: Determine the next phase after this node
    
    Common functionality (parsing, logging, error handling) is handled by the template method.
    """
    
    def __init__(self, llm: BaseChatModel):
        """Initialize the node with an LLM instance."""
        self.llm = llm
        self.writer = get_output_writer()
    
    def _extract_thinking_block(self, response_text: str) -> tuple[str, str]:
        """Extract thinking block from response if present."""
        thinking = ""
        main_response = response_text
        
        if "```thinking" in response_text:
            parts = response_text.split("```thinking")
            if len(parts) > 1:
                thinking_part = parts[1].split("```")[0]
                thinking = thinking_part.strip()
                remaining = "```".join(parts[1].split("```")[1:])
                main_response = parts[0] + remaining
        
        return thinking, main_response.strip()
    
    def _extract_json_with_logging(
        self,
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
        # Extract thinking block first
        thinking, main_response = self._extract_thinking_block(response_content)
        
        # Log raw response for debugging
        response_preview = response_content[:500] + "..." if len(response_content) > 500 else response_content
        
        # Try to extract JSON from code blocks
        json_str = main_response
        if "```json" in main_response:
            json_str = main_response.split("```json")[1].split("```")[0]
        elif "```" in main_response:
            parts = main_response.split("```")
            if len(parts) >= 2:
                json_str = parts[1].split("```")[0]
        
        json_str = json_str.strip()
        
        # Check if we have any content to parse
        if not json_str:
            error_msg = f"Empty response from LLM (phase: {phase})"
            logger.error(f"❌ {error_msg}")
            logger.error(f"   Raw response preview: {response_preview}")
            self.writer.log_reasoning(
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
            
            self.writer.log_reasoning(
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
    
    def _add_reasoning_step(
        self,
        state: AgentState,
        phase: str,
        thought: str,
        action: str,
        observation: str,
        conclusion: str,
        confidence: float,
    ) -> dict[str, Any]:
        """Add a reasoning step to the trace."""
        from datetime import datetime
        
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
    
    @abstractmethod
    def get_phase_name(self) -> str:
        """Return the phase name for this node (e.g., 'planning', 'analyzing')."""
        pass
    
    @abstractmethod
    def get_prompt(self, state: AgentState) -> str:
        """Build the prompt for this node based on the current state."""
        pass
    
    @abstractmethod
    def get_output_schema(self) -> type[BaseModel]:
        """Return the Pydantic schema class for structured output."""
        pass
    
    @abstractmethod
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        """
        Process the parsed output into state updates.
        
        Args:
            state: Current agent state
            output: Parsed output from LLM (dict representation of schema)
            thinking: Extracted thinking/reasoning chain if any
            
        Returns:
            Dictionary of state updates
        """
        pass
    
    @abstractmethod
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        """
        Determine the next phase after this node completes.
        
        Args:
            state: Current agent state
            output: Processed output from this node
            
        Returns:
            Next phase name (ResearchPhase value)
        """
        pass
    
    async def execute(self, state: AgentState) -> dict[str, Any]:
        """
        Template method that orchestrates the node execution.
        
        This method handles:
        1. Building the prompt
        2. Calling the LLM with structured output
        3. Parsing the response (with fallback to JSON extraction)
        4. Processing the output
        5. Logging and error handling
        
        Subclasses only need to implement the abstract methods.
        """
        phase_name = self.get_phase_name()
        log_agent_step(state["iteration"] + 1, f"Executing {phase_name} node")
        
        # Log reasoning start
        self.writer.log_reasoning(
            phase=phase_name,
            step=state["iteration"] + 1,
            action=f"Starting {phase_name}",
            details={"task": state.get("task", "")[:100]},
        )
        
        # Build prompt
        prompt = self.get_prompt(state)
        
        # Prepare messages
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        # Try structured output first, fallback to JSON extraction
        thinking = ""
        try:
            schema_class = self.get_output_schema()
            structured_llm = self.llm.with_structured_output(schema_class)
            output_model = await structured_llm.ainvoke(messages)
            output = output_model.model_dump()
        except Exception as e:
            logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
            response = await self.llm.ainvoke(messages)
            thinking, _ = self._extract_thinking_block(response.content)
            output, error = self._extract_json_with_logging(
                response.content,
                phase=phase_name,
                context=f"Task: {state.get('task', '')[:100]}...",
            )
            if error:
                return {
                    "current_phase": state.get("current_phase"),
                    "iteration": state["iteration"] + 1,
                    "error": error,
                }
        
        # Log LLM response
        self.writer.log_llm_response(
            phase=phase_name,
            prompt_summary=f"{phase_name}: {state.get('task', '')[:100]}",
            response=json.dumps(output, indent=2),
        )
        
        # Log thinking if present
        if thinking:
            self.writer.log_reasoning(
                phase=phase_name,
                step=state["iteration"] + 1,
                action="Chain of thought",
                details={"thinking": thinking},
            )
        
        # Process output into state updates
        updates = self.process_output(state, output, thinking)
        
        # Add common updates
        from datetime import datetime
        
        updates.update({
            "iteration": state["iteration"] + 1,
            "last_updated": datetime.utcnow().isoformat(),
        })
        
        return updates

