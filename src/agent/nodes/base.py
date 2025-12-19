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
    
    def __init__(self, thinking_llm: BaseChatModel, structured_llm: BaseChatModel):
        """Initialize the node with both LLM instances for dual-LLM architecture."""
        self.thinking_llm = thinking_llm
        self.structured_llm = structured_llm
        # Keep llm attribute for backward compatibility (default to structured)
        self.llm = structured_llm
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
    def get_node_type(self) -> str:
        """
        Return the node type to determine which LLM workflow to use.
        
        Returns:
            - "thinking": Uses thinking LLM first, then structured LLM for formatting (JSON output)
            - "structured": Uses only structured LLM with JSON mode
            - "thinking_markdown": Uses only thinking LLM to generate raw markdown (for synthesis)
        """
        pass
    
    def get_thinking_prompt(self, state: AgentState) -> list:
        """
        Build the free-form thinking prompt (for "thinking" nodes).
        Override in subclasses that use "thinking" mode.
        
        Returns:
            List of LangChain messages
        """
        # Default: return same as get_prompt()
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.agent.prompts import SYSTEM_PROMPT
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=self.get_prompt(state)),
        ]
    
    def get_formatting_prompt(self, state: AgentState, thinking_content: str) -> list:
        """
        Build the prompt to format thinking into structured output (for "thinking" nodes).
        Override in subclasses that use "thinking" mode.
        
        Args:
            state: Current agent state
            thinking_content: The free-form thinking output from thinking LLM
            
        Returns:
            List of LangChain messages
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from src.agent.prompts import SYSTEM_PROMPT
        
        formatting_instruction = f"""
Based on the following reasoning and the task context, format the output according to the required schema.

Reasoning:
{thinking_content}

Task: {state.get('task', '')}

Provide a structured JSON response following the schema.
"""
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=formatting_instruction),
        ]
    
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
        Template method that orchestrates the node execution with dual-LLM support.
        
        This method handles:
        1. Determining node type (thinking, structured, or thinking_markdown)
        2. Building the prompt(s)
        3. Calling the appropriate LLM(s)
        4. Parsing the response
        5. Processing the output
        6. Logging and error handling
        
        Subclasses only need to implement the abstract methods.
        """
        phase_name = self.get_phase_name()
        node_type = self.get_node_type()
        # Wrap entire execution in try/except to prevent unhandled errors
        try:
            log_agent_step(state["iteration"] + 1, f"Executing {phase_name} node ({node_type} mode)")
        
            # Log reasoning start
            self.writer.log_reasoning(
                phase=phase_name,
                step=state["iteration"] + 1,
                action=f"Starting {phase_name} ({node_type} mode)",
                details={"task": state.get("task", "")[:100]},
            )
        
            thinking = ""
        
            if node_type == "thinking":
                # Two-step process: thinking LLM → structured LLM
                logger.info(f"[{phase_name}] Step 1: Generating reasoning with thinking LLM")
            
                # Step 1: Get free-form thinking
                thinking_messages = self.get_thinking_prompt(state)
                thinking_response = await self.thinking_llm.ainvoke(thinking_messages)
                thinking = thinking_response.content
            
                # Log thinking
                self.writer.log_reasoning(
                    phase=phase_name,
                    step=state["iteration"] + 1,
                    action="Thinking (reasoning chain)",
                    details={"thinking": thinking[:500] + "..." if len(thinking) > 500 else thinking},
                )
            
                logger.info(f"[{phase_name}] Step 2: Formatting to JSON with structured LLM")
            
                # Step 2: Format thinking into structured output
                formatting_messages = self.get_formatting_prompt(state, thinking)
            
                try:
                    schema_class = self.get_output_schema()
                    structured_llm = self.structured_llm.with_structured_output(schema_class)
                    output_model = await structured_llm.ainvoke(formatting_messages)
                    output = output_model.model_dump()
                except Exception as e:
                    logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
                    
                    # Get schema for JSON example
                    schema_class = self.get_output_schema()
                    schema_json = schema_class.model_json_schema()
                    
                    # Extract the original formatting prompt
                    original_prompt = formatting_messages[-1].content if formatting_messages else ""
                    
                    # Create enhanced prompt with explicit JSON formatting instructions
                    json_instruction_prompt = f"""{original_prompt}

## CRITICAL: JSON Output Required

The structured output method failed. You MUST return ONLY valid JSON matching this exact schema.

**Schema:**
```json
{json.dumps(schema_json, indent=2)}
```

**IMPORTANT RULES:**
1. Return ONLY valid JSON - no markdown, no prose, no explanations
2. Wrap your JSON in ```json ``` code blocks if needed
3. All field names and types must match the schema exactly
4. For confidence values: use decimals 0.0-1.0, NOT percentages 0-100
   - ✅ CORRECT: "new_confidence": 0.75
   - ❌ WRONG: "new_confidence": 75

Respond now with valid JSON only:
"""
                    
                    # Re-invoke LLM with enhanced prompt
                    enhanced_messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=json_instruction_prompt),
                    ]
                    response = await self.structured_llm.ainvoke(enhanced_messages)
                    output, error = self._extract_json_with_logging(
                        response.content,
                        phase=phase_name,
                        context=f"Fallback JSON extraction for {phase_name} (thinking mode)",
                    )
                    if error:
                        return {
                            "current_phase": state.get("current_phase"),
                            "iteration": state["iteration"] + 1,
                            "error": error,
                        }
        
            elif node_type == "thinking_markdown":
                # Special case for synthesis: Generate markdown directly
                logger.info(f"[{phase_name}] Generating markdown report with thinking LLM")
            
                prompt = self.get_prompt(state)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            
                try:
                    response = await self.thinking_llm.ainvoke(messages)
                    markdown_content = response.content
                except Exception as e:
                    # Log detailed error information for debugging
                    error_str = str(e)
                    prompt_length = len(prompt)
                
                    logger.error(f"❌ Error generating markdown report with thinking LLM")
                    logger.error(f"   Error type: {type(e).__name__}")
                    logger.error(f"   Error message: {error_str[:500]}")
                    logger.error(f"   Prompt length: {prompt_length:,} characters")
                
                    # If prompt is too large, try with a simplified version
                    if "400" in error_str or "too large" in error_str.lower() or "context" in error_str.lower():
                        logger.warning(f"⚠️ Prompt appears too large ({prompt_length:,} chars), using fallback synthesis")
                    
                        # Create a minimal fallback report
                        findings_count = len(state.get("findings", []))
                        insights = state.get("key_insights", [])
                    
                        markdown_content = f"""# Intelligence Report

## Executive Summary
Research completed with {findings_count} findings collected.

## Key Insights
"""
                        for i, insight in enumerate(insights[:10], 1):
                            markdown_content += f"{i}. {insight}\n"
                    
                        markdown_content += f"""

## Data Collection Summary
- Total findings: {findings_count}
- Research iterations: {state.get('iteration', 0)}

*Note: Full report could not be generated due to context size limitations. Consider reviewing findings directly.*
"""
                        logger.info(f"   Generated fallback report ({len(markdown_content)} chars)")
                    else:
                        # Unknown error, re-raise it
                        logger.error(f"   Unknown error, re-raising...")
                        raise
            
                # Return markdown directly
                output = {"markdown_report": markdown_content}
            
                # Log the markdown generation
                self.writer.log_reasoning(
                    phase=phase_name,
                    step=state["iteration"] + 1,
                    action="Generated markdown report",
                    details={"length": len(markdown_content), "preview": markdown_content[:200] + "..."},
                )
        
            else:  # structured
                # Direct structured output with structured LLM
                logger.info(f"[{phase_name}] Generating structured output with structured LLM")
            
                prompt = self.get_prompt(state)
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            
                try:
                    schema_class = self.get_output_schema()
                    structured_llm = self.structured_llm.with_structured_output(schema_class)
                    output_model = await structured_llm.ainvoke(messages)
                    output = output_model.model_dump()
                except Exception as e:
                    logger.warning(f"⚠️ Structured output failed, falling back to JSON extraction: {e}")
                    
                    # Get schema for JSON example
                    schema_class = self.get_output_schema()
                    schema_json = schema_class.model_json_schema()
                    
                    # Create enhanced prompt with explicit JSON formatting instructions
                    json_instruction_prompt = f"""{prompt}

## CRITICAL: JSON Output Required

The structured output method failed. You MUST return ONLY valid JSON matching this exact schema.

**Schema:**
```json
{json.dumps(schema_json, indent=2)}
```

**IMPORTANT RULES:**
1. Return ONLY valid JSON - no markdown, no prose, no explanations
2. Wrap your JSON in ```json ``` code blocks if needed
3. All field names and types must match the schema exactly
4. For confidence values: use decimals 0.0-1.0, NOT percentages 0-100
   - ✅ CORRECT: "new_confidence": 0.75
   - ❌ WRONG: "new_confidence": 75

Respond now with valid JSON only:
"""
                    
                    # Re-invoke LLM with enhanced prompt
                    enhanced_messages = [
                        SystemMessage(content=SYSTEM_PROMPT),
                        HumanMessage(content=json_instruction_prompt),
                    ]
                    response = await self.structured_llm.ainvoke(enhanced_messages)
                    thinking, _ = self._extract_thinking_block(response.content)
                    output, error = self._extract_json_with_logging(
                        response.content,
                        phase=phase_name,
                        context=f"Fallback JSON extraction for {phase_name}",
                    )
                    if error:
                        return {
                            "current_phase": state.get("current_phase"),
                            "iteration": state["iteration"] + 1,
                            "error": error,
                        }
        
            # Log LLM response (skip for thinking_markdown which logs separately)
            if node_type != "thinking_markdown":
                self.writer.log_llm_response(
                    phase=phase_name,
                    prompt_summary=f"{phase_name}: {state.get('task', '')[:100]}",
                    response=json.dumps(output, indent=2),
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
        
            
        except Exception as e:
            # Top-level error handler - catch ANY uncaught error
            logger.error(f"❌ Critical error in {phase_name} node execution")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error: {str(e)[:500]}")
            
            import traceback
            logger.error(f"   Traceback:\n{traceback.format_exc()}")
            
            # Return error state instead of raising to prevent TaskGroup errors
            return {
                "current_phase": state.get("current_phase"),
                "iteration": state["iteration"] + 1,
                "error": f"{type(e).__name__}: {str(e)}",
            }


