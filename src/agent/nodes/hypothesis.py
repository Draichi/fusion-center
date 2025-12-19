"""
Hypothesis Nodes - Generate and update hypotheses.
"""

import json
from datetime import datetime
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase, HypothesisStatus
from src.agent.schemas import HypothesisGenerationOutput, HypothesisUpdateOutput
from src.agent.prompts import HYPOTHESIS_GENERATION_PROMPT, HYPOTHESIS_UPDATE_PROMPT
from src.agent.tools import get_tool_definitions
from src.shared.logger import get_logger

logger = get_logger()


class HypothesisGenerationNode(BaseNode):
    """Node that generates testable hypotheses."""
    
    def get_phase_name(self) -> str:
        return "hypothesizing"
    
    def get_node_type(self) -> str:
        return "structured"
    
    def get_prompt(self, state: AgentState) -> str:
        # Build context from existing findings if any
        findings_summary = ""
        if state.get("findings"):
            findings_summary = f"\n\n## Existing Findings ({len(state['findings'])} total)\n{json.dumps(state['findings'][:10], indent=2)}"
        
        return f"""
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
    
    def get_output_schema(self) -> type:
        return HypothesisGenerationOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        result = output
        
        # Build hypotheses
        hypotheses = []
        test_queries = []
        
        for h in result.get("hypotheses", []):
            # Auto-normalize confidence if LLM returned 0-100 instead of 0-1
            initial_conf = h.get("initial_confidence", 0.5)
            if initial_conf > 1:
                normalized_conf = initial_conf / 100.0
                logger.warning(
                    f"⚠️  Auto-normalized initial_confidence for {h.get('id', 'hypothesis')}: "
                    f"{initial_conf} → {normalized_conf:.2f} (LLM returned percentage instead of decimal)"
                )
                initial_conf = normalized_conf
            
            # Clamp to valid range [0, 1] as defensive measure
            initial_conf = max(0.0, min(1.0, initial_conf))
            
            hypothesis = {
                "id": h.get("id", f"h{len(hypotheses) + 1}"),
                "statement": h.get("statement", ""),
                "status": HypothesisStatus.PROPOSED.value,
                "confidence": initial_conf,
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
        
        # Log reasoning chain
        reasoning_chain = result.get("reasoning_chain", [])
        if reasoning_chain:
            self.writer.log_reasoning(
                phase="hypothesizing",
                step=state["iteration"] + 1,
                action="Reasoning chain",
                details={"chain": reasoning_chain},
            )
        
        # Add reasoning step
        reasoning_step = self._add_reasoning_step(
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
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        return ResearchPhase.GATHERING.value


class HypothesisUpdateNode(BaseNode):
    """Node that updates hypothesis confidence based on new evidence."""
    
    def get_phase_name(self) -> str:
        return "hypothesis_update"
    
    def get_node_type(self) -> str:
        return "structured"
    
    def get_prompt(self, state: AgentState) -> str:
        return f"""
{HYPOTHESIS_UPDATE_PROMPT}

## Current Hypotheses
{json.dumps(state.get("hypotheses", []), indent=2)}

## New Findings
{json.dumps(state.get("findings", [])[-20:], indent=2)}  # Last 20 findings

Review each finding and update hypothesis confidence accordingly.
"""
    
    def get_output_schema(self) -> type:
        return HypothesisUpdateOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        result = output
        
        hypotheses = state.get("hypotheses", [])
        if not hypotheses:
            logger.info("No hypotheses to update")
            return {}
        
        # Update hypotheses
        updated_hypotheses = {h["id"]: h.copy() for h in hypotheses}
        
        for update in result.get("hypothesis_updates", []):
            h_id = update.get("hypothesis_id")
            if h_id in updated_hypotheses:
                h = updated_hypotheses[h_id]
                
                # Auto-normalize confidence if LLM returned 0-100 instead of 0-1
                new_conf = update.get("new_confidence", h["confidence"])
                if new_conf > 1:
                    normalized_conf = new_conf / 100.0
                    logger.warning(
                        f"⚠️  Auto-normalized confidence for {h_id}: "
                        f"{new_conf} → {normalized_conf:.2f} (LLM returned percentage instead of decimal)"
                    )
                    new_conf = normalized_conf
                
                # Clamp to valid range [0, 1] as defensive measure
                h["confidence"] = max(0.0, min(1.0, new_conf))
                h["status"] = update.get("new_status", h["status"])
                h["supporting_evidence"].extend(update.get("new_supporting_evidence", []))
                h["contradicting_evidence"].extend(update.get("new_contradicting_evidence", []))
                h["reasoning"] = update.get("confidence_change_reason", "")
                h["updated_at"] = datetime.utcnow().isoformat()
        
        updated_list = list(updated_hypotheses.values())
        
        # Add reasoning step
        reasoning_chain = result.get("reasoning_chain", [])
        reasoning_step = self._add_reasoning_step(
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
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        return ResearchPhase.ANALYZING.value

