"""
Verification Node - Verifies consistency and validity of conclusions.
"""

import json
from typing import Any

from src.agent.nodes.base import BaseNode
from src.agent.state import AgentState, ResearchPhase
from src.agent.schemas import VerificationOutput
from src.agent.prompts import VERIFICATION_PROMPT
from src.shared.logger import get_logger

logger = get_logger()


class VerificationNode(BaseNode):
    """Node that verifies consistency and validity of conclusions."""
    
    def get_phase_name(self) -> str:
        return "verifying"
    
    def get_prompt(self, state: AgentState) -> str:
        return f"""
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
    
    def get_output_schema(self) -> type:
        return VerificationOutput
    
    def process_output(
        self,
        state: AgentState,
        output: dict[str, Any],
        thinking: str = "",
    ) -> dict[str, Any]:
        result = output
        
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
                corr_idx = cv.get("correlation_index", 0)
                correlations = state.get("correlations", [])
                if corr_idx < len(correlations):
                    verified_corr = correlations[corr_idx].copy()
                    verified_corr["confidence"] = cv.get("suggested_confidence", verified_corr.get("confidence", "medium"))
                    verified_correlations.append(verified_corr)
        
        # Add reasoning step
        verification_chain = result.get("verification_chain", [])
        overall = result.get("overall_assessment", {})
        
        reasoning_step = self._add_reasoning_step(
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
        
        return {
            "verification_results": state.get("verification_results", []) + verification_results,
            "verified_insights": verified_insights if verified_insights else state.get("key_insights", []),
            "verified_correlations": verified_correlations if verified_correlations else state.get("correlations", []),
            "reasoning_trace": state.get("reasoning_trace", []) + [reasoning_step],
            "chain_of_thought": state.get("chain_of_thought", []) + verification_chain,
        }
    
    def get_next_phase(self, state: AgentState, output: dict[str, Any]) -> str:
        overall = output.get("overall_assessment", {})
        if overall.get("ready_for_synthesis", True):
            return ResearchPhase.SYNTHESIZING.value
        return ResearchPhase.REFLECTING.value

