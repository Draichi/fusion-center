"""
Pattern Detection - ML-based pattern detection and LLM validation.
"""

from datetime import datetime, timedelta
from typing import Any

from langchain_core.language_models import BaseChatModel

from src.agent.graph import get_llm
from src.correlation_engine.schemas import Correlation
from src.shared.logger import get_logger

logger = get_logger()


class PatternDetector:
    """Detects complex patterns using ML and validates with LLM."""
    
    def __init__(self, llm: BaseChatModel | None = None):
        """Initialize pattern detector.
        
        Args:
            llm: Language model for validation (will create if not provided)
        """
        self.llm = llm or get_llm()
        logger.info("Pattern Detector initialized with LLM validation")
    
    async def validate_correlation(
        self,
        correlation: Correlation,
        events: list[dict[str, Any]]
    ) -> tuple[bool, str]:
        """Validate a correlation using LLM.
        
        Args:
            correlation: Correlation to validate
            events: Events involved in the correlation
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Build prompt for LLM
            prompt = self._build_validation_prompt(correlation, events)
            
            # Get LLM response
            response = await self.llm.ainvoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            is_valid = "yes" in response_text.lower() or "valid" in response_text.lower()
            reason = response_text
            
            logger.debug(f"LLM validation for {correlation.correlation_id}: {is_valid}")
            return is_valid, reason
            
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            # Default to valid if LLM fails
            return True, "LLM validation failed, defaulting to valid"
    
    def _build_validation_prompt(
        self,
        correlation: Correlation,
        events: list[dict[str, Any]]
    ) -> str:
        """Build prompt for LLM validation."""
        events_desc = "\n".join([
            f"- {e['source']} ({e['source_type']}): {e.get('attributes', {})}"
            for e in events[:5]  # Limit to first 5 events
        ])
        
        prompt = f"""Analyze if the following events are genuinely related and form a valid correlation.

Correlation Type: {correlation.correlation_type.value}
Proposed Description: {correlation.description}
Confidence: {correlation.confidence}

Events Involved:
{events_desc}

Question: Are these events genuinely related and does the correlation make sense?
Answer with "Yes" or "No" followed by a brief explanation.
Focus on whether there's a logical causal or contextual relationship between the events.
"""
        
        return prompt
    
    def detect_anomalies(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect statistical anomalies in event patterns.
        
        Args:
            events: List of events to analyze
            
        Returns:
            Dictionary with anomaly detection results
        """
        # Simple statistical anomaly detection
        # Count events by source type and hour
        from collections import defaultdict
        
        hourly_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                source_type = event["source_type"]
                hourly_counts[source_type][hour_key] += 1
            except Exception:
                continue
        
        # Detect spikes (simple: > 2x average)
        anomalies = []
        for source_type, counts in hourly_counts.items():
            if not counts:
                continue
            
            values = list(counts.values())
            avg = sum(values) / len(values)
            
            for hour, count in counts.items():
                if count > avg * 2:
                    anomalies.append({
                        "source_type": source_type,
                        "hour": hour,
                        "count": count,
                        "average": avg,
                        "spike_factor": count / avg if avg > 0 else 0
                    })
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies
        }
    
    def detect_escalation_pattern(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Detect escalation patterns in events.
        
        Args:
            events: List of events to analyze
            
        Returns:
            Dictionary with escalation pattern if detected
        """
        # Group events by day
        from collections import defaultdict
        
        daily_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        for event in events:
            try:
                timestamp = datetime.fromisoformat(event["timestamp"])
                day_key = timestamp.strftime("%Y-%m-%d")
                source_type = event["source_type"]
                daily_counts[day_key][source_type] += 1
            except Exception:
                continue
        
        # Check for increasing trend (simplified)
        satellite_trend = []
        news_trend = []
        
        for day in sorted(daily_counts.keys()):
            satellite_trend.append(daily_counts[day].get("satellite", 0))
            news_trend.append(daily_counts[day].get("news", 0))
        
        # Simple trend detection: increasing over last 3 days
        escalation_detected = False
        if len(satellite_trend) >= 3:
            if satellite_trend[-1] > satellite_trend[-2] > satellite_trend[-3]:
                escalation_detected = True
        
        return {
            "escalation_detected": escalation_detected,
            "satellite_trend": satellite_trend,
            "news_trend": news_trend,
            "days_analyzed": len(daily_counts)
        }
