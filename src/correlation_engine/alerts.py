"""
Alert Generator - creates and manages alerts based on correlations.
"""

from datetime import datetime
from typing import Any

from src.correlation_engine.config import ALERT_MIN_CONFIDENCE, ALERT_EMAIL_SEVERITIES
from src.correlation_engine.notifications import EmailNotifier
from src.correlation_engine.schemas import Correlation
from src.correlation_engine.storage import EventStorage
from src.shared.logger import get_logger

logger = get_logger()


class AlertGenerator:
    """Generates alerts based on correlations."""
    
    def __init__(self, storage: EventStorage, email_notifier: EmailNotifier | None = None):
        """Initialize alert generator.
        
        Args:
            storage: Event storage instance
            email_notifier: Optional email notifier (will create if not provided)
        """
        self.storage = storage
        self.email_notifier = email_notifier or EmailNotifier()
        logger.info("Alert Generator initialized")
    
    def process_correlation(self, correlation: Correlation) -> bool:
        """Process a correlation and generate alert if needed.
        
        Args:
            correlation: Correlation to process
            
        Returns:
            True if alert was generated, False otherwise
        """
        # Check if alert should be generated
        if not self._should_alert(correlation):
            logger.debug(f"Correlation {correlation.correlation_id} does not meet alert criteria")
            return False
        
        # Store correlation
        self.storage.insert_correlation(correlation)
        
        # Check if email should be sent
        send_email = correlation.severity.value in ALERT_EMAIL_SEVERITIES
        email_sent = False
        
        if send_email:
            email_sent = self.email_notifier.send_alert(correlation.to_dict())
        
        # Record alert
        self.storage.record_alert(correlation.correlation_id, email_sent=email_sent)
        
        logger.info(
            f"Alert generated for correlation {correlation.correlation_id} "
            f"(severity: {correlation.severity.value}, email: {email_sent})"
        )
        
        return True
    
    def _should_alert(self, correlation: Correlation) -> bool:
        """Determine if an alert should be generated for a correlation.
        
        Args:
            correlation: Correlation to evaluate
            
        Returns:
            True if alert should be generated
        """
        # Check confidence threshold
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.85}
        confidence_score = confidence_map.get(correlation.confidence, 0.5)
        
        if confidence_score < ALERT_MIN_CONFIDENCE:
            return False
        
        # Check number of sources involved
        # TODO: Get events from storage to check sources
        # For now, assume it's good if confidence is met
        
        return True
    
    def get_recent_alerts(self, hours_back: float = 24) -> list[dict[str, Any]]:
        """Get recent alerts.
        
        Args:
            hours_back: Hours to look back
            
        Returns:
            List of alert dictionaries
        """
        # Get recent correlations (they are the alerts)
        correlations = self.storage.get_correlations(hours_back=hours_back)
        return correlations
