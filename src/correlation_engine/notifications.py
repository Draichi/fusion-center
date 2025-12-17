"""
Email notifications for alerts.
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from src.shared.config import settings
from src.shared.logger import get_logger

logger = get_logger()


class EmailNotifier:
    """Sends email notifications for alerts."""
    
    def __init__(self):
        """Initialize email notifier with SMTP settings."""
        self.smtp_host = getattr(settings, "smtp_host", "smtp.gmail.com")
        self.smtp_port = getattr(settings, "smtp_port", 587)
        self.smtp_user = getattr(settings, "smtp_user", None)
        self.smtp_password = getattr(settings, "smtp_password", None)
        self.from_email = getattr(settings, "smtp_from", self.smtp_user)
        self.to_email = getattr(settings, "alert_email_to", None)
        
        if not all([self.smtp_user, self.smtp_password, self.to_email]):
            logger.warning("Email notifications not fully configured - emails will not be sent")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"Email notifications enabled (sending to {self.to_email})")
    
    def send_alert(self, correlation: dict[str, Any]) -> bool:
        """Send alert email for a correlation.
        
        Args:
            correlation: Correlation data dictionary
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.debug("Email notifications disabled, skipping")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[FUSION CENTER] {correlation['severity'].upper()} Alert"
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            
            # HTML body
            html = self._create_html_body(correlation)
            msg.attach(MIMEText(html, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent for correlation {correlation['correlation_id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send alert email: {e}")
            return False
    
    def _create_html_body(self, correlation: dict[str, Any]) -> str:
        """Create HTML email body."""
        severity_color = {
            "low": "#4CAF50",
            "medium": "#FF9800",
            "high": "#FF5722",
            "critical": "#F44336"
        }.get(correlation['severity'], "#9E9E9E")
        
        implications_html = "".join(
            f"<li>{imp}</li>" 
            for imp in correlation.get('implications', [])
        )
        
        actions_html = "".join(
            f"<li>{action}</li>" 
            for action in correlation.get('recommended_actions', [])
        )
        
        event_ids_html = "".join(
            f"<li><code>{eid}</code></li>" 
            for eid in correlation.get('event_ids', [])
        )
        
        html = f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; }}
              .header {{ background-color: {severity_color}; color: white; padding: 20px; }}
              .content {{ padding: 20px; }}
              .section {{ margin: 20px 0; }}
              .label {{ font-weight: bold; }}
              code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
            </style>
          </head>
          <body>
            <div class="header">
              <h1>{correlation['severity'].upper()} Alert</h1>
              <p style="margin: 0;">Event Correlation Detected</p>
            </div>
            
            <div class="content">
              <div class="section">
                <p class="label">Correlation Type:</p>
                <p>{correlation['correlation_type']}</p>
              </div>
              
              <div class="section">
                <p class="label">Description:</p>
                <p>{correlation['description']}</p>
              </div>
              
              <div class="section">
                <p class="label">Confidence:</p>
                <p>{correlation['confidence'].capitalize()}</p>
              </div>
              
              <div class="section">
                <p class="label">Time Detected:</p>
                <p>{correlation['timestamp']}</p>
              </div>
              
              {f'''
              <div class="section">
                <p class="label">Implications:</p>
                <ul>{implications_html}</ul>
              </div>
              ''' if correlation.get('implications') else ''}
              
              {f'''
              <div class="section">
                <p class="label">Recommended Actions:</p>
                <ul>{actions_html}</ul>
              </div>
              ''' if correlation.get('recommended_actions') else ''}
              
              <div class="section">
                <p class="label">Events Involved:</p>
                <ul>{event_ids_html}</ul>
              </div>
              
              <div class="section">
                <p><a href="http://localhost:8000" style="background: {severity_color}; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">View in Dashboard</a></p>
              </div>
            </div>
          </body>
        </html>
        """
        
        return html
