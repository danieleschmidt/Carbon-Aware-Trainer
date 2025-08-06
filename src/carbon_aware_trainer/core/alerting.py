"""Comprehensive alerting and notification system."""

import asyncio
import logging
import json
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional dependency handling
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

from .health import HealthCheck, HealthStatus
from .exceptions import MonitoringError


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of alerts."""
    SYSTEM_HEALTH = "system_health"
    CARBON_THRESHOLD = "carbon_threshold"
    API_ERROR = "api_error"
    TRAINING_INTERRUPTED = "training_interrupted"
    CONFIGURATION_CHANGED = "configuration_changed"
    SECURITY_EVENT = "security_event"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class Alert:
    """Represents an alert/notification."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


@dataclass
class EmailConfig:
    """Email notification configuration."""
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    from_address: str = ""
    recipients: List[str] = field(default_factory=list)
    use_tls: bool = True


@dataclass
class WebhookConfig:
    """Webhook notification configuration."""
    enabled: bool = False
    url: str = ""
    secret: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30


@dataclass
class SlackConfig:
    """Slack notification configuration."""
    enabled: bool = False
    webhook_url: str = ""
    channel: str = ""
    username: str = "Carbon Aware Trainer"
    emoji: str = ":warning:"


class AlertingManager:
    """Comprehensive alerting and notification management."""
    
    def __init__(
        self,
        email_config: Optional[EmailConfig] = None,
        webhook_config: Optional[WebhookConfig] = None,
        slack_config: Optional[SlackConfig] = None
    ):
        """Initialize alerting manager.
        
        Args:
            email_config: Email notification configuration
            webhook_config: Webhook notification configuration
            slack_config: Slack notification configuration
        """
        self.email_config = email_config or EmailConfig()
        self.webhook_config = webhook_config or WebhookConfig()
        self.slack_config = slack_config or SlackConfig()
        
        # Alert storage and management
        self.alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000
        
        # Alert rules and thresholds
        self.alert_rules: Dict[str, Callable[[Any], bool]] = {}
        self.cooldown_periods: Dict[str, timedelta] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Notification queues and batching
        self.notification_queue: asyncio.Queue = asyncio.Queue()
        self.batch_notifications = True
        self.batch_interval = timedelta(minutes=5)
        self.batched_alerts: List[Alert] = []
        self.last_batch_time = datetime.now()
        
        # Background tasks
        self._notification_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Setup default alert rules
        self._setup_default_alert_rules()
        
        logger.info("AlertingManager initialized")
    
    async def start(self) -> None:
        """Start alerting manager background tasks."""
        if self._running:
            logger.warning("AlertingManager already running")
            return
        
        self._running = True
        self._notification_task = asyncio.create_task(self._notification_worker())
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
        
        logger.info("AlertingManager started")
    
    async def stop(self) -> None:
        """Stop alerting manager background tasks."""
        self._running = False
        
        if self._notification_task:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info("AlertingManager stopped")
    
    async def create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """Create and process a new alert.
        
        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Source system/component
            metadata: Additional metadata
            
        Returns:
            Created alert object
        """
        # Generate unique alert ID
        alert_id = f"{alert_type.value}_{source}_{datetime.now().timestamp()}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        # Check if alert should be suppressed due to cooldown
        if self._should_suppress_alert(alert):
            logger.debug(f"Alert suppressed due to cooldown: {alert_id}")
            return alert
        
        # Store alert
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Trim history if needed
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Queue for notification
        await self.notification_queue.put(alert)
        
        logger.info(f"Alert created: {alert_id} ({severity.value})")
        return alert
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert.
        
        Args:
            alert_id: ID of alert to resolve
            
        Returns:
            True if resolved successfully
        """
        if alert_id not in self.alerts:
            logger.warning(f"Alert not found for resolution: {alert_id}")
            return False
        
        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        del self.alerts[alert_id]
        
        logger.info(f"Alert resolved: {alert_id}")
        return True
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who acknowledged the alert
            
        Returns:
            True if acknowledged successfully
        """
        if alert_id not in self.alerts:
            logger.warning(f"Alert not found for acknowledgment: {alert_id}")
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    def add_alert_rule(
        self,
        rule_name: str,
        rule_function: Callable[[Any], bool],
        cooldown_period: Optional[timedelta] = None
    ) -> None:
        """Add custom alert rule.
        
        Args:
            rule_name: Name of the alert rule
            rule_function: Function that determines if alert should fire
            cooldown_period: Minimum time between alerts of this type
        """
        self.alert_rules[rule_name] = rule_function
        
        if cooldown_period:
            self.cooldown_periods[rule_name] = cooldown_period
        
        logger.info(f"Alert rule added: {rule_name}")
    
    def remove_alert_rule(self, rule_name: str) -> None:
        """Remove alert rule.
        
        Args:
            rule_name: Name of rule to remove
        """
        self.alert_rules.pop(rule_name, None)
        self.cooldown_periods.pop(rule_name, None)
        self.last_alert_times.pop(rule_name, None)
        
        logger.info(f"Alert rule removed: {rule_name}")
    
    async def check_health_alerts(self, health_checks: Dict[str, HealthCheck]) -> None:
        """Check health status and create alerts if needed.
        
        Args:
            health_checks: Dictionary of health check results
        """
        for check_name, health_check in health_checks.items():
            await self._evaluate_health_check_alert(check_name, health_check)
    
    async def check_carbon_threshold_alert(
        self,
        current_intensity: float,
        threshold: float,
        region: str
    ) -> None:
        """Check carbon intensity threshold and create alert if needed.
        
        Args:
            current_intensity: Current carbon intensity
            threshold: Threshold value
            region: Region code
        """
        if current_intensity > threshold:
            await self.create_alert(
                alert_type=AlertType.CARBON_THRESHOLD,
                severity=AlertSeverity.WARNING,
                title=f"Carbon threshold exceeded in {region}",
                message=f"Carbon intensity {current_intensity:.1f} exceeds threshold {threshold:.1f} gCO2/kWh",
                source=f"carbon_monitor_{region}",
                metadata={
                    'current_intensity': current_intensity,
                    'threshold': threshold,
                    'region': region
                }
            )
    
    async def check_api_error_alert(self, provider: str, error: Exception) -> None:
        """Create alert for API errors.
        
        Args:
            provider: API provider name
            error: Error that occurred
        """
        await self.create_alert(
            alert_type=AlertType.API_ERROR,
            severity=AlertSeverity.WARNING,
            title=f"API error for {provider}",
            message=f"API error occurred: {str(error)}",
            source=f"api_{provider}",
            metadata={
                'provider': provider,
                'error_type': type(error).__name__,
                'error_message': str(error)
            }
        )
    
    def get_active_alerts(
        self,
        severity_filter: Optional[AlertSeverity] = None,
        type_filter: Optional[AlertType] = None
    ) -> List[Alert]:
        """Get list of active alerts with optional filtering.
        
        Args:
            severity_filter: Filter by severity level
            type_filter: Filter by alert type
            
        Returns:
            List of matching active alerts
        """
        alerts = list(self.alerts.values())
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        if type_filter:
            alerts = [a for a in alerts if a.alert_type == type_filter]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting system statistics.
        
        Returns:
            Dictionary with statistics
        """
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]
        
        severity_counts = {}
        type_counts = {}
        
        for alert in recent_alerts:
            severity_counts[alert.severity.value] = severity_counts.get(alert.severity.value, 0) + 1
            type_counts[alert.alert_type.value] = type_counts.get(alert.alert_type.value, 0) + 1
        
        return {
            'active_alerts': len(self.alerts),
            'total_alerts_24h': len(recent_alerts),
            'severity_breakdown_24h': severity_counts,
            'type_breakdown_24h': type_counts,
            'alert_rules_count': len(self.alert_rules),
            'notification_channels': {
                'email': self.email_config.enabled,
                'webhook': self.webhook_config.enabled,
                'slack': self.slack_config.enabled
            }
        }
    
    async def _notification_worker(self) -> None:
        """Background worker for processing notifications."""
        while self._running:
            try:
                # Process queued notifications
                while not self.notification_queue.empty():
                    alert = await self.notification_queue.get()
                    
                    if self.batch_notifications:
                        self.batched_alerts.append(alert)
                    else:
                        await self._send_notifications([alert])
                
                # Send batched notifications if interval has passed
                if (self.batch_notifications and 
                    self.batched_alerts and 
                    datetime.now() - self.last_batch_time >= self.batch_interval):
                    
                    await self._send_notifications(self.batched_alerts)
                    self.batched_alerts.clear()
                    self.last_batch_time = datetime.now()
                
                await asyncio.sleep(1)  # Prevent busy loop
                
            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                await asyncio.sleep(5)  # Backoff on error
    
    async def _cleanup_worker(self) -> None:
        """Background worker for cleaning up old alerts."""
        while self._running:
            try:
                # Clean up resolved alerts older than 7 days
                cutoff_time = datetime.now() - timedelta(days=7)
                
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.alerts.items()
                    if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self.alerts[alert_id]
                
                if alerts_to_remove:
                    logger.info(f"Cleaned up {len(alerts_to_remove)} old resolved alerts")
                
                # Clean up old history entries
                if len(self.alert_history) > self.max_history_size:
                    self.alert_history = self.alert_history[-self.max_history_size:]
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                await asyncio.sleep(3600)
    
    async def _send_notifications(self, alerts: List[Alert]) -> None:
        """Send notifications for alerts.
        
        Args:
            alerts: List of alerts to send notifications for
        """
        if not alerts:
            return
        
        # Send to each enabled notification channel
        notification_tasks = []
        
        if self.email_config.enabled:
            notification_tasks.append(self._send_email_notifications(alerts))
        
        if self.webhook_config.enabled:
            notification_tasks.append(self._send_webhook_notifications(alerts))
        
        if self.slack_config.enabled:
            notification_tasks.append(self._send_slack_notifications(alerts))
        
        # Execute all notifications concurrently
        if notification_tasks:
            results = await asyncio.gather(*notification_tasks, return_exceptions=True)
            
            # Log any notification failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    channel = ['email', 'webhook', 'slack'][i]
                    logger.error(f"Failed to send {channel} notification: {result}")
    
    async def _send_email_notifications(self, alerts: List[Alert]) -> None:
        """Send email notifications.
        
        Args:
            alerts: List of alerts to send
        """
        try:
            if not self.email_config.recipients:
                return
            
            # Create email content
            subject = self._create_email_subject(alerts)
            body = self._create_email_body(alerts)
            
            # Send email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config.from_address
            msg['To'] = ', '.join(self.email_config.recipients)
            
            # Add plain text and HTML versions
            text_part = MIMEText(body, 'plain')
            html_part = MIMEText(self._create_email_html_body(alerts), 'html')
            
            msg.attach(text_part)
            msg.attach(html_part)
            
            # Send via SMTP
            with smtplib.SMTP(self.email_config.smtp_host, self.email_config.smtp_port) as server:
                if self.email_config.use_tls:
                    server.starttls()
                
                if self.email_config.username and self.email_config.password:
                    server.login(self.email_config.username, self.email_config.password)
                
                server.send_message(msg)
            
            logger.info(f"Email notification sent for {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notifications(self, alerts: List[Alert]) -> None:
        """Send webhook notifications.
        
        Args:
            alerts: List of alerts to send
        """
        if not HAS_AIOHTTP:
            logger.error("Cannot send webhook notifications: aiohttp not available (install aiohttp)")
            return
            
        try:
            payload = {
                'timestamp': datetime.now().isoformat(),
                'alert_count': len(alerts),
                'alerts': [
                    {
                        'id': alert.alert_id,
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'title': alert.title,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'source': alert.source,
                        'metadata': alert.metadata
                    }
                    for alert in alerts
                ]
            }
            
            headers = {
                'Content-Type': 'application/json',
                **self.webhook_config.headers
            }
            
            # Add signature if secret is configured
            if self.webhook_config.secret:
                import hmac
                import hashlib
                
                payload_json = json.dumps(payload)
                signature = hmac.new(
                    self.webhook_config.secret.encode(),
                    payload_json.encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Signature'] = f'sha256={signature}'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_config.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.webhook_config.timeout)
                ) as response:
                    response.raise_for_status()
            
            logger.info(f"Webhook notification sent for {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def _send_slack_notifications(self, alerts: List[Alert]) -> None:
        """Send Slack notifications.
        
        Args:
            alerts: List of alerts to send
        """
        if not HAS_AIOHTTP:
            logger.error("Cannot send Slack notifications: aiohttp not available (install aiohttp)")
            return
            
        try:
            # Create Slack message
            message = self._create_slack_message(alerts)
            
            payload = {
                'channel': self.slack_config.channel,
                'username': self.slack_config.username,
                'icon_emoji': self.slack_config.emoji,
                'text': message['text'],
                'attachments': message.get('attachments', [])
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.slack_config.webhook_url,
                    json=payload
                ) as response:
                    response.raise_for_status()
            
            logger.info(f"Slack notification sent for {len(alerts)} alerts")
            
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def _should_suppress_alert(self, alert: Alert) -> bool:
        """Check if alert should be suppressed due to cooldown.
        
        Args:
            alert: Alert to check
            
        Returns:
            True if alert should be suppressed
        """
        alert_key = f"{alert.alert_type.value}_{alert.source}"
        
        if alert_key not in self.cooldown_periods:
            return False
        
        last_alert_time = self.last_alert_times.get(alert_key)
        if not last_alert_time:
            self.last_alert_times[alert_key] = alert.timestamp
            return False
        
        cooldown_period = self.cooldown_periods[alert_key]
        if alert.timestamp - last_alert_time < cooldown_period:
            return True
        
        self.last_alert_times[alert_key] = alert.timestamp
        return False
    
    async def _evaluate_health_check_alert(self, check_name: str, health_check: HealthCheck) -> None:
        """Evaluate health check and create alert if needed.
        
        Args:
            check_name: Name of health check
            health_check: Health check result
        """
        if health_check.status == HealthStatus.CRITICAL:
            await self.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.CRITICAL,
                title=f"Critical health check failure: {check_name}",
                message=health_check.message,
                source=f"health_check_{check_name}",
                metadata={
                    'check_name': check_name,
                    'status': health_check.status.value,
                    'duration_ms': health_check.duration_ms,
                    'details': health_check.details
                }
            )
        elif health_check.status == HealthStatus.WARNING:
            await self.create_alert(
                alert_type=AlertType.SYSTEM_HEALTH,
                severity=AlertSeverity.WARNING,
                title=f"Health check warning: {check_name}",
                message=health_check.message,
                source=f"health_check_{check_name}",
                metadata={
                    'check_name': check_name,
                    'status': health_check.status.value,
                    'duration_ms': health_check.duration_ms,
                    'details': health_check.details
                }
            )
    
    def _setup_default_alert_rules(self) -> None:
        """Setup default alert rules and cooldown periods."""
        # Default cooldown periods to prevent alert spam
        self.cooldown_periods.update({
            f"{AlertType.SYSTEM_HEALTH.value}_health_check": timedelta(minutes=15),
            f"{AlertType.CARBON_THRESHOLD.value}_carbon_monitor": timedelta(minutes=30),
            f"{AlertType.API_ERROR.value}_api": timedelta(minutes=10),
            f"{AlertType.PERFORMANCE_DEGRADATION.value}_performance": timedelta(minutes=20)
        })
    
    def _create_email_subject(self, alerts: List[Alert]) -> str:
        """Create email subject line.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Email subject
        """
        if len(alerts) == 1:
            alert = alerts[0]
            return f"[{alert.severity.value.upper()}] {alert.title}"
        else:
            critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
            
            return f"Carbon Aware Trainer: {len(alerts)} alerts ({critical_count} critical, {warning_count} warnings)"
    
    def _create_email_body(self, alerts: List[Alert]) -> str:
        """Create plain text email body.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Email body text
        """
        body_lines = [
            "Carbon Aware Trainer Alert Notification",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"Alert Count: {len(alerts)}",
            ""
        ]
        
        for alert in alerts:
            body_lines.extend([
                f"Alert ID: {alert.alert_id}",
                f"Type: {alert.alert_type.value}",
                f"Severity: {alert.severity.value.upper()}",
                f"Title: {alert.title}",
                f"Message: {alert.message}",
                f"Source: {alert.source}",
                f"Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                "",
                "-" * 40,
                ""
            ])
        
        return "\n".join(body_lines)
    
    def _create_email_html_body(self, alerts: List[Alert]) -> str:
        """Create HTML email body.
        
        Args:
            alerts: List of alerts
            
        Returns:
            HTML email body
        """
        # Simple HTML template - in production, use a proper template engine
        html_parts = [
            "<html><body>",
            "<h2>Carbon Aware Trainer Alert Notification</h2>",
            f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
            f"<p><strong>Alert Count:</strong> {len(alerts)}</p>",
            "<hr>"
        ]
        
        for alert in alerts:
            severity_color = {
                AlertSeverity.INFO: "blue",
                AlertSeverity.WARNING: "orange",
                AlertSeverity.CRITICAL: "red",
                AlertSeverity.EMERGENCY: "darkred"
            }.get(alert.severity, "black")
            
            html_parts.extend([
                "<div style='margin-bottom: 20px; padding: 10px; border-left: 4px solid {};'>".format(severity_color),
                f"<h3 style='color: {severity_color};'>{alert.title}</h3>",
                f"<p><strong>Severity:</strong> <span style='color: {severity_color};'>{alert.severity.value.upper()}</span></p>",
                f"<p><strong>Message:</strong> {alert.message}</p>",
                f"<p><strong>Source:</strong> {alert.source}</p>",
                f"<p><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>",
                "</div>"
            ])
        
        html_parts.append("</body></html>")
        return "".join(html_parts)
    
    def _create_slack_message(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Create Slack message payload.
        
        Args:
            alerts: List of alerts
            
        Returns:
            Slack message dictionary
        """
        if len(alerts) == 1:
            alert = alerts[0]
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "good")
            
            return {
                "text": f":warning: Carbon Aware Trainer Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color,
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Message", "value": alert.message, "short": False},
                            {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": True}
                        ]
                    }
                ]
            }
        else:
            critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)
            
            text = f":warning: Carbon Aware Trainer: {len(alerts)} alerts"
            if critical_count > 0:
                text += f" ({critical_count} critical)"
            
            return {"text": text}


# Global alerting manager instance
alerting_manager = AlertingManager()