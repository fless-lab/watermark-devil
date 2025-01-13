"""
Alert management system.
"""
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging
from enum import Enum
import smtplib
from email.mime.text import MIMEText
import json
from metrics import counter, gauge

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Niveaux de sévérité des alertes."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Représente une alerte."""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    tags: Dict[str, str]
    value: Optional[float] = None
    threshold: Optional[float] = None

class AlertManager:
    """Gestionnaire d'alertes."""
    
    def __init__(self):
        self.handlers: Dict[str, List[Callable[[Alert], None]]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self._lock = threading.Lock()
    
    def add_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Ajoute un handler pour un niveau de sévérité."""
        with self._lock:
            if severity.value not in self.handlers:
                self.handlers[severity.value] = []
            self.handlers[severity.value].append(handler)
    
    def remove_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Supprime un handler."""
        with self._lock:
            if severity.value in self.handlers:
                self.handlers[severity.value].remove(handler)
    
    def trigger(self, alert: Alert):
        """Déclenche une alerte."""
        alert_key = f"{alert.name}:{alert.severity.value}"
        
        with self._lock:
            # Vérifier si l'alerte est déjà active
            if alert_key in self.active_alerts:
                return
                
            # Activer l'alerte
            self.active_alerts[alert_key] = alert
            
            # Métriques
            counter(f"alerts.{alert.severity.value}", 1)
            gauge("alerts.active", len(self.active_alerts))
            
            # Logger
            logger.warning(
                f"Alert triggered: {alert.name}",
                extra={
                    "alert": {
                        "name": alert.name,
                        "severity": alert.severity.value,
                        "message": alert.message,
                        "tags": alert.tags
                    }
                }
            )
            
            # Notifier les handlers
            for handler in self.handlers.get(alert.severity.value, []):
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(
                        f"Alert handler failed: {e}",
                        exc_info=True
                    )
    
    def resolve(self, name: str, severity: AlertSeverity):
        """Résout une alerte."""
        alert_key = f"{name}:{severity.value}"
        
        with self._lock:
            if alert_key in self.active_alerts:
                del self.active_alerts[alert_key]
                gauge("alerts.active", len(self.active_alerts))
                
                logger.info(f"Alert resolved: {name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Retourne les alertes actives."""
        with self._lock:
            return list(self.active_alerts.values())

class EmailAlertHandler:
    """Handler d'alertes par email."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
    
    def __call__(self, alert: Alert):
        """Envoie l'alerte par email."""
        subject = f"[{alert.severity.value.upper()}] {alert.name}"
        
        body = {
            "name": alert.name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "tags": alert.tags
        }
        
        if alert.value is not None:
            body["value"] = alert.value
        if alert.threshold is not None:
            body["threshold"] = alert.threshold
            
        msg = MIMEText(json.dumps(body, indent=2))
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self.username, self.password)
                smtp.send_message(msg)
                
            counter("alerts.email.sent", 1)
            
        except Exception as e:
            logger.error(
                f"Failed to send alert email: {e}",
                exc_info=True
            )
            counter("alerts.email.failed", 1)

class SlackAlertHandler:
    """Handler d'alertes Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def __call__(self, alert: Alert):
        """Envoie l'alerte sur Slack."""
        import requests
        
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffd700",
            AlertSeverity.ERROR: "#ff4500",
            AlertSeverity.CRITICAL: "#ff0000"
        }[alert.severity]
        
        attachment = {
            "color": color,
            "title": alert.name,
            "text": alert.message,
            "fields": [
                {
                    "title": "Severity",
                    "value": alert.severity.value,
                    "short": True
                }
            ]
        }
        
        for key, value in alert.tags.items():
            attachment["fields"].append({
                "title": key,
                "value": value,
                "short": True
            })
            
        if alert.value is not None:
            attachment["fields"].append({
                "title": "Value",
                "value": str(alert.value),
                "short": True
            })
            
        if alert.threshold is not None:
            attachment["fields"].append({
                "title": "Threshold",
                "value": str(alert.threshold),
                "short": True
            })
            
        payload = {
            "attachments": [attachment]
        }
        
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5
            )
            response.raise_for_status()
            
            counter("alerts.slack.sent", 1)
            
        except Exception as e:
            logger.error(
                f"Failed to send Slack alert: {e}",
                exc_info=True
            )
            counter("alerts.slack.failed", 1)

# Alert manager global
alert_manager = AlertManager()
