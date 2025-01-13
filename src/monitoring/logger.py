"""
Logging configuration and utilities.
"""
import logging
import logging.handlers
import json
from typing import Any, Dict, Optional
from pathlib import Path
import time
import threading
from datetime import datetime
import os
from metrics import counter

# Configuration
LOG_DIR = Path("logs")
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5

class StructuredLogger:
    """Logger with structured logging and metrics."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
        self.context = {}
        self._local = threading.local()
    
    def _setup_logger(self):
        """Setup logger handlers and formatters."""
        # Créer le répertoire de logs
        LOG_DIR.mkdir(exist_ok=True)
        
        # Handler fichier
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.log",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Handler JSON
        json_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "app.json",
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT
        )
        json_handler.setFormatter(JsonFormatter())
        
        # Handler console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        
        # Configurer le logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(logging.INFO)
    
    def set_context(self, **kwargs):
        """Set context for all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context."""
        self.context.clear()
    
    def _get_request_context(self) -> Dict[str, Any]:
        """Get request-specific context."""
        try:
            return self._local.context
        except AttributeError:
            self._local.context = {}
            return self._local.context
    
    def set_request_context(self, **kwargs):
        """Set context for current request."""
        self._get_request_context().update(kwargs)
    
    def clear_request_context(self):
        """Clear request-specific context."""
        self._local.context = {}
    
    def _format_message(self,
                       message: str,
                       extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Format log message with context."""
        data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
        }
        
        # Ajouter les contextes
        data.update(self.context)
        data.update(self._get_request_context())
        
        if extra:
            data.update(extra)
            
        return data
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message."""
        self.logger.debug(
            message,
            extra={"structured": self._format_message(message, extra)}
        )
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message."""
        self.logger.info(
            message,
            extra={"structured": self._format_message(message, extra)}
        )
        counter("log.info", 1)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message."""
        self.logger.warning(
            message,
            extra={"structured": self._format_message(message, extra)}
        )
        counter("log.warning", 1)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message."""
        self.logger.error(
            message,
            extra={"structured": self._format_message(message, extra)}
        )
        counter("log.error", 1)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log exception with traceback."""
        self.logger.exception(
            message,
            extra={"structured": self._format_message(message, extra)}
        )
        counter("log.exception", 1)

class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the record as JSON."""
        data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno
        }
        
        # Ajouter les données structurées
        if hasattr(record, "structured"):
            data.update(record.structured)
        else:
            data["message"] = record.getMessage()
            
        # Ajouter l'exception si présente
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(data)

# Logger global
logger = StructuredLogger("watermark_evil")

def init_logging():
    """Initialize logging system."""
    try:
        # Créer les répertoires
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Configurer le logging de base
        logging.basicConfig(
            level=logging.INFO,
            format=LOG_FORMAT
        )
        
        # Désactiver les logs externes bruyants
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.WARNING)
        
        logger.info("Logging system initialized", extra={
            "log_dir": str(LOG_DIR),
            "max_bytes": MAX_BYTES,
            "backup_count": BACKUP_COUNT
        })
        
    except Exception as e:
        print(f"Failed to initialize logging: {e}")
        raise

# Contexte de requête
class RequestContext:
    """Context manager for request logging."""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    
    def __enter__(self):
        logger.set_request_context(**self.kwargs)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.clear_request_context()
        
        if exc_type is not None:
            logger.exception(
                "Request failed",
                extra={
                    "error_type": exc_type.__name__,
                    "error": str(exc_val)
                }
            )
