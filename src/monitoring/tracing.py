"""
Distributed tracing utilities.
"""
import time
import uuid
import threading
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging
from metrics import histogram

logger = logging.getLogger(__name__)

class Span:
    """Représente une opération tracée."""
    
    def __init__(
        self,
        name: str,
        parent_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ):
        self.name = name
        self.id = str(uuid.uuid4())
        self.parent_id = parent_id
        self.trace_id = trace_id or str(uuid.uuid4())
        self.start_time = time.perf_counter_ns()
        self.end_time: Optional[int] = None
        self.tags: Dict[str, str] = {}
        self.events: List[Dict[str, Any]] = []
    
    def finish(self):
        """Termine le span."""
        self.end_time = time.perf_counter_ns()
        duration_ms = (self.end_time - self.start_time) / 1_000_000
        
        # Métriques
        histogram(f"trace.{self.name}.duration_ms", duration_ms)
    
    def set_tag(self, key: str, value: str):
        """Ajoute un tag au span."""
        self.tags[key] = value
    
    def log_event(self, event: str, payload: Optional[Dict[str, Any]] = None):
        """Ajoute un événement au span."""
        self.events.append({
            "timestamp": time.perf_counter_ns(),
            "event": event,
            "payload": payload or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit le span en dictionnaire."""
        return {
            "name": self.name,
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "tags": self.tags,
            "events": self.events
        }

class Tracer:
    """Gestionnaire de tracing."""
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def active_span(self) -> Optional[Span]:
        """Retourne le span actif pour le thread courant."""
        return getattr(self._local, "span_stack", [])[-1] if hasattr(self._local, "span_stack") else None
    
    def start_active_span(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Span:
        """Démarre un nouveau span actif."""
        if not hasattr(self._local, "span_stack"):
            self._local.span_stack = []
            
        parent = self.active_span
        span = Span(
            name=name,
            parent_id=parent.id if parent else None,
            trace_id=parent.trace_id if parent else None
        )
        
        if tags:
            for key, value in tags.items():
                span.set_tag(key, value)
                
        self._local.span_stack.append(span)
        return span
    
    def finish_active_span(self):
        """Termine le span actif."""
        if not hasattr(self._local, "span_stack") or not self._local.span_stack:
            raise RuntimeError("No active span")
            
        span = self._local.span_stack.pop()
        span.finish()
        
        # Logger
        logger.debug(
            f"Finished span {span.name}",
            extra={"span": span.to_dict()}
        )
    
    @contextmanager
    def start_span(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ):
        """Context manager pour le tracing."""
        try:
            span = self.start_active_span(name, tags)
            yield span
        finally:
            self.finish_active_span()
    
    def inject(self, span: Span) -> Dict[str, str]:
        """Injecte le contexte de tracing dans les headers."""
        return {
            "x-trace-id": span.trace_id,
            "x-span-id": span.id
        }
    
    def extract(self, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Extrait le contexte de tracing des headers."""
        trace_id = headers.get("x-trace-id")
        parent_id = headers.get("x-span-id")
        
        if not trace_id or not parent_id:
            return None
            
        return {
            "trace_id": trace_id,
            "parent_id": parent_id
        }

# Tracer global
tracer = Tracer()

def trace(name: str, tags: Optional[Dict[str, str]] = None):
    """Décorateur pour le tracing."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_span(name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator
