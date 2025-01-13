"""
Performance profiling utilities.
"""
import time
import cProfile
import pstats
import io
import functools
from typing import Optional, Callable, Any
from pathlib import Path
import logging
from contextlib import contextmanager
import threading
from metrics import gauge, histogram

logger = logging.getLogger(__name__)

class Profiler:
    """Performance profiler with metrics."""
    
    def __init__(self, name: str):
        self.name = name
        self.profiler = cProfile.Profile()
        self._local = threading.local()
        self._start_time = None
    
    def start(self):
        """Start profiling."""
        self._start_time = time.perf_counter()
        self.profiler.enable()
    
    def stop(self) -> pstats.Stats:
        """Stop profiling and return stats."""
        self.profiler.disable()
        duration = time.perf_counter() - self._start_time
        
        # Métriques
        histogram(f"profiler.{self.name}.duration_seconds", duration)
        
        # Créer les stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats("cumulative")
        ps.print_stats()
        
        # Logger
        logger.debug(f"Profile for {self.name}:\n{s.getvalue()}")
        
        return ps
    
    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        try:
            self.start()
            yield
        finally:
            self.stop()

def profile(name: Optional[str] = None) -> Callable:
    """Decorator for profiling functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler_name = name or f"{func.__module__}.{func.__name__}"
            profiler = Profiler(profiler_name)
            
            with profiler.profile():
                result = func(*args, **kwargs)
                
            return result
        return wrapper
    return decorator

class FunctionTimer:
    """Utility for timing function execution."""
    
    def __init__(self, name: str):
        self.name = name
        self._local = threading.local()
    
    @property
    def start_time(self) -> Optional[float]:
        """Get start time for current thread."""
        return getattr(self._local, "start_time", None)
    
    @start_time.setter
    def start_time(self, value: float):
        """Set start time for current thread."""
        self._local.start_time = value
    
    def start(self):
        """Start timing."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return duration."""
        if self.start_time is None:
            raise RuntimeError("Timer not started")
            
        duration = time.perf_counter() - self.start_time
        self.start_time = None
        
        # Métriques
        histogram(f"timer.{self.name}.duration_seconds", duration)
        
        return duration
    
    @contextmanager
    def time(self):
        """Context manager for timing."""
        try:
            self.start()
            yield
        finally:
            self.stop()

def time_function(name: Optional[str] = None) -> Callable:
    """Decorator for timing functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            timer_name = name or f"{func.__module__}.{func.__name__}"
            timer = FunctionTimer(timer_name)
            
            with timer.time():
                result = func(*args, **kwargs)
                
            return result
        return wrapper
    return decorator

class MemoryProfiler:
    """Memory usage profiler."""
    
    def __init__(self, name: str):
        self.name = name
        self._local = threading.local()
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def start(self):
        """Start memory profiling."""
        self._local.start_memory = self._get_memory_usage()
    
    def stop(self) -> float:
        """Stop profiling and return memory delta."""
        if not hasattr(self._local, "start_memory"):
            raise RuntimeError("Profiler not started")
            
        end_memory = self._get_memory_usage()
        delta = end_memory - self._local.start_memory
        
        # Métriques
        gauge(f"memory.{self.name}.usage_mb", end_memory)
        gauge(f"memory.{self.name}.delta_mb", delta)
        
        return delta
    
    @contextmanager
    def profile(self):
        """Context manager for memory profiling."""
        try:
            self.start()
            yield
        finally:
            self.stop()

def profile_memory(name: Optional[str] = None) -> Callable:
    """Decorator for memory profiling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler_name = name or f"{func.__module__}.{func.__name__}"
            profiler = MemoryProfiler(profiler_name)
            
            with profiler.profile():
                result = func(*args, **kwargs)
                
            return result
        return wrapper
    return decorator
