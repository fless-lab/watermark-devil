"""
Concurrency management for the API.
"""
import asyncio
from typing import Dict, Optional, Any, Callable
import logging
from dataclasses import dataclass
import time
from metrics import gauge, counter

logger = logging.getLogger(__name__)

@dataclass
class TaskInfo:
    """Information about a running task"""
    task: asyncio.Task
    start_time: float
    metadata: Dict[str, Any]

class ConcurrencyManager:
    """Manage concurrent tasks and resources"""
    
    def __init__(self, 
                 max_concurrent_tasks: int = 10,
                 task_timeout: int = 300):
        self.max_tasks = max_concurrent_tasks
        self.timeout = task_timeout
        self.tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Métriques initiales
        gauge("api.concurrency.max_tasks", max_concurrent_tasks)
        gauge("api.concurrency.active_tasks", 0)
    
    async def start_task(self,
                        task_id: str,
                        coroutine: Callable,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start a new task with management"""
        try:
            async with self._lock:
                # Vérifier si la tâche existe déjà
                if task_id in self.tasks:
                    counter("api.concurrency.errors", 
                           tags={"type": "duplicate_task"})
                    raise ValueError(f"Task {task_id} already exists")
                
                # Créer et démarrer la tâche
                task = asyncio.create_task(
                    self._run_managed_task(task_id, coroutine))
                
                self.tasks[task_id] = TaskInfo(
                    task=task,
                    start_time=time.monotonic(),
                    metadata=metadata or {}
                )
                
                # Métriques
                gauge("api.concurrency.active_tasks", len(self.tasks))
                
        except Exception as e:
            logger.error(f"Failed to start task {task_id}: {e}")
            counter("api.concurrency.errors", 
                   tags={"type": "task_start_failed"})
            raise
    
    async def _run_managed_task(self,
                              task_id: str,
                              coroutine: Callable) -> Any:
        """Run a task with resource management"""
        try:
            async with self._semaphore:
                # Démarrer le timer
                start_time = time.monotonic()
                
                # Exécuter avec timeout
                result = await asyncio.wait_for(
                    coroutine(),
                    timeout=self.timeout
                )
                
                # Métriques
                duration = time.monotonic() - start_time
                gauge("api.task.duration_seconds", duration, 
                     tags={"task_id": task_id})
                
                return result
                
        except asyncio.TimeoutError:
            logger.error(f"Task {task_id} timed out")
            counter("api.concurrency.errors", 
                   tags={"type": "task_timeout"})
            raise
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            counter("api.concurrency.errors", 
                   tags={"type": "task_failed"})
            raise
            
        finally:
            # Nettoyer
            async with self._lock:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                    gauge("api.concurrency.active_tasks", len(self.tasks))
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a running task"""
        try:
            async with self._lock:
                if task_id not in self.tasks:
                    return {
                        "status": "not_found",
                        "error": "Task not found"
                    }
                
                task_info = self.tasks[task_id]
                task = task_info.task
                
                if task.done():
                    status = "completed"
                    try:
                        result = task.result()
                        error = None
                    except Exception as e:
                        status = "failed"
                        result = None
                        error = str(e)
                else:
                    status = "running"
                    result = None
                    error = None
                
                duration = time.monotonic() - task_info.start_time
                
                return {
                    "status": status,
                    "duration": duration,
                    "result": result,
                    "error": error,
                    "metadata": task_info.metadata
                }
                
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            counter("api.concurrency.errors", 
                   tags={"type": "status_check_failed"})
            return {
                "status": "error",
                "error": f"Failed to get task status: {e}"
            }
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        try:
            async with self._lock:
                if task_id not in self.tasks:
                    return False
                
                task_info = self.tasks[task_id]
                task = task_info.task
                
                if not task.done():
                    task.cancel()
                    counter("api.concurrency.task_cancellations", 1)
                    
                return True
                
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            counter("api.concurrency.errors", 
                   tags={"type": "task_cancel_failed"})
            return False
    
    async def cleanup(self):
        """Clean up completed and timed out tasks"""
        try:
            async with self._lock:
                now = time.monotonic()
                for task_id in list(self.tasks.keys()):
                    task_info = self.tasks[task_id]
                    task = task_info.task
                    
                    # Nettoyer les tâches terminées
                    if task.done():
                        del self.tasks[task_id]
                        continue
                    
                    # Vérifier le timeout
                    duration = now - task_info.start_time
                    if duration > self.timeout:
                        task.cancel()
                        del self.tasks[task_id]
                        counter("api.concurrency.task_timeouts", 1)
                
                # Métriques
                gauge("api.concurrency.active_tasks", len(self.tasks))
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            counter("api.concurrency.errors", 
                   tags={"type": "cleanup_failed"})

# Instance globale
concurrency_manager = ConcurrencyManager()
