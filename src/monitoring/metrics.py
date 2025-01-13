from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
REQUEST_COUNT = Counter(
    'watermark_evil_requests_total',
    'Total number of requests processed',
    ['endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'watermark_evil_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint']
)

# Processing metrics
PROCESSING_TIME = Histogram(
    'watermark_evil_processing_time_seconds',
    'Time spent processing images',
    ['operation', 'method']
)

DETECTION_COUNT = Counter(
    'watermark_evil_detections_total',
    'Number of watermarks detected',
    ['type']
)

REMOVAL_SUCCESS_RATE = Gauge(
    'watermark_evil_removal_success_rate',
    'Success rate of watermark removal',
    ['method']
)

# System metrics
GPU_MEMORY_USAGE = Gauge(
    'watermark_evil_gpu_memory_usage_bytes',
    'GPU memory usage in bytes'
)

ACTIVE_WORKERS = Gauge(
    'watermark_evil_active_workers',
    'Number of active workers'
)

class MetricsMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        path = scope["path"]
        start_time = time.time()
        
        async def wrapped_send(message):
            if message["type"] == "http.response.start":
                status_code = message["status"]
                REQUEST_COUNT.labels(
                    endpoint=path,
                    status=status_code
                ).inc()
                
                REQUEST_LATENCY.labels(
                    endpoint=path
                ).observe(time.time() - start_time)
            
            await send(message)
        
        await self.app(scope, receive, wrapped_send)

def record_processing_time(operation: str, method: str, duration: float):
    """Record the time spent processing an image"""
    PROCESSING_TIME.labels(
        operation=operation,
        method=method
    ).observe(duration)

def record_detection(watermark_type: str):
    """Record a watermark detection"""
    DETECTION_COUNT.labels(
        type=watermark_type
    ).inc()

def update_removal_success_rate(method: str, success_rate: float):
    """Update the success rate for a removal method"""
    REMOVAL_SUCCESS_RATE.labels(
        method=method
    ).set(success_rate)

def update_gpu_memory_usage(bytes_used: int):
    """Update GPU memory usage"""
    GPU_MEMORY_USAGE.set(bytes_used)

def update_active_workers(count: int):
    """Update number of active workers"""
    ACTIVE_WORKERS.set(count)
