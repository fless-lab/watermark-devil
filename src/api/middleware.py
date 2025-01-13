"""
Security and monitoring middleware for the WatermarkEvil API.
"""
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import logging
import prometheus_client
from prometheus_client import Counter, Histogram
from ratelimit import RateLimitMiddleware, Rule
from ratelimit.backends.redis import RedisBackend

from .config import settings

# Setup logging
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for request validation and protection."""
    
    def __init__(
        self,
        app: FastAPI,
        api_key_header: APIKeyHeader = APIKeyHeader(name="X-API-Key")
    ):
        super().__init__(app)
        self.api_key_header = api_key_header

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        # Validate API key
        if not await self._validate_api_key(request):
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid API key"}
            )

        # Validate file size and type
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length:
                if int(content_length) > settings.security.max_file_size:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "File too large"}
                    )

            content_type = request.headers.get("content-type", "")
            if "multipart/form-data" in content_type:
                form = await request.form()
                for field in form:
                    if hasattr(form[field], "content_type"):
                        if form[field].content_type not in settings.security.allowed_file_types:
                            return JSONResponse(
                                status_code=415,
                                content={"detail": "Unsupported file type"}
                            )

        return await call_next(request)

    async def _validate_api_key(self, request: Request) -> bool:
        try:
            api_key = await self.api_key_header(request)
            return api_key == settings.api.api_key
        except:
            return False

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring and metrics collection."""
    
    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"Status: {response.status_code} "
            f"Duration: {duration:.3f}s"
        )

        return response

def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application."""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security
    app.add_middleware(SecurityMiddleware)

    # Rate limiting
    if settings.security.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            authenticate=lambda r: r.client.host,
            backend=RedisBackend(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                ssl=False
            ),
            config={
                r".*": Rule(
                    second=settings.security.rate_limit_requests,
                    block_time=settings.security.rate_limit_period
                )
            }
        )

    # Monitoring
    app.add_middleware(MonitoringMiddleware)

    # Prometheus metrics endpoint
    if settings.monitoring.prometheus_enabled:
        @app.get("/metrics")
        async def metrics():
            return Response(
                prometheus_client.generate_latest(),
                media_type="text/plain"
            )
