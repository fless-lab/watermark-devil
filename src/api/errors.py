"""
Error handling and documentation for the API.
"""
from typing import Dict, Any, Optional, Type
from enum import Enum
import logging
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from metrics import counter

logger = logging.getLogger(__name__)

class ErrorCode(str, Enum):
    """Error codes for the API"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    CONCURRENCY_ERROR = "CONCURRENCY_ERROR"
    PROCESSING_ERROR = "PROCESSING_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"

class APIError(Exception):
    """Base exception for API errors"""
    def __init__(self,
                 code: ErrorCode,
                 message: str,
                 status_code: int = 400,
                 details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(message)

class ValidationError(APIError):
    """Validation error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=400,
            details=details
        )

class RateLimitError(APIError):
    """Rate limit error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.RATE_LIMIT_ERROR,
            message=message,
            status_code=429,
            details=details
        )

class ConcurrencyError(APIError):
    """Concurrency error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.CONCURRENCY_ERROR,
            message=message,
            status_code=409,
            details=details
        )

class ProcessingError(APIError):
    """Processing error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.PROCESSING_ERROR,
            message=message,
            status_code=500,
            details=details
        )

class NotFoundError(APIError):
    """Not found error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=message,
            status_code=404,
            details=details
        )

class UnauthorizedError(APIError):
    """Unauthorized error"""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            code=ErrorCode.UNAUTHORIZED,
            message=message,
            status_code=401,
            details=details
        )

# Mapping des exceptions aux codes HTTP
ERROR_HANDLERS: Dict[Type[Exception], int] = {
    ValidationError: 400,
    RateLimitError: 429,
    ConcurrencyError: 409,
    ProcessingError: 500,
    NotFoundError: 404,
    UnauthorizedError: 401
}

async def error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for the API"""
    try:
        # Gérer les erreurs API personnalisées
        if isinstance(exc, APIError):
            counter("api.errors", tags={
                "type": exc.code.value,
                "status": exc.status_code
            })
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "code": exc.code.value,
                    "message": exc.message,
                    "details": exc.details,
                    "request_id": request.state.request_id
                }
            )
        
        # Gérer les erreurs HTTP FastAPI
        if isinstance(exc, HTTPException):
            counter("api.errors", tags={
                "type": "http_exception",
                "status": exc.status_code
            })
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "code": "HTTP_ERROR",
                    "message": exc.detail,
                    "request_id": request.state.request_id
                }
            )
        
        # Gérer les autres erreurs
        logger.exception("Unhandled error")
        counter("api.errors", tags={
            "type": "internal_error",
            "status": 500
        })
        return JSONResponse(
            status_code=500,
            content={
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "Internal server error",
                "request_id": request.state.request_id
            }
        )
        
    except Exception as e:
        logger.exception("Error in error handler")
        return JSONResponse(
            status_code=500,
            content={
                "code": ErrorCode.INTERNAL_ERROR.value,
                "message": "Critical error in error handler",
                "request_id": getattr(request.state, "request_id", None)
            }
        )

# Documentation des erreurs
ERROR_RESPONSES = {
    400: {
        "description": "Validation Error",
        "content": {
            "application/json": {
                "example": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": {
                        "field": "image",
                        "error": "File too large"
                    }
                }
            }
        }
    },
    401: {
        "description": "Unauthorized",
        "content": {
            "application/json": {
                "example": {
                    "code": "UNAUTHORIZED",
                    "message": "Invalid API key"
                }
            }
        }
    },
    404: {
        "description": "Not Found",
        "content": {
            "application/json": {
                "example": {
                    "code": "NOT_FOUND",
                    "message": "Resource not found"
                }
            }
        }
    },
    409: {
        "description": "Concurrency Error",
        "content": {
            "application/json": {
                "example": {
                    "code": "CONCURRENCY_ERROR",
                    "message": "Task already running"
                }
            }
        }
    },
    429: {
        "description": "Rate Limit Error",
        "content": {
            "application/json": {
                "example": {
                    "code": "RATE_LIMIT_ERROR",
                    "message": "Too many requests"
                }
            }
        }
    },
    500: {
        "description": "Internal Server Error",
        "content": {
            "application/json": {
                "example": {
                    "code": "INTERNAL_ERROR",
                    "message": "Internal server error"
                }
            }
        }
    }
}
