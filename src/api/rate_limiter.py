"""
Rate limiting implementation.
"""
import time
import asyncio
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from fastapi import HTTPException, Request
from metrics import gauge, counter

logger = logging.getLogger(__name__)

@dataclass
class RateLimit:
    """Rate limit configuration"""
    requests: int  # Nombre max de requêtes
    window: int   # Fenêtre de temps en secondes
    burst: int    # Burst autorisé

class TokenBucket:
    """Token bucket implementation"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> Tuple[bool, float]:
        """Try to acquire tokens"""
        async with self._lock:
            now = time.monotonic()
            # Ajouter les nouveaux tokens
            elapsed = now - self.last_update
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            # Vérifier si assez de tokens
            if tokens <= self.tokens:
                self.tokens -= tokens
                return True, 0.0
            else:
                # Calculer le temps d'attente
                wait_time = (tokens - self.tokens) / self.rate
                return False, wait_time

class RateLimiter:
    """Rate limiter implementation"""
    
    def __init__(self):
        # Limites par défaut
        self.default_limits = {
            "anonymous": RateLimit(100, 3600, 10),   # 100/hour, burst 10
            "authenticated": RateLimit(1000, 3600, 50)  # 1000/hour, burst 50
        }
        
        # Buckets par IP
        self.buckets: Dict[str, TokenBucket] = {}
        
        # Cache des IPs bannies
        self.banned_ips: Dict[str, float] = {}
        
        # Métriques
        gauge("api.rate_limiter.buckets", len(self.buckets))
    
    def _get_bucket(self, ip: str, authenticated: bool) -> TokenBucket:
        """Get or create bucket for IP"""
        if ip not in self.buckets:
            limit = self.default_limits["authenticated" if authenticated else "anonymous"]
            self.buckets[ip] = TokenBucket(
                rate=limit.requests / limit.window,
                capacity=limit.burst
            )
            gauge("api.rate_limiter.buckets", len(self.buckets))
        return self.buckets[ip]
    
    def _is_banned(self, ip: str) -> bool:
        """Check if IP is banned"""
        if ip in self.banned_ips:
            ban_time = self.banned_ips[ip]
            if time.monotonic() >= ban_time:
                del self.banned_ips[ip]
                return False
            return True
        return False
    
    def _ban_ip(self, ip: str, duration: int = 3600):
        """Ban an IP for duration seconds"""
        self.banned_ips[ip] = time.monotonic() + duration
        counter("api.rate_limiter.bans", 1)
        logger.warning(f"IP banned: {ip}")
    
    async def check_rate_limit(self, request: Request) -> None:
        """Check rate limit for request"""
        try:
            # Obtenir l'IP
            ip = request.client.host
            
            # Vérifier si IP bannie
            if self._is_banned(ip):
                counter("api.rate_limiter.blocked_requests", 1)
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. IP temporarily banned."
                )
            
            # Vérifier l'authentification
            authenticated = bool(request.headers.get("X-API-Key"))
            
            # Obtenir le bucket
            bucket = self._get_bucket(ip, authenticated)
            
            # Essayer d'acquérir un token
            allowed, wait_time = await bucket.acquire()
            
            if not allowed:
                # Incrémenter le compteur d'erreurs
                counter("api.rate_limiter.exceeded_requests", 1)
                
                # Vérifier si on doit bannir
                if wait_time > 60:  # Plus d'une minute d'attente
                    self._ban_ip(ip)
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests. IP temporarily banned."
                    )
                
                # Sinon juste rate limit
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds."
                )
            
            # Métriques
            gauge("api.rate_limiter.wait_time", wait_time)
            
        except HTTPException:
            raise
            
        except Exception as e:
            logger.error(f"Rate limiting failed: {e}")
            # En cas d'erreur, on laisse passer
            pass
    
    async def cleanup(self):
        """Clean up old buckets and bans"""
        try:
            now = time.monotonic()
            
            # Nettoyer les buckets inactifs
            for ip in list(self.buckets.keys()):
                bucket = self.buckets[ip]
                if now - bucket.last_update > 3600:  # 1 heure d'inactivité
                    del self.buckets[ip]
            
            # Nettoyer les bans expirés
            for ip in list(self.banned_ips.keys()):
                if now >= self.banned_ips[ip]:
                    del self.banned_ips[ip]
            
            # Métriques
            gauge("api.rate_limiter.buckets", len(self.buckets))
            gauge("api.rate_limiter.bans", len(self.banned_ips))
            
        except Exception as e:
            logger.error(f"Rate limiter cleanup failed: {e}")

# Instance globale
rate_limiter = RateLimiter()
