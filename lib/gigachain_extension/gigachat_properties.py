from enum import Enum
from typing import Optional

from pydantic import BaseModel


class GigaChatScopeType(str, Enum):
    GIGACHAT_API_CORP = "GIGACHAT_API_CORP"
    GIGACHAT_API_PERS = "GIGACHAT_API_PERS"

class ConnectionPoolProperties(BaseModel):
    max_keepalive_connections: Optional[int] = None
    max_connections: Optional[int] = None
    keepalive_expiry: Optional[float] = None

class GigaChatProperties(BaseModel):
    class Config:
        use_enum_values = True

    base_url: Optional[str] = None
    model: Optional[str] = None
    auth_url: Optional[str] = None
    credentials: Optional[str] = None
    access_token: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    scope: Optional[GigaChatScopeType] = None  # GIGACHAT_API_CORP, GIGACHAT_API_PERS enum
    timeout: Optional[float] = None
    verify_ssl_certs: Optional[bool] = None
    ca_bundle_file: Optional[str] = None
    cert_file: Optional[str] = None
    key_file: Optional[str] = None
    key_file_password: Optional[str] = None
    profanity_check: Optional[bool] = None
    streaming: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    top_p: Optional[float] = None
    verbose: Optional[bool] = True

    connection_pool: Optional[ConnectionPoolProperties] = None
