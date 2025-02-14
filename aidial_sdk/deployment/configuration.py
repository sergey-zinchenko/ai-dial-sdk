import fastapi

from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from aidial_sdk.pydantic_v1 import BaseModel
from aidial_sdk.utils.pydantic import ExtraForbidModel
from typing import Optional, Dict, Any

class ConfigurationRequestCustomFields(ExtraForbidModel):
    application_properties: Optional[Dict[str, Any]] = None

class ConfigurationRequest(FromRequestDeploymentMixin):
    custom_fields: Optional[ConfigurationRequestCustomFields] = None

class ConfigurationResponse(BaseModel):
    class Config:
        extra = "allow"