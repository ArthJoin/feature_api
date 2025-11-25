from pydantic import BaseModel
from typing import List, Optional

import json
from uuid import UUID

class Response(BaseModel):
    score: int


"""
class Input(BaseModel):
    payload: dict[str, Any]

"""