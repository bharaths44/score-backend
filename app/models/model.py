from typing import Dict, List, Optional
from pydantic import BaseModel


class ProcessingResult(BaseModel):
    file_path: str
    text: Optional[str] = None
    classification: Optional[List[Dict[str, float]]] = None
