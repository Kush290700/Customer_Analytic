from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class FiltersPayload(BaseModel):
    start_date: Optional[str] = Field(default=None)
    end_date: Optional[str] = Field(default=None)
    regions: List[str] = Field(default_factory=list)
    methods: List[str] = Field(default_factory=list)
    customers: List[str] = Field(default_factory=list)

    @validator("regions", "methods", "customers", pre=True)
    def _coerce_list_str(cls, v):  # noqa: N805
        if v is None:
            return []
        if isinstance(v, (set, tuple)):
            return [str(x) for x in v]
        if isinstance(v, list):
            out: list[str] = []
            for x in v:
                if isinstance(x, dict):
                    name = x.get("name") or x.get("label") or x.get("value")
                    if name is not None:
                        s = str(name).strip()
                        if s:
                            out.append(s)
                        continue
                s = str(x).strip()
                if s:
                    out.append(s)
            # Treat common "all" sentinels as no filter
            if {s.lower() for s in out} & {"__all__", "all", "*"}:
                return []
            return out
        # comma-separated
        return [s.strip() for s in str(v).split(",") if s.strip()]
