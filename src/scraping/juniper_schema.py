from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class JuniperTemplate:
    action: str
    domain: str
    sub_domain: str
    template: str
    requires_commit: bool
    default_params: Dict[str, Optional[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
