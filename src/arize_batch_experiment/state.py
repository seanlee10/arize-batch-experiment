import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunState:
    """Progress of one run, persisted so a restarted run never resubmits a batch job."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, runs_dir: Path, run_id: str) -> "RunState":
        path = runs_dir / run_id / "state.json"
        data = json.loads(path.read_text()) if path.exists() else {"run_id": run_id}
        return cls(path=path, data=data)

    @property
    def dir(self) -> Path:
        return self.path.parent

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, default=str))
        tmp.replace(self.path)
