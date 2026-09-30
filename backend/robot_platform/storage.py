from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import Project, SavedProject


class ProjectStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, project: Project, project_id: str | None = None, verdict: dict | None = None) -> SavedProject:
        saved_id = project_id or uuid4().hex
        data = project.model_dump(exclude={"id"})
        if verdict is not None:
            data.update(
                {
                    "executable": bool(verdict["valid"]),
                    "constraint_version": verdict["constraint_version"],
                    "grasp_events": verdict["grasp_events"],
                    "release_events": verdict["release_events"],
                    "segments_solved": verdict["segments_solved"],
                    "validated_duration": verdict["duration"],
                }
            )
        saved = SavedProject(id=saved_id, **data)
        (self.directory / f"{saved_id}.json").write_text(
            json.dumps(saved.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return saved

    def get(self, project_id: str) -> SavedProject:
        path = self.directory / f"{project_id}.json"
        return SavedProject.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[SavedProject]:
        projects = [
            SavedProject.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in self.directory.glob("*.json")
        ]
        return sorted(projects, key=lambda project: project.name)
