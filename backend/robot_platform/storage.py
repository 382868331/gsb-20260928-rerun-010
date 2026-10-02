from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .schemas import Project, SavedProject, SavedTask, TaskDefinition


class ProjectStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, project: Project, project_id: str | None = None) -> SavedProject:
        saved_id = project_id or uuid4().hex
        saved = SavedProject(id=saved_id, **project.model_dump(exclude={"id"}))
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


class TaskStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, task: TaskDefinition, task_id: str | None = None) -> SavedTask:
        saved_id = task_id or uuid4().hex
        saved = SavedTask(id=saved_id, **task.model_dump(exclude={"id"}))
        (self.directory / f"{saved_id}.json").write_text(
            json.dumps(saved.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return saved

    def get(self, task_id: str) -> SavedTask:
        path = self.directory / f"{task_id}.json"
        return SavedTask.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[SavedTask]:
        tasks = [
            SavedTask.model_validate(json.loads(path.read_text(encoding="utf-8")))
            for path in self.directory.glob("*.json")
        ]
        return sorted(tasks, key=lambda task: task.name)
