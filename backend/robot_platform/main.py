import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .robotics import validate_motion
from .schemas import Project, SavedProject, SavedTask, TaskDefinition, TaskValidationRequest, ValidationRequest
from .storage import ProjectStore, TaskStore
from .taskchain import validate_task

app = FastAPI(title="Offline Robot Programming Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
store = ProjectStore(Path(__file__).resolve().parents[2] / "projects")
task_store = TaskStore(Path(__file__).resolve().parents[2] / "tasks")


def project_validation_kwargs(project: Project) -> dict:
    return {
        "segment_speed_limits": project.segment_speed_limits,
        "acceleration_limits": project.acceleration_limits,
        "velocity_transition_time": project.velocity_transition_time,
        "gripper_radius": project.gripper_radius,
        "carried_objects": [item.model_dump() for item in project.carried_objects],
        "pick_events": [item.model_dump() for item in project.pick_events],
        "place_events": [item.model_dump() for item in project.place_events],
        "constraint_version": project.constraint_version,
        "wait_duration": project.wait_duration,
        "reviewed_segments": project.reviewed_segments,
        "previous_validation_digest": project.previous_validation_digest,
    }


def run_validation(project: Project) -> dict:
    return validate_motion(
        project.waypoints,
        project.speed_limits,
        [(pair[0], pair[1]) for pair in project.joint_limits],
        project.link_lengths,
        [obstacle.model_dump() for obstacle in project.obstacles],
        project.link_radius,
        project.sample_dt,
        **project_validation_kwargs(project),
    )


def project_digest(project: Project) -> str:
    payload = project.model_dump(exclude={"baseline_snapshot"})
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_task_validation(task: TaskDefinition, previous: dict | None = None) -> dict:
    return validate_task(
        task.model_dump(),
        failures=[failure.model_dump() for failure in task.failures],
        previous=previous,
    )


def task_digest(task: TaskDefinition) -> str:
    encoded = json.dumps(task.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_validated_task(task: TaskDefinition, task_id: str | None = None) -> SavedTask:
    previous_sources = None
    if task_id:
        try:
            stored = task_store.get(task_id)
        except FileNotFoundError:
            stored = None
        if stored is not None and stored.last_task_validation:
            previous_sources = stored.last_task_validation.get("conclusion_sources")
    validation = None
    error = None
    try:
        validation = run_task_validation(task, previous=previous_sources)
        executable = bool(validation["valid"])
    except ValueError as exc:
        executable = False
        error = str(exc)
    saved = SavedTask(
        id=task_id or "",
        executable=executable,
        validation_digest=task_digest(task),
        last_task_validation=validation,
        validation_error=error,
        **task.model_dump(),
    )
    return task_store.save(saved, task_id)


def save_validated_project(project: Project, project_id: str | None = None) -> SavedProject:
    previous = None
    if project_id:
        try:
            previous = store.get(project_id)
        except FileNotFoundError:
            previous = None
    validation = None
    error = None
    try:
        if previous is not None and project.baseline_snapshot is None:
            project.baseline_snapshot = previous.last_validation
        validation = run_validation(project)
        executable = bool(validation["valid"])
    except ValueError as exc:
        executable = False
        error = str(exc)
    saved = SavedProject(
        id=project_id or "",
        executable=executable,
        planned_events=validation.get("planned_events", []) if validation else [],
        validation_digest=project_digest(project),
        last_validation=validation,
        validation_error=error,
        **project.model_dump(),
    )
    return store.save(saved, project_id)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/validate", response_model=dict)
def validate_project(project: Project) -> dict:
    try:
        return run_validation(project)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/validate-segments", response_model=dict)
def validate_project_segments(request: ValidationRequest) -> dict:
    request.project.reviewed_segments = request.reviewed_segments
    request.project.previous_validation_digest = request.previous_validation_digest
    return validate_project(request.project)


@app.get("/api/projects", response_model=list[SavedProject])
def list_projects() -> list[SavedProject]:
    return store.list()


@app.post("/api/projects", response_model=SavedProject)
def save_project(project: Project) -> SavedProject:
    return save_validated_project(project)


@app.put("/api/projects/{project_id}", response_model=SavedProject)
def update_project(project_id: str, project: Project) -> SavedProject:
    return save_validated_project(project, project_id)


@app.get("/api/projects/{project_id}", response_model=SavedProject)
def get_project(project_id: str) -> SavedProject:
    try:
        saved = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc

    project = Project.model_validate(saved.model_dump(exclude={"id", "executable", "planned_events", "validation_digest", "last_validation", "validation_error"}))
    try:
        validation = run_validation(project)
        revalidated = saved.model_copy(
            update={
                "executable": bool(validation["valid"]),
                "planned_events": validation.get("planned_events", []),
                "validation_digest": project_digest(project),
                "last_validation": validation,
                "validation_error": None,
            }
        )
    except ValueError as exc:
        revalidated = saved.model_copy(
            update={
                "executable": False,
                "planned_events": [],
                "validation_digest": project_digest(project),
                "last_validation": None,
                "validation_error": str(exc),
            }
        )
    return store.save(revalidated, project_id)


@app.get("/api/projects/{project_id}/replay", response_model=dict)
def replay_project(project_id: str) -> dict:
    try:
        saved = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
    project = Project.model_validate(saved.model_dump(exclude={"id", "executable", "planned_events", "validation_digest", "last_validation", "validation_error"}))
    try:
        validation = run_validation(project)
        saved = saved.model_copy(
            update={
                "executable": bool(validation["valid"]),
                "planned_events": validation.get("planned_events", []),
                "validation_digest": project_digest(project),
                "last_validation": validation,
                "validation_error": None,
            }
        )
    except ValueError as exc:
        validation = {"valid": False, "error": str(exc)}
        saved = saved.model_copy(
            update={
                "executable": False,
                "planned_events": [],
                "validation_digest": project_digest(project),
                "last_validation": None,
                "validation_error": str(exc),
            }
        )
    store.save(saved, project_id)
    return validation


@app.post("/api/task/validate", response_model=dict)
def validate_chained_task(request: TaskValidationRequest) -> dict:
    try:
        return run_task_validation(request.task, previous=request.previous)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/tasks", response_model=list[SavedTask])
def list_tasks() -> list[SavedTask]:
    return task_store.list()


@app.post("/api/tasks", response_model=SavedTask)
def save_task(task: TaskDefinition) -> SavedTask:
    return save_validated_task(task)


@app.put("/api/tasks/{task_id}", response_model=SavedTask)
def update_task(task_id: str, task: TaskDefinition) -> SavedTask:
    return save_validated_task(task, task_id)


@app.get("/api/tasks/{task_id}", response_model=SavedTask)
def get_task(task_id: str) -> SavedTask:
    try:
        saved = task_store.get(task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="task not found") from exc
    task = TaskDefinition.model_validate(
        saved.model_dump(exclude={"id", "executable", "validation_digest", "last_task_validation", "validation_error"})
    )
    previous_sources = None
    if saved.last_task_validation:
        previous_sources = saved.last_task_validation.get("conclusion_sources")
    try:
        validation = run_task_validation(task, previous=previous_sources)
        saved = saved.model_copy(
            update={
                "executable": bool(validation["valid"]),
                "validation_digest": task_digest(task),
                "last_task_validation": validation,
                "validation_error": None,
            }
        )
    except ValueError as exc:
        saved = saved.model_copy(
            update={
                "executable": False,
                "validation_digest": task_digest(task),
                "last_task_validation": None,
                "validation_error": str(exc),
            }
        )
    return task_store.save(saved, task_id)


@app.get("/api/tasks/{task_id}/replay", response_model=dict)
def replay_task(task_id: str) -> dict:
    saved = get_task(task_id)
    if saved.last_task_validation is None:
        return {"valid": False, "error": saved.validation_error}
    return saved.last_task_validation
