import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .robotics import validate_motion
from .schemas import Project, SavedProject, ValidationRequest
from .storage import ProjectStore

app = FastAPI(title="Offline Robot Programming Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
store = ProjectStore(Path(__file__).resolve().parents[2] / "projects")


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
