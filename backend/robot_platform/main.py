import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .robotics import validate_motion
from .schemas import Project, RecheckRequest, RecheckResult, SavedProject
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
        "safety_resolution": project.safety_resolution,
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
    payload = project.model_dump()
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_validated_project(project: Project, project_id: str | None = None) -> SavedProject:
    validation = None
    error = None
    try:
        validation = run_validation(project)
        executable = bool(validation["valid"])
    except ValueError as exc:
        executable = False
        error = str(exc)
    revisions: list[dict] = []
    if project_id:
        try:
            previous = store.get(project_id)
            revisions = previous.revisions
        except FileNotFoundError:
            revisions = []
    saved = SavedProject(
        id=project_id or "",
        executable=executable,
        planned_events=validation.get("planned_events", []) if validation else [],
        validation_digest=project_digest(project),
        last_validation=validation,
        validation_error=error,
        revisions=revisions,
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


def _revalidation_summary(validation: dict | None, error: str | None) -> dict:
    if validation is None:
        return {"valid": False, "error": error}
    return {
        "valid": bool(validation["valid"]),
        "checked_segments": list(validation.get("checked_segments", [])),
        "danger_intervals": list(validation.get("danger_intervals", [])),
        "first_collision": validation.get("first_collision"),
        "replay_window": validation.get("replay_window"),
        "constraint_version": validation.get("constraint_version"),
        "safety_resolution": validation.get("safety_resolution"),
        "duration": validation.get("duration"),
    }


@app.post("/api/projects/{project_id}/recheck", response_model=RecheckResult)
def recheck_segments(project_id: str, request: RecheckRequest) -> RecheckResult:
    """Re-verify only the edited segments after a local trajectory edit.

    Segment durations depend only on the two bounding waypoints, so later
    segments keep their timings. Conclusions on the edited segments and every
    downstream segment are invalidated; earlier certified segments stay valid.
    The previous plan is retained as a revision together with its constraint
    version and the ranges already checked.
    """

    try:
        saved = store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc

    project = Project.model_validate(
        saved.model_dump(
            exclude={
                "id", "executable", "planned_events", "validation_digest",
                "last_validation", "validation_error", "revisions",
            }
        )
    )
    segment_count = len(project.waypoints) - 1
    try:
        checked = sorted({int(index) for index in request.segments})
        if not checked or any(not 0 <= index < segment_count for index in checked):
            raise ValueError("segments must reference existing trajectory segments")
        if request.safety_resolution is not None:
            project.safety_resolution = request.safety_resolution
        validation = validate_motion(
            project.waypoints,
            project.speed_limits,
            [(pair[0], pair[1]) for pair in project.joint_limits],
            project.link_lengths,
            [obstacle.model_dump() for obstacle in project.obstacles],
            project.link_radius,
            project.sample_dt,
            recheck_segments=checked,
            **project_validation_kwargs(project),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    first_edited = checked[0]
    invalidated = list(range(first_edited, segment_count))
    rechecked_clear = [index for index in checked if index not in {
        interval["segment"] for interval in validation["danger_intervals"]
    }]
    revision = {
        "revision_id": project_digest(saved)[:12],
        "created_from_digest": saved.validation_digest,
        "constraint_version": saved.constraint_version,
        "checked_segments": checked,
        "rechecked_clear_segments": rechecked_clear,
        "invalidated_segments": invalidated,
        "danger_intervals": list(validation["danger_intervals"]),
        "first_collision": validation["first_collision"],
        "project_snapshot": project.model_dump(),
        "previous_summary": _revalidation_summary(saved.last_validation, saved.validation_error),
    }
    revisions = [*saved.revisions, revision]
    updated = saved.model_copy(
        update={
            "executable": False,
            "planned_events": validation.get("planned_events", []),
            "validation_digest": project_digest(project),
            "last_validation": validation,
            "validation_error": None,
            "revisions": revisions,
        }
    )
    store.save(updated, project_id)
    return RecheckResult(
        checked_segments=checked,
        invalidated_segments=invalidated,
        validation=validation,
        revision=revision,
    )
