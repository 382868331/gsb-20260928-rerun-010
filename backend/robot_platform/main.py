from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .robotics import validate_motion
from .schemas import Project, SavedProject
from .storage import ProjectStore

app = FastAPI(title="Offline Robot Programming Platform", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
store = ProjectStore(Path(__file__).resolve().parents[2] / "projects")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/validate", response_model=dict)
def validate_project(project: Project) -> dict:
    try:
        return validate_motion(
            project.waypoints,
            project.speed_limits,
            [(pair[0], pair[1]) for pair in project.joint_limits],
            project.link_lengths,
            [obstacle.model_dump() for obstacle in project.obstacles],
            project.link_radius,
            project.sample_dt,
            [list(values) for values in project.segment_speed_limits] if project.segment_speed_limits else None,
            [obj.model_dump() for obj in project.carried_objects],
            [action.model_dump() for action in project.actions],
            project.grasp_tolerance,
            project.junction_speed_tolerance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/projects", response_model=list[SavedProject])
def list_projects() -> list[SavedProject]:
    return store.list()


@app.post("/api/projects", response_model=SavedProject)
def save_project(project: Project) -> SavedProject:
    return store.save(project, verdict=validate_project(project))


@app.put("/api/projects/{project_id}", response_model=SavedProject)
def update_project(project_id: str, project: Project) -> SavedProject:
    return store.save(project, project_id, verdict=validate_project(project))


@app.get("/api/projects/{project_id}", response_model=SavedProject)
def get_project(project_id: str) -> SavedProject:
    try:
        return store.get(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="project not found") from exc
