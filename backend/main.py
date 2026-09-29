from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .robotics import default_project, simulate


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECTS_DIR = BASE_DIR / "data" / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="机械臂离线编程平台", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    robot: dict
    max_joint_speeds: list[float]
    waypoints: list[dict]
    obstacles: list[dict] = []


def project_path(project_id: str) -> Path:
    if not project_id.replace("-", "").isalnum() or len(project_id) > 36:
        raise HTTPException(status_code=400, detail="非法项目 ID")
    path = (PROJECTS_DIR / f"{project_id}.json").resolve()
    if PROJECTS_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="非法项目 ID")
    return path


def read_project(project_id: str) -> dict:
    path = project_path(project_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="项目不存在")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def write_project(project_id: str, payload: SaveRequest) -> dict:
    import json

    project = payload.model_dump()
    project["id"] = project_id
    project_path(project_id).write_text(
        json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return project


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/default-project")
def get_default_project() -> dict:
    return default_project()


@app.get("/api/projects")
def list_projects() -> list[dict]:
    import json

    projects = []
    for path in sorted(PROJECTS_DIR.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        projects.append({"id": data["id"], "name": data["name"]})
    return projects


@app.post("/api/projects")
def create_project(payload: SaveRequest) -> dict:
    return write_project(str(uuid4()), payload)


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    return read_project(project_id)


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, payload: SaveRequest) -> dict:
    read_project(project_id)
    return write_project(project_id, payload)


@app.post("/api/projects/{project_id}/duplicate")
def duplicate_project(project_id: str) -> dict:
    source = read_project(project_id)
    source.pop("id", None)
    return write_project(str(uuid4()), SaveRequest(**source))


@app.post("/api/simulate")
def run_simulation(payload: SaveRequest) -> dict:
    try:
        return simulate(payload.model_dump())
    except (ValueError, IndexError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
