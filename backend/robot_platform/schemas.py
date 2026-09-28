from pydantic import BaseModel, Field


class Obstacle(BaseModel):
    name: str = "obstacle"
    x: float
    y: float
    radius: float = Field(gt=0)


class Project(BaseModel):
    name: str = "untitled"
    waypoints: list[list[float]]
    speed_limits: list[float]
    joint_limits: list[list[float]]
    link_lengths: list[float]
    obstacles: list[Obstacle]
    link_radius: float = 0.08
    sample_dt: float = Field(default=0.02, gt=0)


class SavedProject(Project):
    id: str
