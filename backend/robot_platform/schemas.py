from pydantic import BaseModel, Field


class Obstacle(BaseModel):
    name: str = "obstacle"
    x: float
    y: float
    radius: float = Field(gt=0)


class CarriedObjectSpec(BaseModel):
    name: str
    x: float
    y: float
    radius: float = Field(gt=0)
    hold_dx: float = 0.0
    hold_dy: float = 0.0


class GripperAction(BaseModel):
    type: str = Field(pattern="^(grasp|release)$")
    waypoint_index: int = Field(ge=0)
    object: str


class Project(BaseModel):
    name: str = "untitled"
    waypoints: list[list[float]]
    speed_limits: list[float]
    segment_speed_limits: list[list[float]] | None = None
    joint_limits: list[list[float]]
    link_lengths: list[float]
    obstacles: list[Obstacle]
    carried_objects: list[CarriedObjectSpec] = []
    actions: list[GripperAction] = []
    link_radius: float = 0.08
    sample_dt: float = Field(default=0.02, gt=0)
    grasp_tolerance: float = Field(default=0.2, ge=0)
    junction_speed_tolerance: float = Field(default=0.15, ge=0)


class SavedProject(Project):
    id: str
    executable: bool = False
    constraint_version: str | None = None
    grasp_events: list[dict] = []
    release_events: list[dict] = []
    segments_solved: list[bool] = []
    validated_duration: float | None = None
