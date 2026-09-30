from pydantic import BaseModel, Field


class Obstacle(BaseModel):
    name: str = "obstacle"
    x: float
    y: float
    radius: float = Field(gt=0)


class CarriedObject(BaseModel):
    id: str
    name: str
    radius: float = Field(gt=0)
    offset_x: float = 0.0
    offset_y: float = 0.0


class CarryEvent(BaseModel):
    waypoint_index: int = Field(ge=0)
    object_id: str


class Project(BaseModel):
    name: str = "untitled"
    waypoints: list[list[float]]
    speed_limits: list[float]
    joint_limits: list[list[float]]
    link_lengths: list[float]
    obstacles: list[Obstacle]
    carried_objects: list[CarriedObject] = []
    pick_events: list[CarryEvent] = []
    place_events: list[CarryEvent] = []
    segment_speed_limits: list[list[float]] | None = None
    acceleration_limits: list[float] | None = None
    velocity_transition_time: float = Field(default=0.2, gt=0)
    gripper_radius: float = Field(default=0.06, gt=0)
    constraint_version: str = "2.0"
    link_radius: float = 0.08
    sample_dt: float = Field(default=0.05, gt=0, description="display/replay frame step; never affects the safety verdict")
    safety_resolution: float = Field(default=0.005, gt=0, description="time resolution of the conservative swept-volume certification")


class SavedProject(Project):
    id: str
    executable: bool = False
    planned_events: list[dict] = []
    validation_digest: str = ""
    last_validation: dict | None = None
    validation_error: str | None = None
    revisions: list[dict] = []


class RecheckRequest(BaseModel):
    segments: list[int]
    safety_resolution: float | None = Field(default=None, gt=0)


class RecheckResult(BaseModel):
    checked_segments: list[int]
    invalidated_segments: list[int]
    validation: dict
    revision: dict
