from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
import numpy as np


@dataclass(frozen=True)
class Collision:
    time: float
    segment: int
    kind: str
    message: str


def _as_vector(values: list[float], name: str, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains a non-finite value")
    return arr


def segment_duration(delta: np.ndarray, speed_limits: np.ndarray) -> float:
    if np.any(speed_limits <= 0):
        raise ValueError("speed limits must be positive")
    return float(np.max(np.abs(delta) / speed_limits))


def normalize_segment_limits(
    segment_count: int,
    dof: int,
    speed_limits: list[float],
    segment_speed_limits: list[list[float]] | None = None,
) -> list[np.ndarray]:
    default_limits = _as_vector(speed_limits, "speed_limits", dof)
    if segment_speed_limits is None:
        return [default_limits.copy() for _ in range(segment_count)]
    if len(segment_speed_limits) != segment_count:
        raise ValueError("segment_speed_limits must contain one speed list per trajectory segment")
    return [
        _as_vector(limits, f"segment_speed_limits[{index}]", dof)
        for index, limits in enumerate(segment_speed_limits)
    ]


def interpolate_trajectory(
    waypoints: list[list[float]],
    speed_limits: list[float],
    sample_dt: float = 0.02,
    segment_speed_limits: list[list[float]] | None = None,
) -> tuple[list[float], list[np.ndarray], list[float]]:
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")

    dof = len(waypoints[0])
    points = [_as_vector(values, "waypoint", dof) for values in waypoints]
    limits_by_segment = normalize_segment_limits(len(points) - 1, dof, speed_limits, segment_speed_limits)
    durations = [
        segment_duration(points[index + 1] - points[index], limits_by_segment[index])
        for index in range(len(points) - 1)
    ]

    times = [0.0]
    samples = [points[0].copy()]
    elapsed = 0.0
    for segment, duration in enumerate(durations):
        if duration == 0.0:
            continue
        start = points[segment]
        end = points[segment + 1]
        count = max(1, int(np.ceil(duration / sample_dt)))
        for index in range(1, count + 1):
            local_time = min(duration, index * sample_dt)
            current_time = elapsed + local_time
            if abs(current_time - times[-1]) <= 1e-9:
                continue
            alpha = local_time / duration
            times.append(current_time)
            samples.append(start + alpha * (end - start))
        elapsed += duration

    return times, samples, durations


def segment_velocities(points: list[np.ndarray], durations: list[float]) -> list[np.ndarray]:
    velocities = []
    for start, end, duration in zip(points, points[1:], durations):
        velocities.append(np.zeros_like(start) if duration == 0.0 else (end - start) / duration)
    return velocities


def forward_kinematics(joint_angles: np.ndarray, link_lengths: list[float]) -> list[tuple[float, float]]:
    if len(joint_angles) != len(link_lengths):
        raise ValueError("joint_angles and link_lengths must have the same length")
    vertices = [(0.0, 0.0)]
    x = y = angle = 0.0
    for joint_angle, length in zip(joint_angles, link_lengths):
        angle += float(joint_angle)
        x += length * cos(angle)
        y += length * sin(angle)
        vertices.append((x, y))
    return vertices


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return sqrt(wx * wx + wy * wy)
    projection = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    dx = px - (ax + projection * vx)
    dy = py - (ay + projection * vy)
    return sqrt(dx * dx + dy * dy)


def capsule_distance(
    first: tuple[float, float],
    second: tuple[float, float],
    third: tuple[float, float],
    fourth: tuple[float, float],
) -> float:
    return min(
        point_segment_distance(*first, *third, *fourth),
        point_segment_distance(*second, *third, *fourth),
        point_segment_distance(*third, *first, *second),
        point_segment_distance(*fourth, *first, *second),
    )


def validate_motion(
    waypoints: list[list[float]],
    speed_limits: list[float],
    joint_limits: list[tuple[float, float]],
    link_lengths: list[float],
    obstacles: list[dict[str, float]],
    link_radius: float = 0.08,
    sample_dt: float = 0.02,
    segment_speed_limits: list[list[float]] | None = None,
    acceleration_limits: list[float] | None = None,
    velocity_transition_time: float = 0.2,
    gripper_radius: float = 0.06,
    carried_objects: list[dict] | None = None,
    pick_events: list[dict] | None = None,
    place_events: list[dict] | None = None,
    constraint_version: str = "1.1",
) -> dict:
    dof = len(link_lengths)
    limits = _as_vector(speed_limits, "speed_limits", dof)
    acceleration = _as_vector(
        acceleration_limits or [max(2.0, float(value)) for value in limits],
        "acceleration_limits",
        dof,
    )
    bounds = np.asarray(joint_limits, dtype=float)
    if bounds.shape != (dof, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("joint_limits must be [minimum, maximum] pairs")
    if link_radius <= 0:
        raise ValueError("link_radius must be positive")
    if gripper_radius <= 0:
        raise ValueError("gripper_radius must be positive")
    if velocity_transition_time <= 0:
        raise ValueError("velocity_transition_time must be positive")
    if np.any(acceleration <= 0):
        raise ValueError("acceleration limits must be positive")

    objects_by_id = {str(item["id"]): item for item in (carried_objects or [])}
    if len(objects_by_id) != len(carried_objects or []):
        raise ValueError("carried object ids must be unique")
    pick_events = pick_events or []
    place_events = place_events or []
    pick_keys = set()
    place_keys = set()
    for event in [*pick_events, *place_events]:
        if str(event["object_id"]) not in objects_by_id:
            raise ValueError(f"event references unknown object {event['object_id']}")
        index = int(event["waypoint_index"])
        if not 0 <= index <= len(waypoints) - 1:
            raise ValueError("event waypoint_index must reference an existing waypoint")
    for event in pick_events:
        key = (int(event["waypoint_index"]), str(event["object_id"]))
        if key in pick_keys:
            raise ValueError("duplicate pick event")
        pick_keys.add(key)
    for event in place_events:
        key = (int(event["waypoint_index"]), str(event["object_id"]))
        if key in place_keys:
            raise ValueError("duplicate place event")
        place_keys.add(key)
    active_objects: set[str] = set()
    for index in range(len(waypoints)):
        current_picks = {str(event["object_id"]) for event in pick_events if int(event["waypoint_index"]) == index}
        current_places = {str(event["object_id"]) for event in place_events if int(event["waypoint_index"]) == index}
        if current_picks & active_objects:
            raise ValueError("object cannot be picked while it is already carried")
        if current_places - active_objects:
            raise ValueError("object cannot be placed before it has been picked")
        active_objects.difference_update(current_places)
        active_objects.update(current_picks)
        if len(active_objects) > 1:
            raise ValueError("only one object may be carried at a time")

    times, samples, segment_durations = interpolate_trajectory(
        waypoints, speed_limits, sample_dt, segment_speed_limits
    )
    points = [np.asarray(values, dtype=float) for values in waypoints]
    velocities = segment_velocities(points, segment_durations)
    boundaries = np.cumsum([0.0, *segment_durations[:-1]])
    collisions: list[Collision] = []
    frames: list[dict] = []

    for segment_index in range(len(velocities) - 1):
        velocity_change = np.abs(velocities[segment_index + 1] - velocities[segment_index])
        allowed_change = acceleration * velocity_transition_time
        for joint in np.where(velocity_change > allowed_change + 1e-9)[0]:
            collisions.append(
                Collision(
                    float(boundaries[segment_index + 1]),
                    segment_index + 1,
                    "velocity_transition",
                    f"joint {int(joint) + 1} changes speed too quickly between segments {segment_index + 1} and {segment_index + 2}",
                )
            )

    picks_by_waypoint: dict[int, list[str]] = {}
    places_by_waypoint: dict[int, list[str]] = {}
    for event in pick_events:
        picks_by_waypoint.setdefault(int(event["waypoint_index"]), []).append(str(event["object_id"]))
    for event in place_events:
        places_by_waypoint.setdefault(int(event["waypoint_index"]), []).append(str(event["object_id"]))
    planned_events: list[dict] = []
    elapsed = 0.0
    for index in range(len(points)):
        for object_id in places_by_waypoint.get(index, []):
            planned_events.append({"type": "place", "waypoint_index": index, "object_id": object_id, "time": elapsed})
        for object_id in picks_by_waypoint.get(index, []):
            planned_events.append({"type": "pick", "waypoint_index": index, "object_id": object_id, "time": elapsed})
        if index < len(segment_durations):
            elapsed += segment_durations[index]

    def active_object_at(segment: int, time: float) -> str | None:
        active: set[str] = set()
        boundary_time = 0.0
        for index, duration in enumerate(segment_durations):
            for object_id in picks_by_waypoint.get(index, []):
                active.add(object_id)
            boundary_time += duration
            for object_id in places_by_waypoint.get(index + 1, []):
                if index < segment or time >= boundary_time - 1e-9:
                    active.discard(object_id)
            if index == segment:
                break
        if len(active) > 1:
            raise ValueError("only one object may be carried at a time")
        return next(iter(active), None)

    for time, angles in zip(times, samples):
        segment = int(np.searchsorted(boundaries, time, side="right") - 1)
        segment = min(segment, max(0, len(segment_durations) - 1))

        for joint in np.where((angles < bounds[:, 0]) | (angles > bounds[:, 1]))[0]:
            collisions.append(Collision(float(time), segment, "joint_limit", f"joint {int(joint) + 1} exceeded its limit"))

        vertices = forward_kinematics(angles, link_lengths)
        active_object_id = active_object_at(segment, float(time))
        active_object = objects_by_id[active_object_id] if active_object_id is not None else None
        end_x, end_y = vertices[-1]
        end_orientation = float(np.sum(angles))
        object_position: tuple[float, float] | None = None
        for obstacle in obstacles:
            ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
            for link_index in range(len(vertices) - 1):
                distance = point_segment_distance(ox, oy, *vertices[link_index], *vertices[link_index + 1])
                if distance < radius + link_radius:
                    name = obstacle.get("name", f"obstacle-{link_index + 1}")
                    collisions.append(Collision(float(time), segment, "obstacle", f"link {link_index + 1} collided with obstacle {name}"))

            end_distance = sqrt((ox - end_x) ** 2 + (oy - end_y) ** 2)
            if end_distance < radius + gripper_radius:
                name = obstacle.get("name", "obstacle")
                collisions.append(Collision(float(time), segment, "gripper_obstacle", f"gripper collided with obstacle {name}"))

        if active_object is not None:
            offset_x = float(active_object.get("offset_x", active_object.get("offset", [0.0, 0.0])[0]))
            offset_y = float(active_object.get("offset_y", active_object.get("offset", [0.0, 0.0])[1]))
            object_radius = float(active_object["radius"])
            rotated_x = offset_x * cos(end_orientation) - offset_y * sin(end_orientation)
            rotated_y = offset_x * sin(end_orientation) + offset_y * cos(end_orientation)
            object_position = (end_x + rotated_x, end_y + rotated_y)
            for obstacle in obstacles:
                ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
                distance = sqrt((ox - object_position[0]) ** 2 + (oy - object_position[1]) ** 2)
                if distance < radius + object_radius:
                    name = obstacle.get("name", "obstacle")
                    collisions.append(
                        Collision(float(time), segment, "carried_object_obstacle", f"carried object {active_object_id} collided with obstacle {name}")
                    )

        for left in range(len(vertices) - 1):
            for right in range(left + 2, len(vertices) - 1):
                distance = capsule_distance(vertices[left], vertices[left + 1], vertices[right], vertices[right + 1])
                if distance < link_radius * 2:
                    collisions.append(Collision(float(time), segment, "self_collision", f"links {left + 1} and {right + 1} collided"))

        frames.append(
            {
                "time": float(time),
                "joint_angles": [float(value) for value in angles],
                "links": [[float(x), float(y)] for x, y in vertices],
                "carried_object_id": active_object_id,
                "object_position": None if object_position is None else [float(object_position[0]), float(object_position[1])],
            }
        )

    first = collisions[0] if collisions else None
    return {
        "valid": not collisions,
        "duration": float(times[-1]),
        "segment_durations": [float(value) for value in segment_durations],
        "frames": frames,
        "planned_events": planned_events,
        "constraint_version": constraint_version,
        "collisions": [collision.__dict__ for collision in collisions[:20]],
        "first_collision": None if first is None else first.__dict__,
    }
