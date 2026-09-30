from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt
import numpy as np


CONSTRAINT_VERSION = "grasp-multisegment-1"


@dataclass(frozen=True)
class Collision:
    time: float
    segment: int
    kind: str
    message: str


@dataclass
class CarriedObject:
    name: str
    x: float
    y: float
    radius: float
    hold_dx: float = 0.0
    hold_dy: float = 0.0

    def held_position(self, end_effector: tuple[float, float], tip: tuple[float, float]) -> tuple[float, float]:
        ax, ay = end_effector
        bx, by = tip
        direction_x = bx - ax
        direction_y = by - ay
        length = sqrt(direction_x * direction_x + direction_y * direction_y)
        if length == 0.0:
            return float(bx), float(by)
        tangent_x = direction_x / length
        tangent_y = direction_y / length
        normal_x = -tangent_y
        normal_y = tangent_x
        return (
            float(bx + self.hold_dx * tangent_x + self.hold_dy * normal_x),
            float(by + self.hold_dx * tangent_y + self.hold_dy * normal_y),
        )


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
    speed_limits: list[float],
    segment_speed_limits: list[list[float]] | None,
) -> np.ndarray:
    dof = len(speed_limits)
    base = _as_vector(speed_limits, "speed_limits", dof)
    if segment_speed_limits is None:
        return np.tile(base, (segment_count, 1))
    if len(segment_speed_limits) != segment_count:
        raise ValueError(
            f"segment_speed_limits must contain exactly {segment_count} entries for {segment_count + 1} waypoints"
        )
    rows = [_as_vector(values, "segment_speed_limits", dof) for values in segment_speed_limits]
    return np.vstack(rows)


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
    limits_per_segment = normalize_segment_limits(len(points) - 1, speed_limits, segment_speed_limits)
    durations = [
        segment_duration(points[index + 1] - points[index], limits_per_segment[index])
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


def segment_velocities(
    points: list[np.ndarray],
    durations: list[float],
    limits_per_segment: np.ndarray,
) -> np.ndarray:
    velocities = np.zeros((len(durations), len(points[0])))
    for index, duration in enumerate(durations):
        delta = points[index + 1] - points[index]
        if duration == 0.0:
            continue
        speeds = np.abs(delta) / duration
        joint = int(np.argmax(speeds))
        speeds = np.minimum(speeds, limits_per_segment[index])
        speeds[joint] = limits_per_segment[index, joint]
        velocities[index] = np.sign(delta) * speeds
    return velocities


def normalize_actions(actions: list[dict], objects: dict[str, CarriedObject], waypoint_count: int) -> list[dict]:
    normalized: list[dict] = []
    for action in actions:
        action_type = action.get("type")
        if action_type not in ("grasp", "release"):
            raise ValueError("each action type must be 'grasp' or 'release'")
        waypoint_index = int(action.get("waypoint_index", -1))
        if not 0 <= waypoint_index < waypoint_count:
            raise ValueError("action waypoint_index must point at an existing waypoint")
        object_name = action.get("object")
        if object_name not in objects:
            raise ValueError(f"action references unknown object {object_name!r}")
        normalized.append(
            {
                "type": action_type,
                "waypoint_index": waypoint_index,
                "object": object_name,
            }
        )
    normalized.sort(key=lambda item: (item["waypoint_index"], 0 if item["type"] == "release" else 1))
    held: set[str] = set()
    for action in normalized:
        name = action["object"]
        if action["type"] == "grasp":
            if name in held:
                raise ValueError(f"object {name!r} is grasped twice without a release")
            held.add(name)
        elif name not in held:
            raise ValueError(f"object {name!r} is released before being grasped")
        else:
            held.discard(name)
    return normalized


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
    carried_objects: list[dict] | None = None,
    actions: list[dict] | None = None,
    grasp_tolerance: float = 0.2,
    junction_speed_tolerance: float = 0.15,
) -> dict:
    dof = len(link_lengths)
    limits = _as_vector(speed_limits, "speed_limits", dof)
    bounds = np.asarray(joint_limits, dtype=float)
    if bounds.shape != (dof, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("joint_limits must be [minimum, maximum] pairs")
    if link_radius <= 0:
        raise ValueError("link_radius must be positive")
    if grasp_tolerance < 0:
        raise ValueError("grasp_tolerance must be non-negative")
    if junction_speed_tolerance < 0:
        raise ValueError("junction_speed_tolerance must be non-negative")

    objects = {
        spec["name"]: CarriedObject(
            name=str(spec["name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            radius=float(spec["radius"]),
            hold_dx=float(spec.get("hold_dx", 0.0)),
            hold_dy=float(spec.get("hold_dy", 0.0)),
        )
        for spec in (carried_objects or [])
    }
    if any(obj.radius <= 0 for obj in objects.values()):
        raise ValueError("carried object radius must be positive")
    actions = normalize_actions(actions or [], objects, len(waypoints))

    limits_per_segment = normalize_segment_limits(len(waypoints) - 1, speed_limits, segment_speed_limits)
    times, samples, segment_durations = interpolate_trajectory(
        waypoints, speed_limits, sample_dt, segment_speed_limits
    )
    points = [np.asarray(values, dtype=float) for values in waypoints]
    velocities = segment_velocities(points, segment_durations, limits_per_segment)
    boundaries = np.cumsum([0.0, *segment_durations[:-1]])
    collisions: list[Collision] = []
    frames: list[dict] = []
    segment_ok = [True] * len(segment_durations)
    grasp_events: list[dict] = []
    release_events: list[dict] = []
    held_names: set[str] = set()
    placements = {name: (obj.x, obj.y) for name, obj in objects.items()}

    def record(collision: Collision) -> None:
        collisions.append(collision)
        segment_ok[collision.segment] = False

    for action in actions:
        name = action["object"]
        obj = objects[name]
        index = action["waypoint_index"]
        event_time = float(sum(segment_durations[:index]))
        vertices = forward_kinematics(points[index], link_lengths)
        tip = vertices[-1]
        distance_to_tip = sqrt((tip[0] - obj.x) ** 2 + (tip[1] - obj.y) ** 2)
        if action["type"] == "grasp" and distance_to_tip > grasp_tolerance:
            record(
                Collision(
                    event_time,
                    min(index, len(segment_durations) - 1),
                    "grasp_unreachable",
                    f"grasp waypoint P{index + 1} is {distance_to_tip:.2f}m from object {name}, beyond tolerance {grasp_tolerance:.2f}m",
                )
            )
        if action["type"] == "grasp":
            held_names.add(name)
            grasp_events.append({"time": event_time, "waypoint_index": index, "object": name})
        else:
            held_names.discard(name)
            placements[name] = tip
            release_events.append({"time": event_time, "waypoint_index": index, "object": name})

    for index in range(len(segment_durations) - 1):
        junction_time = float(sum(segment_durations[: index + 1]))
        difference = np.abs(velocities[index + 1] - velocities[index])
        joint = int(np.argmax(difference))
        if difference[joint] > junction_speed_tolerance + 1e-9:
            record(
                Collision(
                    junction_time,
                    index + 1,
                    "speed_jump",
                    f"joint {joint + 1} speed jumps {velocities[index, joint]:.2f} -> {velocities[index + 1, joint]:.2f} rad/s at P{index + 2}",
                )
            )

    for time, angles in zip(times, samples):
        segment = int(np.searchsorted(boundaries, time, side="right") - 1)
        segment = min(segment, max(0, len(segment_durations) - 1))

        for joint in np.where((angles < bounds[:, 0]) | (angles > bounds[:, 1]))[0]:
            collisions.append(Collision(float(time), segment, "joint_limit", f"joint {int(joint) + 1} exceeded its limit"))

        vertices = forward_kinematics(angles, link_lengths)
        for obstacle in obstacles:
            ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
            for link_index in range(len(vertices) - 1):
                distance = point_segment_distance(ox, oy, *vertices[link_index], *vertices[link_index + 1])
                if distance < radius + link_radius:
                    name = obstacle.get("name", f"obstacle-{link_index + 1}")
                    record(Collision(float(time), segment, "obstacle", f"link {link_index + 1} collided with obstacle {name}"))

        for left in range(len(vertices) - 1):
            for right in range(left + 2, len(vertices) - 1):
                distance = capsule_distance(vertices[left], vertices[left + 1], vertices[right], vertices[right + 1])
                if distance < link_radius * 2:
                    record(Collision(float(time), segment, "self_collision", f"links {left + 1} and {right + 1} collided"))

        held_now: dict[str, CarriedObject] = {}
        for name in objects:
            grasp_time = next((event["time"] for event in grasp_events if event["object"] == name), None)
            release_time = next((event["time"] for event in release_events if event["object"] == name), None)
            if grasp_time is not None and grasp_time <= time + 1e-9 and (
                release_time is None or release_time > time + 1e-9
            ):
                held_now[name] = objects[name]
        frame_objects: list[dict] = []
        for name, obj in objects.items():
            if name in held_now:
                center = obj.held_position(vertices[-2], vertices[-1])
                state = "held"
            else:
                center = placements[name]
                state = "resting"
            frame_objects.append({"name": name, "x": center[0], "y": center[1], "radius": obj.radius, "state": state})

            for obstacle in obstacles:
                ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
                gap = sqrt((center[0] - ox) ** 2 + (center[1] - oy) ** 2)
                if gap < radius + obj.radius:
                    obstacle_name = obstacle.get("name", "obstacle")
                    if state == "held":
                        record(
                            Collision(
                                float(time),
                                segment,
                                "carried_obstacle",
                                f"carried object {name} collided with obstacle {obstacle_name} during transport",
                            )
                        )
                    else:
                        record(
                            Collision(
                                float(time),
                                segment,
                                "object_obstacle",
                                f"resting object {name} overlaps obstacle {obstacle_name}",
                            )
                        )

        for name, obj in held_now.items():
            center = next(item for item in frame_objects if item["name"] == name)
            center_xy = (center["x"], center["y"])
            for link_index in range(len(vertices) - 2):
                gap = point_segment_distance(
                    center_xy[0], center_xy[1], *vertices[link_index], *vertices[link_index + 1]
                )
                if gap < obj.radius + link_radius:
                    record(
                        Collision(
                            float(time),
                            segment,
                            "carried_self",
                            f"carried object {name} collided with link {link_index + 1}",
                        )
                    )

        frames.append(
            {
                "time": float(time),
                "joint_angles": [float(value) for value in angles],
                "links": [[float(x), float(y)] for x, y in vertices],
                "objects": frame_objects,
            }
        )

    first = collisions[0] if collisions else None
    valid = not collisions
    return {
        "valid": valid,
        "duration": float(times[-1]),
        "segment_durations": [float(value) for value in segment_durations],
        "segment_velocities": [[float(v) for v in row] for row in velocities],
        "segments_solved": [bool(ok) for ok in segment_ok],
        "grasp_events": grasp_events,
        "release_events": release_events,
        "constraint_version": CONSTRAINT_VERSION,
        "frames": frames,
        "collisions": [collision.__dict__ for collision in collisions[:20]],
        "first_collision": None if first is None else first.__dict__,
    }
