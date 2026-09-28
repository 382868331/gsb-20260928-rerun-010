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


def interpolate_trajectory(
    waypoints: list[list[float]],
    speed_limits: list[float],
    sample_dt: float = 0.02,
) -> tuple[list[float], list[np.ndarray], list[float]]:
    if sample_dt <= 0:
        raise ValueError("sample_dt must be positive")
    if len(waypoints) < 2:
        raise ValueError("at least two waypoints are required")

    dof = len(waypoints[0])
    points = [_as_vector(values, "waypoint", dof) for values in waypoints]
    limits = _as_vector(speed_limits, "speed_limits", dof)
    durations = [
        segment_duration(points[index + 1] - points[index], limits)
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
) -> dict:
    dof = len(link_lengths)
    limits = _as_vector(speed_limits, "speed_limits", dof)
    bounds = np.asarray(joint_limits, dtype=float)
    if bounds.shape != (dof, 2) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("joint_limits must be [minimum, maximum] pairs")
    if link_radius <= 0:
        raise ValueError("link_radius must be positive")

    times, samples, segment_durations = interpolate_trajectory(waypoints, speed_limits, sample_dt)
    boundaries = np.cumsum([0.0, *segment_durations[:-1]])
    collisions: list[Collision] = []
    frames: list[dict] = []

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
                    collisions.append(Collision(float(time), segment, "obstacle", f"link {link_index + 1} collided with obstacle {name}"))

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
            }
        )

    first = collisions[0] if collisions else None
    return {
        "valid": not collisions,
        "duration": float(times[-1]),
        "segment_durations": [float(value) for value in segment_durations],
        "frames": frames,
        "collisions": [collision.__dict__ for collision in collisions[:20]],
        "first_collision": None if first is None else first.__dict__,
    }
