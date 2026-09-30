from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin, sqrt
import numpy as np


@dataclass(frozen=True)
class Collision:
    time: float
    segment: int
    kind: str
    message: str


@dataclass
class DangerInterval:
    segment: int
    start_time: float
    end_time: float
    kinds: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "segment": self.segment,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "kinds": list(self.kinds),
            "messages": list(self.messages),
        }


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


def pose_at(segment: int, alpha: float, points: list[np.ndarray]) -> np.ndarray:
    return points[segment] + alpha * (points[segment + 1] - points[segment])


def vertex_speed_bounds(
    angles_a: np.ndarray,
    angles_b: np.ndarray,
    link_lengths: list[float],
) -> np.ndarray:
    """Maximum speed of every link vertex over a synchronized linear segment.

    Joint angles move linearly, so each joint speed is constant. Vertex k is
    affected only by joints 1..k; rotating one such joint moves the vertex
    perpendicularly at speed |omega| * distance(joint, vertex). The sum of
    these contributions is a rigorous upper bound on the vertex speed at any
    point in the interval and therefore on path length per unit time.
    """

    omega = (angles_b - angles_a)
    vertex_count = len(link_lengths) + 1
    speeds = np.zeros(vertex_count)
    for joint_index, joint_speed in enumerate(omega):
        downstream_length = float(sum(link_lengths[joint_index:]))
        speeds[joint_index + 1 :] += abs(float(joint_speed)) * downstream_length
    return speeds


def carried_object_pose(
    vertices: list[tuple[float, float]], angles: np.ndarray, active_object: dict | None
) -> tuple[tuple[float, float], float] | None:
    if active_object is None:
        return None
    end_x, end_y = vertices[-1]
    orientation = float(np.sum(angles))
    offset = active_object.get("offset")
    if offset is not None:
        offset_x, offset_y = float(offset[0]), float(offset[1])
    else:
        offset_x = float(active_object.get("offset_x", 0.0))
        offset_y = float(active_object.get("offset_y", 0.0))
    rotated_x = offset_x * cos(orientation) - offset_y * sin(orientation)
    rotated_y = offset_x * sin(orientation) + offset_y * cos(orientation)
    return (end_x + rotated_x, end_y + rotated_y), float(active_object["radius"])


def point_collisions(
    angles: np.ndarray,
    link_lengths: list[float],
    obstacles: list[dict[str, float]],
    bounds: np.ndarray,
    link_radius: float,
    gripper_radius: float,
    active_object: dict | None,
) -> list[tuple[str, str]]:
    """Exact collision check at a single pose; returns (kind, message) pairs."""

    findings: list[tuple[str, str]] = []
    for joint in np.where((angles < bounds[:, 0]) | (angles > bounds[:, 1]))[0]:
        findings.append(("joint_limit", f"joint {int(joint) + 1} exceeded its limit"))

    vertices = forward_kinematics(angles, link_lengths)
    end_x, end_y = vertices[-1]
    for obstacle in obstacles:
        ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
        name = obstacle.get("name", "obstacle")
        for link_index in range(len(vertices) - 1):
            distance = point_segment_distance(ox, oy, *vertices[link_index], *vertices[link_index + 1])
            if distance < radius + link_radius:
                findings.append(("obstacle", f"link {link_index + 1} collided with obstacle {name}"))
        end_distance = sqrt((ox - end_x) ** 2 + (oy - end_y) ** 2)
        if end_distance < radius + gripper_radius:
            findings.append(("gripper_obstacle", f"gripper collided with obstacle {name}"))

    carried = carried_object_pose(vertices, angles, active_object)
    if carried is not None:
        (object_x, object_y), object_radius = carried
        object_id = str(active_object["id"])
        for obstacle in obstacles:
            ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
            distance = sqrt((ox - object_x) ** 2 + (oy - object_y) ** 2)
            if distance < radius + object_radius:
                name = obstacle.get("name", "obstacle")
                findings.append(
                    ("carried_object_obstacle", f"carried object {object_id} collided with obstacle {name}")
                )

    for left in range(len(vertices) - 1):
        for right in range(left + 2, len(vertices) - 1):
            distance = capsule_distance(vertices[left], vertices[left + 1], vertices[right], vertices[right + 1])
            if distance < link_radius * 2:
                findings.append(("self_collision", f"links {left + 1} and {right + 1} collided"))
    return findings


def interval_maybe_collides(
    angles_a: np.ndarray,
    angles_b: np.ndarray,
    link_lengths: list[float],
    obstacles: list[dict[str, float]],
    bounds: np.ndarray,
    link_radius: float,
    gripper_radius: float,
    active_object: dict | None,
    interval_time: float,
) -> bool:
    """Conservative check covering the whole swept volume between two poses.

    Every geometry is inflated by a rigorous upper bound on its motion across
    the interval, so ``False`` certifies the entire interval as collision free
    regardless of the time step used for display sampling.
    """

    vertices_a = forward_kinematics(angles_a, link_lengths)
    vertices_b = forward_kinematics(angles_b, link_lengths)
    motion = vertex_speed_bounds(angles_a, angles_b, link_lengths) * interval_time
    if np.any((angles_a < bounds[:, 0]) | (angles_a > bounds[:, 1])) or np.any(
        (angles_b < bounds[:, 0]) | (angles_b > bounds[:, 1])
    ):
        return True
    for obstacle in obstacles:
        ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
        for link_index in range(len(link_lengths) - 1):
            inflation = max(motion[link_index], motion[link_index + 1]) + link_radius
            if point_segment_distance(ox, oy, *vertices_a[link_index], *vertices_a[link_index + 1]) < radius + inflation:
                return True
        end_slack = motion[-1]
        end_x, end_y = vertices_a[-1]
        if sqrt((ox - end_x) ** 2 + (oy - end_y) ** 2) < radius + gripper_radius + end_slack:
            return True

    carried_a = carried_object_pose(vertices_a, angles_a, active_object)
    if carried_a is not None:
        (ax, ay), object_radius = carried_a
        offset = active_object.get("offset")
        if offset is not None:
            offset_radius = abs(float(offset[0])) + abs(float(offset[1]))
        else:
            offset_radius = abs(float(active_object.get("offset_x", 0.0))) + abs(
                float(active_object.get("offset_y", 0.0))
            )
        # gripper endpoint motion plus rotation of the object about the endpoint
        center_motion = end_slack + float(np.sum(np.abs(angles_b - angles_a))) * offset_radius
        for obstacle in obstacles:
            ox, oy, radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
            if sqrt((ox - ax) ** 2 + (oy - ay) ** 2) < radius + object_radius + center_motion:
                return True

    for left in range(len(link_lengths) - 1):
        for right in range(left + 2, len(link_lengths) - 1):
            inflation = max(motion[left], motion[left + 1], motion[right], motion[right + 1]) + link_radius * 2
            if capsule_distance(
                vertices_a[left], vertices_a[left + 1], vertices_a[right], vertices_a[right + 1]
            ) < inflation:
                return True
    return False


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
    safety_resolution: float = 0.005,
    recheck_segments: list[int] | None = None,
) -> dict:
    if safety_resolution <= 0:
        raise ValueError("safety_resolution must be positive")
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
    transition_collisions: list[Collision] = []

    for segment_index in range(len(velocities) - 1):
        velocity_change = np.abs(velocities[segment_index + 1] - velocities[segment_index])
        allowed_change = acceleration * velocity_transition_time
        for joint in np.where(velocity_change > allowed_change + 1e-9)[0]:
            transition_collisions.append(
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

    carried_at_waypoint: list[str | None] = []
    carried: set[str] = set()
    for index in range(len(points)):
        carried.difference_update(places_by_waypoint.get(index, []))
        carried.update(picks_by_waypoint.get(index, []))
        carried_at_waypoint.append(next(iter(carried), None))

    total_segments = len(segment_durations)
    if recheck_segments is None:
        checked_segments = list(range(total_segments))
    else:
        checked_segments = sorted({int(index) for index in recheck_segments})
        if any(not 0 <= index < total_segments for index in checked_segments):
                raise ValueError("recheck_segments must reference existing trajectory segments")

    frames: list[dict] = []
    danger_intervals: list[DangerInterval] = []

    def certify_segment(segment_index: int) -> None:
        """Certify one interpolation segment with recursive swept checks.

        interval_maybe_collides inflates geometry by a rigorous motion bound,
        so a negative result covers the whole swept interval. Ambiguous
        intervals are subdivided to the safety resolution; anything still
        ambiguous is reported as danger rather than assumed safe.
        """

        duration = segment_durations[segment_index]
        base_time = float(boundaries[segment_index])
        object_id = carried_at_waypoint[segment_index]
        active_object = objects_by_id[object_id] if object_id is not None else None

        def record_interval(start_alpha: float, end_alpha: float, findings: list[tuple[str, str]]) -> None:
            danger_intervals.append(
                DangerInterval(
                    segment_index,
                    base_time + start_alpha * duration,
                    base_time + end_alpha * duration,
                    [kind for kind, _ in findings],
                    [message for _, message in findings],
                )
            )

        def probe(alpha_a: float, alpha_b: float, depth: int) -> None:
            pose_a = pose_at(segment_index, alpha_a, points)
            pose_b = pose_at(segment_index, alpha_b, points)
            if not interval_maybe_collides(
                pose_a, pose_b, link_lengths, obstacles, bounds,
                link_radius, gripper_radius, active_object, (alpha_b - alpha_a) * duration,
            ):
                return
            if (alpha_b - alpha_a) * duration <= safety_resolution or depth >= 64:
                findings: list[tuple[str, str]] = []
                for alpha in (alpha_a, 0.5 * (alpha_a + alpha_b), alpha_b):
                    for kind, message in point_collisions(
                        pose_at(segment_index, alpha, points), link_lengths, obstacles, bounds,
                        link_radius, gripper_radius, active_object,
                    ):
                        if (kind, message) not in findings:
                            findings.append((kind, message))
                if not findings:
                    findings = [
                        ("swept_uncertain", "motion could not be certified collision free at the safety resolution")
                    ]
                record_interval(alpha_a, alpha_b, findings)
                return
            midpoint = 0.5 * (alpha_a + alpha_b)
            probe(alpha_a, midpoint, depth + 1)
            probe(midpoint, alpha_b, depth + 1)

        if duration > 0.0:
            probe(0.0, 1.0, 0)
        else:
            findings = point_collisions(
                points[segment_index], link_lengths, obstacles, bounds,
                link_radius, gripper_radius, active_object,
            )
            if findings:
                record_interval(0.0, 0.0, findings)

    for segment_index in checked_segments:
        certify_segment(segment_index)

    waypoint_collisions: list[Collision] = []
    for waypoint_index, object_id in enumerate(carried_at_waypoint):
        active_object = objects_by_id[object_id] if object_id is not None else None
        time = float(boundaries[waypoint_index]) if waypoint_index < len(boundaries) else float(times[-1])
        for kind, message in point_collisions(
            points[waypoint_index], link_lengths, obstacles, bounds,
            link_radius, gripper_radius, active_object,
        ):
            waypoint_collisions.append(
                Collision(time, min(waypoint_index, max(0, total_segments - 1)), kind, message)
            )

    for time, angles in zip(times, samples):
        segment = int(np.searchsorted(boundaries, time, side="right") - 1)
        segment = min(segment, max(0, total_segments - 1))
        vertices = forward_kinematics(angles, link_lengths)
        active_object_id = carried_at_waypoint[segment]
        carried_pose = carried_object_pose(
            vertices, angles, objects_by_id[active_object_id] if active_object_id else None
        )
        object_position = None if carried_pose is None else [float(carried_pose[0][0]), float(carried_pose[0][1])]
        frames.append(
            {
                "time": float(time),
                "joint_angles": [float(value) for value in angles],
                "links": [[float(x), float(y)] for x, y in vertices],
                "carried_object_id": active_object_id,
                "object_position": object_position,
            }
        )

    merged: list[DangerInterval] = []
    for interval in sorted(danger_intervals, key=lambda item: (item.start_time, item.end_time)):
        if (
            merged
            and interval.segment == merged[-1].segment
            and interval.start_time <= merged[-1].end_time + safety_resolution * 0.5
        ):
            merged[-1].end_time = max(merged[-1].end_time, interval.end_time)
            for kind in interval.kinds:
                if kind not in merged[-1].kinds:
                    merged[-1].kinds.append(kind)
            for message in interval.messages:
                if message not in merged[-1].messages:
                    merged[-1].messages.append(message)
        else:
            merged.append(interval)

    transition_intervals = [
        DangerInterval(
            collision.segment,
            collision.time,
            collision.time,
            [collision.kind],
            [collision.message],
        )
        for collision in transition_collisions
    ]
    merged = transition_intervals + merged
    merged.sort(key=lambda item: (item.start_time, item.end_time))

    collision_events: list[Collision] = list(waypoint_collisions)
    for interval in merged:
        kinds = [kind for kind in interval.kinds if kind != "swept_uncertain"] or interval.kinds
        messages = [
            message for kind, message in zip(interval.kinds, interval.messages) if kind != "swept_uncertain"
        ] or interval.messages
        kind = kinds[0] if kinds else "swept_uncertain"
        message = messages[0] if messages else "uncertified motion interval"
        collision_events.append(Collision(interval.start_time, interval.segment, kind, message))
    collision_events.sort(key=lambda collision: (collision.time, collision.segment, collision.kind))

    first = collision_events[0] if collision_events else None
    replay_window = None
    if merged:
        first_interval = next(
            (item for item in merged if first is not None and item.start_time == first.time), merged[0]
        )
        replay_window = {
            "start_time": first_interval.start_time,
            "end_time": first_interval.end_time,
            "segment": first_interval.segment,
            "kinds": list(first_interval.kinds),
        }
    elif first is not None:
        replay_window = {
            "start_time": first.time,
            "end_time": first.time,
            "segment": first.segment,
            "kinds": [first.kind],
        }

    return {
        "valid": not collision_events,
        "duration": float(times[-1]),
        "segment_durations": [float(value) for value in segment_durations],
        "frames": frames,
        "planned_events": planned_events,
        "constraint_version": constraint_version,
        "safety_resolution": safety_resolution,
        "display_sample_dt": sample_dt,
        "checked_segments": checked_segments,
        "collisions": [collision.__dict__ for collision in collision_events[:20]],
        "first_collision": None if first is None else first.__dict__,
        "danger_intervals": [interval.as_dict() for interval in merged],
        "replay_window": replay_window,
        "certified_safe": not collision_events,
    }
