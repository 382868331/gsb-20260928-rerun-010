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
    end_time: float | None = None
    confidence: str = "interval"


@dataclass(frozen=True)
class HazardInterval:
    start_time: float
    end_time: float
    segment: int
    kind: str
    message: str
    confidence: str


COLLISION_CHECK_DT = 0.005


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
    dof = len(waypoints[0])
    points = [_as_vector(values, "waypoint", dof) for values in waypoints]
    if not points:
        raise ValueError("at least one waypoint is required")
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
        count = max(1, int(np.floor(duration / sample_dt + 1e-9)))
        for index in range(1, count + 1):
            local_time = index * sample_dt
            current_time = elapsed + local_time
            if abs(current_time - times[-1]) <= 1e-9:
                continue
            alpha = local_time / duration
            times.append(current_time)
            samples.append(start + alpha * (end - start))
        if times[-1] < elapsed + duration - 1e-9:
            times.append(elapsed + duration)
            samples.append(end.copy())
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


def swept_segment_envelope_distance(start_a, end_a, start_b, end_b):
    return min(
        point_segment_distance(*start_a, *start_b, *end_b),
        point_segment_distance(*end_a, *start_b, *end_b),
        point_segment_distance(*start_b, *start_a, *end_a),
        point_segment_distance(*end_b, *start_a, *end_a),
    )


def carried_object_pose(vertices, angles, carried_object):
    end_x, end_y = vertices[-1]
    end_orientation = float(np.sum(angles))
    offset_x = float(carried_object.get("offset_x", carried_object.get("offset", [0.0, 0.0])[0]))
    offset_y = float(carried_object.get("offset_y", carried_object.get("offset", [0.0, 0.0])[1]))
    rotated_x = offset_x * cos(end_orientation) - offset_y * sin(end_orientation)
    rotated_y = offset_x * sin(end_orientation) + offset_y * cos(end_orientation)
    return end_x + rotated_x, end_y + rotated_y


def active_object_for_segment(picks_by_waypoint, places_by_waypoint, segment_index, initial=None):
    active = set()
    if initial is not None:
        active.add(initial)
    for index in range(segment_index + 1):
        active.update(picks_by_waypoint.get(index, []))
        active.difference_update(places_by_waypoint.get(index + 1, []))
    return next(iter(active), None) if active else None


def _end_effector_velocity(joint_angles: np.ndarray, joint_velocity: np.ndarray, link_lengths: list[float]) -> dict:
    x = y = angle = 0.0
    joint_positions = [(0.0, 0.0)]
    for joint_angle, length in zip(joint_angles, link_lengths):
        angle += float(joint_angle)
        x += length * cos(angle)
        y += length * sin(angle)
        joint_positions.append((x, y))

    linear_x = linear_y = 0.0
    for joint_index, joint_position in enumerate(joint_positions[:-1]):
        end_x, end_y = joint_positions[-1]
        dx = end_x - joint_position[0]
        dy = end_y - joint_position[1]
        speed = float(joint_velocity[joint_index])
        linear_x += -dy * speed
        linear_y += dx * speed
    angular_velocity = float(np.sum(joint_velocity))
    return {
        "linear_velocity": [linear_x, linear_y],
        "angular_velocity": angular_velocity,
    }


def _boundary_dynamic_state(
    points: list[np.ndarray],
    velocities: list[np.ndarray],
    durations: list[float],
    link_lengths: list[float],
    carried_object_id: str | None,
    side: str,
    wait_duration: float = 0.0,
) -> dict:
    nonzero = [index for index, duration in enumerate(durations) if duration > 0.0]
    if not nonzero:
        motion_kind = "unconstrained_zero_duration"
        joint_velocity = [0.0 for _ in link_lengths]
    else:
        edge_index = nonzero[0] if side == "start" else nonzero[-1]
        motion_kind = "motion"
        joint_velocity = [float(value) for value in velocities[edge_index]]
    angles = points[0] if side == "start" else points[-1]
    is_turn_in_place = motion_kind == "motion" and np.allclose(points[0], points[-1], atol=1e-9)
    endpoint_motion = _end_effector_velocity(angles, np.asarray(joint_velocity, dtype=float), link_lengths)
    return {
        "motion_kind": "wait" if wait_duration > 0 else ("turn_in_place" if is_turn_in_place else motion_kind),
        "joint_angles": [float(value) for value in angles],
        "joint_velocities": joint_velocity,
        "translation_velocity": endpoint_motion["linear_velocity"],
        "rotation_velocity": endpoint_motion["angular_velocity"],
        "carried_object_id": None if carried_object_id is None else str(carried_object_id),
        "available_buffer_time": float(wait_duration),
    }


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
    reviewed_segments: list[int] | None = None,
    previous_validation_digest: str | None = None,
    initial_carried_object_id: str | None = None,
    wait_duration: float = 0.0,
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
    if wait_duration < 0:
        raise ValueError("wait_duration must be non-negative")
    if np.any(acceleration <= 0):
        raise ValueError("acceleration limits must be positive")

    objects_by_id = {str(item["id"]): item for item in (carried_objects or [])}
    if len(objects_by_id) != len(carried_objects or []):
        raise ValueError("carried object ids must be unique")
    if initial_carried_object_id is not None and str(initial_carried_object_id) not in objects_by_id:
        raise ValueError(f"initial carried object {initial_carried_object_id} is not defined")
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
    if initial_carried_object_id is not None:
        active_objects.add(str(initial_carried_object_id))
    picks_by_waypoint: dict[int, list[str]] = {}
    places_by_waypoint: dict[int, list[str]] = {}
    for event in pick_events:
        picks_by_waypoint.setdefault(int(event["waypoint_index"]), []).append(str(event["object_id"]))
    for event in place_events:
        places_by_waypoint.setdefault(int(event["waypoint_index"]), []).append(str(event["object_id"]))
    for index in range(len(waypoints)):
        current_picks = set(picks_by_waypoint.get(index, []))
        current_places = set(places_by_waypoint.get(index, []))
        if current_picks & active_objects:
            raise ValueError("object cannot be picked while it is already carried")
        if current_places - active_objects:
            raise ValueError("object cannot be placed before it has been picked")
        active_objects.difference_update(current_places)
        active_objects.update(current_picks)
        if len(active_objects) > 1:
            raise ValueError("only one object may be carried at a time")
    final_carried_object_id = next(iter(active_objects), None) if active_objects else None

    display_times, display_samples, segment_durations = interpolate_trajectory(
        waypoints, speed_limits, sample_dt, segment_speed_limits
    )
    check_times, check_samples, _ = interpolate_trajectory(
        waypoints, speed_limits, COLLISION_CHECK_DT, segment_speed_limits
    )
    points = [np.asarray(values, dtype=float) for values in waypoints]
    if len(points) < 2 and wait_duration <= 0:
        raise ValueError("at least two waypoints are required")
    velocities = segment_velocities(points, segment_durations)
    moving_duration = float(sum(segment_durations))
    stationary_action = moving_duration == 0.0
    if wait_duration > 0.0 and not stationary_action:
        raise ValueError("wait_duration is only supported for a stationary wait operation")
    duration = moving_duration + float(wait_duration)
    if wait_duration > 0.0:
        display_times.append(duration)
        display_samples.append(points[-1].copy())
        check_times.append(duration)
        check_samples.append(points[-1].copy())
    boundaries = np.cumsum([0.0, *segment_durations[:-1]])
    hazards: list[HazardInterval] = []

    for segment_index in range(len(velocities) - 1):
        velocity_change = np.abs(velocities[segment_index + 1] - velocities[segment_index])
        allowed_change = acceleration * velocity_transition_time
        for joint in np.where(velocity_change > allowed_change + 1e-9)[0]:
            hazards.append(
                HazardInterval(
                    float(boundaries[segment_index + 1]),
                    float(boundaries[segment_index + 1]),
                    segment_index + 1,
                    "velocity_transition",
                    f"joint {int(joint) + 1} changes speed too quickly between segments {segment_index + 1} and {segment_index + 2}",
                    "boundary",
                )
            )

    planned_events: list[dict] = []
    elapsed = 0.0
    for index in range(len(points)):
        event_time = elapsed + (wait_duration if index == len(points) - 1 and stationary_action else 0.0)
        for object_id in places_by_waypoint.get(index, []):
            planned_events.append({"type": "place", "waypoint_index": index, "object_id": object_id, "time": event_time})
        for object_id in picks_by_waypoint.get(index, []):
            planned_events.append({"type": "pick", "waypoint_index": index, "object_id": object_id, "time": event_time})
        if index < len(segment_durations):
            elapsed += segment_durations[index]

    def add_hazard(start_time: float, end_time: float, segment: int, kind: str, message: str, confidence: str = "interval") -> None:
        hazards.append(HazardInterval(start_time, end_time, segment, kind, message, confidence))

    for interval_index in range(len(check_samples) - 1):
        start_time = float(check_times[interval_index])
        end_time = float(check_times[interval_index + 1])
        start_angles = check_samples[interval_index]
        end_angles = check_samples[interval_index + 1]
        segment = int(np.searchsorted(boundaries, start_time, side="right") - 1)
        segment = min(segment, max(0, len(segment_durations) - 1))
        envelope_vertices = [
            forward_kinematics(start_angles + (step / 32) * (end_angles - start_angles), link_lengths)
            for step in range(33)
        ]
        start_vertices = envelope_vertices[0]
        end_vertices = envelope_vertices[-1]
        active_object_id = active_object_for_segment(picks_by_waypoint, {}, segment, initial_carried_object_id)

        for joint in range(dof):
            low = min(float(start_angles[joint]), float(end_angles[joint]))
            high = max(float(start_angles[joint]), float(end_angles[joint]))
            if high < float(bounds[joint, 0]) or low > float(bounds[joint, 1]):
                add_hazard(start_time, end_time, segment, "joint_limit", f"joint {joint + 1} exceeded its limit")
                break
        if hazards and hazards[-1].start_time == start_time:
            continue

        hit = False
        for obstacle in obstacles:
            ox, oy, obstacle_radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
            name = obstacle.get("name", "obstacle")
            for link_index in range(len(start_vertices) - 1):
                distance = min(
                    min(
                        point_segment_distance(ox, oy, *vertices[link_index], *vertices[link_index + 1])
                        for vertices in envelope_vertices
                    ),
                    min(
                        swept_segment_envelope_distance(
                            first[link_index + 1], second[link_index + 1], (ox, oy), (ox, oy)
                        )
                        for first, second in zip(envelope_vertices, envelope_vertices[1:])
                    ),
                    min(
                        swept_segment_envelope_distance(
                            first[link_index], second[link_index], (ox, oy), (ox, oy)
                        )
                        for first, second in zip(envelope_vertices, envelope_vertices[1:])
                    ),
                )
                if distance < obstacle_radius + link_radius:
                    add_hazard(start_time, end_time, segment, "obstacle", f"link {link_index + 1} swept through obstacle {name}")
                    hit = True
                    break
            if hit:
                break
            gripper_distance = min(
                swept_segment_envelope_distance(first[-1], second[-1], (ox, oy), (ox, oy))
                for first, second in zip(envelope_vertices, envelope_vertices[1:])
            )
            if gripper_distance < obstacle_radius + gripper_radius:
                add_hazard(start_time, end_time, segment, "gripper_obstacle", f"gripper swept through obstacle {name}")
                hit = True
                break
        if hit:
            continue

        if active_object_id is not None:
            active_object = objects_by_id[active_object_id]
            envelope_angles = [start_angles + (step / 32) * (end_angles - start_angles) for step in range(33)]
            envelope_poses = [
                carried_object_pose(vertices, angles, active_object)
                for vertices, angles in zip(envelope_vertices, envelope_angles)
            ]
            object_radius = float(active_object["radius"])
            for obstacle in obstacles:
                ox, oy, obstacle_radius = float(obstacle["x"]), float(obstacle["y"]), float(obstacle["radius"])
                distance = min(
                    swept_segment_envelope_distance(first, second, (ox, oy), (ox, oy))
                    for first, second in zip(envelope_poses, envelope_poses[1:])
                )
                if distance < obstacle_radius + object_radius:
                    name = obstacle.get("name", "obstacle")
                    add_hazard(start_time, end_time, segment, "carried_object_obstacle", f"carried object {active_object_id} swept through obstacle {name}")
                    hit = True
                    break
        if hit:
            continue

        for left in range(len(start_vertices) - 1):
            for right in range(left + 2, len(start_vertices) - 1):
                distance = min(
                    capsule_distance(vertices[left], vertices[left + 1], vertices[right], vertices[right + 1])
                    for vertices in envelope_vertices
                )
                for first, second in zip(envelope_vertices, envelope_vertices[1:]):
                    for left_point in (left, left + 1):
                        for right_point in (right, right + 1):
                            distance = min(
                                distance,
                                swept_segment_envelope_distance(
                                    first[left_point], second[left_point], first[right_point], second[right_point]
                                ),
                            )
                if distance < link_radius * 2:
                    add_hazard(start_time, end_time, segment, "self_collision", f"links {left + 1} and {right + 1} collided")
                    hit = True
                    break
            if hit:
                break

    hazards.sort(key=lambda item: (item.start_time, item.end_time, item.kind))

    frames: list[dict] = []
    for time, angles in zip(display_times, display_samples):
        segment = int(np.searchsorted(boundaries, time, side="right") - 1)
        segment = min(segment, max(0, len(segment_durations) - 1))
        vertices = forward_kinematics(angles, link_lengths)
        active_object_id = active_object_for_segment(
            picks_by_waypoint, places_by_waypoint, segment, initial_carried_object_id
        )
        object_position = None
        if active_object_id is not None:
            object_position = carried_object_pose(vertices, angles, objects_by_id[active_object_id])
        frames.append(
            {
                "time": float(time),
                "joint_angles": [float(value) for value in angles],
                "links": [[float(x), float(y)] for x, y in vertices],
                "carried_object_id": active_object_id,
                "object_position": None if object_position is None else [float(object_position[0]), float(object_position[1])],
            }
        )

    first_hazard = hazards[0] if hazards else None
    first = None if first_hazard is None else Collision(
        first_hazard.start_time,
        first_hazard.segment,
        first_hazard.kind,
        first_hazard.message,
        first_hazard.end_time,
        first_hazard.confidence,
    )
    reviewed = sorted({int(value) for value in (reviewed_segments or []) if 0 <= int(value) < len(segment_durations)})
    affected_segments = sorted({item.segment for item in hazards})
    return {
        "valid": not hazards,
        "duration": duration,
        "motion_duration": moving_duration,
        "wait_duration": float(wait_duration),
        "start_state": {
            "joint_angles": [float(value) for value in points[0]],
            "carried_object_id": None if initial_carried_object_id is None else str(initial_carried_object_id),
        },
        "end_state": {
            "joint_angles": [float(value) for value in points[-1]],
            "carried_object_id": final_carried_object_id,
        },
        "dynamic_state": {
            "at_start": _boundary_dynamic_state(
                points, velocities, segment_durations, link_lengths,
                initial_carried_object_id, "start", wait_duration,
            ),
            "at_end": _boundary_dynamic_state(
                points, velocities, segment_durations, link_lengths,
                final_carried_object_id, "end", wait_duration,
            ),
        },
        "segment_velocities": [[float(value) for value in velocity] for velocity in velocities],
        "segment_durations": [float(value) for value in segment_durations],
        "frames": frames,
        "planned_events": planned_events,
        "constraint_version": constraint_version,
        "collision_check_dt": COLLISION_CHECK_DT,
        "display_sample_dt": float(sample_dt),
        "collisions": [
            Collision(item.start_time, item.segment, item.kind, item.message, item.end_time, item.confidence).__dict__
            for item in hazards[:20]
        ],
        "hazard_intervals": [item.__dict__ for item in hazards],
        "replay_window": None if first_hazard is None else {
            "start_time": max(0.0, first_hazard.start_time - max(COLLISION_CHECK_DT, sample_dt)),
            "end_time": min(duration, first_hazard.end_time + max(COLLISION_CHECK_DT, sample_dt)),
            "segment": first_hazard.segment,
        },
        "review": {
            "previous_validation_digest": previous_validation_digest,
            "reviewed_segments": reviewed,
            "affected_segments": affected_segments,
            "stale_segments": [segment for segment in reviewed if segment in affected_segments],
            "fully_rechecked": not reviewed or not affected_segments,
        },
        "first_collision": None if first is None else first.__dict__,
    }
