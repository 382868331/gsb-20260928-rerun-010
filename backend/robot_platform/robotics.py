from __future__ import annotations

import copy
import hashlib
import json
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


def active_object_for_segment(picks_by_waypoint, places_by_waypoint, segment_index):
    active = set()
    for index in range(segment_index + 1):
        active.update(picks_by_waypoint.get(index, []))
        active.difference_update(places_by_waypoint.get(index + 1, []))
    return next(iter(active), None) if active else None


def _carried_ranges(picks_by_waypoint, places_by_waypoint, durations):
    boundaries = [0.0]
    for duration in durations:
        boundaries.append(boundaries[-1] + duration)
    active = set()
    started = {}
    ranges = []
    for index, current_time in enumerate(boundaries):
        for object_id in places_by_waypoint.get(index, []):
            active.discard(object_id)
            ranges.append({
                "object_id": object_id,
                "start_time": started.pop(object_id, current_time),
                "end_time": current_time,
                "confidence": "confirmed",
            })
        for object_id in picks_by_waypoint.get(index, []):
            active.add(object_id)
            started[object_id] = current_time
    for object_id in sorted(active):
        ranges.append({
            "object_id": object_id,
            "start_time": started.get(object_id, boundaries[-1]),
            "end_time": boundaries[-1],
            "confidence": "confirmed",
        })
    return sorted(ranges, key=lambda item: (item["start_time"], item["object_id"]))


def _stable_digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_motion_core(
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
    uncertain_from_segment: int | None = None,
    uncertain_object_id: str | None = None,
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

    display_times, display_samples, segment_durations = interpolate_trajectory(
        waypoints, speed_limits, sample_dt, segment_speed_limits
    )
    check_times, check_samples, _ = interpolate_trajectory(
        waypoints, speed_limits, COLLISION_CHECK_DT, segment_speed_limits
    )
    points = [np.asarray(values, dtype=float) for values in waypoints]
    velocities = segment_velocities(points, segment_durations)
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
        for object_id in places_by_waypoint.get(index, []):
            planned_events.append({"type": "place", "waypoint_index": index, "object_id": object_id, "time": elapsed})
        for object_id in picks_by_waypoint.get(index, []):
            planned_events.append({"type": "pick", "waypoint_index": index, "object_id": object_id, "time": elapsed})
        if index < len(segment_durations):
            elapsed += segment_durations[index]

    def add_hazard(start_time: float, end_time: float, segment: int, kind: str, message: str, confidence: str = "interval") -> None:
        hazards.append(HazardInterval(start_time, end_time, segment, kind, message, confidence))

    if uncertain_from_segment is not None:
        uncertain_from_segment = max(0, min(uncertain_from_segment, len(segment_durations) - 1))
        for segment in range(uncertain_from_segment, len(segment_durations)):
            add_hazard(
                float(boundaries[segment]),
                float(boundaries[segment] + segment_durations[segment]),
                segment,
                "carry_state_unknown",
                f"carried state for {uncertain_object_id or 'the manipulated object'} cannot be confirmed",
                "unknown",
            )

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
        active_object_id = active_object_for_segment(picks_by_waypoint, {}, segment)

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
        active_object_id = active_object_for_segment(picks_by_waypoint, places_by_waypoint, segment)
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
                "gripper_state": "closed" if active_object_id is not None else "open",
                "carry_confidence": "unknown" if uncertain_from_segment is not None and segment >= uncertain_from_segment else "confirmed",
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
    result = {
        "valid": not hazards,
        "duration": float(display_times[-1]),
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
            "end_time": min(float(display_times[-1]), first_hazard.end_time + max(COLLISION_CHECK_DT, sample_dt)),
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
        "carried_ranges": _carried_ranges(picks_by_waypoint, places_by_waypoint, segment_durations),
        "unknown_ranges": [] if uncertain_from_segment is None else [{
            "start_time": float(boundaries[uncertain_from_segment]),
            "end_time": float(display_times[-1]),
            "object_id": uncertain_object_id,
            "reason": "carry_state_unconfirmed",
        }],
    }
    return result


def validate_motion(
    *args,
    task_operations: list[dict] | None = None,
    handoff_conditions: list[dict] | None = None,
    task_failure_events: list[dict] | None = None,
    baseline_snapshot: dict | None = None,
    **kwargs,
) -> dict:
    result = _validate_motion_core(*args, **kwargs)
    waypoints = kwargs.get("waypoints") or (args[0] if args else [])
    segment_count = max(0, len(waypoints) - 1)
    boundaries = [0.0]
    for duration in result["segment_durations"]:
        boundaries.append(boundaries[-1] + duration)
    operations = task_operations or []
    handoffs = handoff_conditions or []
    failures = task_failure_events or []

    active_at = {}
    picks_at = {}
    places_at = {}
    for event in kwargs.get("pick_events", []) or []:
        picks_at.setdefault(int(event["waypoint_index"]), set()).add(str(event["object_id"]))
    for event in kwargs.get("place_events", []) or []:
        places_at.setdefault(int(event["waypoint_index"]), set()).add(str(event["object_id"]))
    current = set()
    for index in range(len(waypoints)):
        current.difference_update(places_at.get(index, set()))
        current.update(picks_at.get(index, set()))
        active_at[index] = set(current)

    handoff_issues = []
    for handoff in handoffs:
        waypoint_index = int(handoff["waypoint_index"])
        if not 0 < waypoint_index <= segment_count:
            continue
        expected_object = next(iter(active_at.get(waypoint_index, set())), None)
        expected_gripper = "closed" if expected_object else "open"
        pose = handoff.get("pose")
        unknown = handoff.get("confidence") == "unknown"
        if pose is not None and [float(value) for value in pose] != [float(value) for value in waypoints[waypoint_index]]:
            handoff_issues.append((waypoint_index, "handoff_pose_mismatch", "handoff pose does not match the actual trajectory waypoint", "unknown" if unknown else "interval"))
        if handoff.get("carried_object_id") != expected_object:
            handoff_issues.append((waypoint_index, "handoff_carry_unknown" if unknown else "handoff_carry_mismatch", "handoff carried object does not match the previous operation endpoint", "unknown" if unknown else "interval"))
        if handoff.get("gripper_state") != expected_gripper:
            handoff_issues.append((waypoint_index, "handoff_gripper_mismatch", "handoff gripper state does not match the planned carried object", "unknown" if unknown else "interval"))

    def add_issues(target, issues):
        target = copy.deepcopy(target)
        for item in target["hazard_intervals"]:
            item.setdefault("time", item["start_time"])
        target["hazard_intervals"].extend([
            {
                "start_time": float(boundaries[segment]),
                "end_time": float(boundaries[segment]),
                "segment": segment,
                "kind": kind,
                "message": message,
                "time": float(boundaries[segment]),
                "confidence": confidence,
            }
            for segment, kind, message, confidence in issues
        ])
        target["hazard_intervals"].sort(key=lambda item: (item["start_time"], item["end_time"], item["kind"]))
        for item in target["hazard_intervals"]:
            item.setdefault("time", item["start_time"])
        target["collisions"] = target["hazard_intervals"][:20]
        target["valid"] = not target["hazard_intervals"]
        target["first_collision"] = target["hazard_intervals"][0] if target["hazard_intervals"] else None
        first_hazard = target["first_collision"]
        target["replay_window"] = None if first_hazard is None else {
            "start_time": max(0.0, first_hazard["start_time"] - max(target["collision_check_dt"], target["display_sample_dt"])),
            "end_time": min(target["duration"], first_hazard["end_time"] + max(target["collision_check_dt"], target["display_sample_dt"])),
            "segment": first_hazard["segment"],
        }
        return target

    result = add_issues(result, handoff_issues)
    segment_provenance = []
    configured_speeds = kwargs.get("segment_speed_limits")
    for segment in range(segment_count):
        operation = next((item for item in operations if int(item["start_waypoint_index"]) <= segment < int(item["end_waypoint_index"])), None)
        handoff = next((item for item in handoffs if int(item["waypoint_index"]) == segment), None)
        source = {
            "segment": segment,
            "operation_id": None if operation is None else operation.get("id"),
            "operation_version": None if operation is None else operation.get("version"),
            "operation_report_digest": None if operation is None else operation.get("report_digest"),
            "incoming_handoff_id": None if handoff is None else handoff.get("id"),
            "handoff_condition": None if handoff is None else {
                key: handoff.get(key)
                for key in ("waypoint_index", "carried_object_id", "gripper_state", "confidence", "pose")
            },
            "configuration_digest": _stable_digest({
                "start_pose": waypoints[segment],
                "end_pose": waypoints[segment + 1],
                "speed_limits": configured_speeds[segment] if configured_speeds else kwargs.get("speed_limits", args[1] if len(args) > 1 else []),
            }),
        }
        source["digest"] = _stable_digest(source)
        segment_provenance.append(source)

    previous_provenance = {}
    if isinstance(baseline_snapshot, dict):
        for item in (baseline_snapshot.get("task_review") or {}).get("segment_provenance", []):
            previous_provenance[int(item["segment"])] = item.get("digest")
    changed_segments = [item["segment"] for item in segment_provenance if previous_provenance.get(item["segment"]) not in (None, item["digest"])]
    recheck_required = set(range(min(changed_segments), segment_count)) if changed_segments else set()
    if operations and not previous_provenance:
        recheck_required.update(range(segment_count))
    result["task_review"] = {
        "operations": copy.deepcopy(operations),
        "handoff_conditions": copy.deepcopy(handoffs),
        "segment_provenance": segment_provenance,
        "changed_segments": sorted(changed_segments),
        "revalidation_required_segments": sorted(recheck_required),
        "baseline_digest": None if baseline_snapshot is None else _stable_digest(baseline_snapshot),
    }

    abnormal_scenarios = []
    for failure in failures:
        waypoint_index = int(failure["waypoint_index"])
        object_id = str(failure["object_id"])
        scenario_kwargs = dict(kwargs)
        scenario_kwargs["pick_events"] = [
            event for event in (kwargs.get("pick_events") or [])
            if not (failure["kind"] == "pick_failure" and int(event["waypoint_index"]) == waypoint_index and str(event["object_id"]) == object_id)
        ]
        kept_places = [
            event for event in (kwargs.get("place_events") or [])
            if not (failure["kind"] == "pick_failure" and str(event["object_id"]) == object_id)
        ]
        scenario_kwargs["place_events"] = [
            event for event in kept_places
            if not (failure["kind"] == "early_release" and int(event["waypoint_index"]) >= waypoint_index and str(event["object_id"]) == object_id)
        ]
        if failure["kind"] == "early_release":
            scenario_kwargs["place_events"].append({"waypoint_index": waypoint_index, "object_id": object_id})
        scenario_kwargs["uncertain_from_segment"] = min(waypoint_index, segment_count - 1) if segment_count else None
        scenario_kwargs["uncertain_object_id"] = object_id
        scenario = _validate_motion_core(*args, **scenario_kwargs)
        scenario["scenario"] = {
            "id": failure.get("id"),
            "name": failure.get("name"),
            "kind": failure["kind"],
            "waypoint_index": waypoint_index,
            "object_id": object_id,
        }
        abnormal_scenarios.append(scenario)
    result["abnormal_scenarios"] = abnormal_scenarios
    return result
