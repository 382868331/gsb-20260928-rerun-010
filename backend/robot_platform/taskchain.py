from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .robotics import COLLISION_CHECK_DT, segment_velocities, validate_motion
from .robotics import carried_object_pose, forward_kinematics

DEFAULT_HANDOVER_TOLERANCE = 1e-6
REPLAY_PAD = 0.02


def operation_digest(
    operation: dict | None,
    initial_carried: str | None,
    constraint_version: str,
    wait_before: float = 0.0,
    wait_after: float = 0.0,
    wait: float = 0.0,
) -> str:
    payload = {
        "operation": operation,
        "initial_carried": initial_carried,
        "constraint_version": constraint_version,
        "wait_before": wait_before,
        "wait_after": wait_after,
        "wait": wait,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_validation_kwargs(operation: dict) -> dict:
    return {
        "segment_speed_limits": operation.get("segment_speed_limits"),
        "acceleration_limits": operation.get("acceleration_limits"),
        "velocity_transition_time": operation.get("velocity_transition_time", 0.2),
        "gripper_radius": operation.get("gripper_radius", 0.06),
        "carried_objects": operation.get("carried_objects", []),
        "pick_events": operation.get("pick_events", []),
        "place_events": operation.get("place_events", []),
        "constraint_version": operation.get("constraint_version", "1.1"),
    }


def _segment_constraints(operation: dict, dof: int) -> tuple[np.ndarray, float]:
    speed_limits = np.asarray(operation["speed_limits"], dtype=float)
    if operation.get("acceleration_limits") is not None:
        acceleration = np.asarray(operation["acceleration_limits"], dtype=float)
    else:
        acceleration = np.maximum(2.0, speed_limits)
    transition_time = float(operation.get("velocity_transition_time", 0.2))
    if acceleration.shape != (dof,):
        raise ValueError("acceleration_limits must contain one value per joint")
    return acceleration, transition_time


def _operation_velocities(operation: dict, report: dict) -> list[np.ndarray]:
    points = [np.asarray(values, dtype=float) for values in operation["waypoints"]]
    durations = [float(value) for value in report["segment_durations"]]
    return segment_velocities(points, durations)


def _run_segment(operation: dict, initial_carried: str | None) -> dict:
    return validate_motion(
        operation["waypoints"],
        operation["speed_limits"],
        [tuple(pair) for pair in operation["joint_limits"]],
        operation["link_lengths"],
        operation.get("obstacles", []),
        operation.get("link_radius", 0.08),
        operation.get("sample_dt", 0.02),
        initial_carried_object_id=initial_carried,
        **_segment_validation_kwargs(operation),
    )


def _carried_ranges_for_segment(report: dict, offset: float, segment_index: int, initial: str | None) -> list[dict]:
    ranges = []
    current = initial
    current_start = offset
    for event in sorted(report["planned_events"], key=lambda item: item["time"]):
        event_time = offset + float(event["time"])
        if event["type"] == "pick":
            current = str(event["object_id"])
            current_start = event_time
        elif event["type"] == "place" and current == str(event["object_id"]):
            ranges.append({
                "object_id": current,
                "segment_index": segment_index,
                "start_time": current_start,
                "end_time": event_time,
            })
            current = None
    if current is not None:
        ranges.append({
            "object_id": current,
            "segment_index": segment_index,
            "start_time": current_start,
            "end_time": offset + float(report["duration"]),
        })
    return ranges


def _apply_failures(segments: list[dict], failures: list[dict]) -> list[dict]:
    transformed = copy.deepcopy(segments)
    for failure in failures:
        index = int(failure["segment_index"])
        if not 0 <= index < len(transformed):
            raise ValueError("failure segment_index is out of range")
        operation = transformed[index]["operation"]
        if operation is None:
            raise ValueError(f"failure segment {index + 1} is a wait segment without a motion operation")
        object_id = str(failure["object_id"])
        if failure["type"] == "pick_failed":
            operation["pick_events"] = [
                event for event in operation.get("pick_events", []) if str(event["object_id"]) != object_id
            ]
            operation["place_events"] = [
                event for event in operation.get("place_events", []) if str(event["object_id"]) != object_id
            ]
            for later in transformed[index + 1:]:
                later_operation = later["operation"]
                if later_operation is None:
                    continue
                later_operation["place_events"] = [
                    event for event in later_operation.get("place_events", []) if str(event["object_id"]) != object_id
                ]
        elif failure["type"] == "early_release":
            waypoint_index = failure.get("waypoint_index")
            if waypoint_index is None:
                raise ValueError("early_release requires waypoint_index")
            if not 0 <= int(waypoint_index) < len(operation["waypoints"]):
                raise ValueError("early_release waypoint_index is out of range")
            remaining = [
                event for event in operation.get("place_events", []) if str(event["object_id"]) != object_id
            ]
            remaining.append({"waypoint_index": int(waypoint_index), "object_id": object_id})
            operation["place_events"] = remaining
            for later in transformed[index + 1:]:
                later_operation = later["operation"]
                if later_operation is None:
                    continue
                later_operation["place_events"] = [
                    event for event in later_operation.get("place_events", []) if str(event["object_id"]) != object_id
                ]
        else:
            raise ValueError(f"unknown failure type {failure['type']}")
    return transformed


def _segment_waits(segment: dict) -> tuple[float, float, float]:
    wait_before = float(segment.get("wait_before") or 0.0)
    wait_after = float(segment.get("wait_after") or 0.0)
    wait = float(segment.get("wait") or 0.0)
    for label, value in (("wait_before", wait_before), ("wait_after", wait_after), ("wait", wait)):
        if value < 0:
            raise ValueError(f"{label} must be non-negative")
    return wait_before, wait_after, wait


def _build_timeline(segments: list[dict]) -> list[dict]:
    items: list[dict] = []
    for index, segment in enumerate(segments):
        wait_before, wait_after, wait = _segment_waits(segment)
        operation = segment.get("operation")
        if operation is None:
            if wait <= 0:
                raise ValueError(
                    f"segment {index + 1} ('{segment.get('name', segment.get('segment_id', index))}') "
                    "has no motion operation and must define a positive wait duration"
                )
            items.append({"kind": "wait", "duration": wait, "segment": index, "role": "segment"})
            continue
        if wait_before > 0:
            items.append({"kind": "wait", "duration": wait_before, "segment": index, "role": "wait_before"})
        items.append({"kind": "motion", "segment": index})
        if wait_after > 0:
            items.append({"kind": "wait", "duration": wait_after, "segment": index, "role": "wait_after"})
    if not any(item["kind"] == "motion" for item in items):
        raise ValueError("a chained task requires at least one motion segment")
    motion_operations = [
        segments[item["segment"]]["operation"] for item in items if item["kind"] == "motion"
    ]
    for position, item in enumerate(items):
        if item["kind"] != "wait":
            continue
        own = segments[item["segment"]].get("operation")
        if own is not None:
            item["context"] = own
            continue
        context = None
        for later in items[position + 1:]:
            if later["kind"] == "motion":
                context = segments[later["segment"]]["operation"]
                break
        if context is None:
            for earlier in reversed(items[:position]):
                if earlier["kind"] == "motion":
                    context = segments[earlier["segment"]]["operation"]
                    break
        item["context"] = context or motion_operations[0]
    return items


def _wait_frames(
    start_time: float,
    duration: float,
    joint_angles: list[float],
    carried_object_id: str | None,
    context: dict,
    segment_index: int,
) -> list[dict]:
    sample_dt = float(context.get("sample_dt", 0.02))
    link_lengths = context["link_lengths"]
    objects_by_id = {str(item["id"]): item for item in context.get("carried_objects", [])}
    angles = np.asarray(joint_angles, dtype=float)
    vertices = forward_kinematics(angles, link_lengths)
    links = [[float(x), float(y)] for x, y in vertices]
    object_position = None
    if carried_object_id is not None and str(carried_object_id) in objects_by_id:
        pose = carried_object_pose(vertices, angles, objects_by_id[str(carried_object_id)])
        object_position = [float(pose[0]), float(pose[1])]
    count = max(1, int(np.floor(duration / sample_dt + 1e-9)))
    frames = []
    for step in range(count + 1):
        local = min(duration, step * sample_dt)
        if frames and abs(start_time + local - frames[-1]["time"]) <= 1e-9:
            continue
        frames.append({
            "time": start_time + local,
            "local_time": local,
            "segment": segment_index,
            "joint_angles": [float(value) for value in angles],
            "links": links,
            "carried_object_id": carried_object_id,
            "object_position": object_position,
        })
    if frames[-1]["time"] < start_time + duration - 1e-9:
        frames.append({**frames[-1], "time": start_time + duration, "local_time": duration})
    return frames


def _merge_carried_ranges(ranges: list[dict], wait_intervals: list[tuple[float, float]]) -> list[dict]:
    merged: list[dict] = []
    for item in sorted(ranges, key=lambda entry: (entry["start_time"], entry["end_time"])):
        if merged and merged[-1]["object_id"] == item["object_id"]:
            gap_start = merged[-1]["end_time"]
            gap_end = item["start_time"]
            bridged = gap_end - gap_start <= 1e-9 or any(
                start <= gap_start + 1e-9 and end >= gap_end - 1e-9 for start, end in wait_intervals
            )
            if bridged:
                merged[-1]["end_time"] = max(merged[-1]["end_time"], item["end_time"])
                continue
        merged.append(dict(item))
    return merged


def _run_chain(segments: list[dict], handover_tolerance: float) -> dict:
    items = _build_timeline(segments)
    per_segment: list[dict] = []
    hazards: list[dict] = []
    frames: list[dict] = []
    carried_ranges: list[dict] = []
    seam_checks: list[dict] = []
    boundaries: list[float] = []
    wait_intervals: list[tuple[float, float]] = []
    offset = 0.0
    initial: str | None = None
    previous_end_angles: list[float] | None = None
    previous_motion: dict | None = None
    pending_wait = 0.0
    current_angles: list[float] | None = None

    def record_wait(item: dict) -> None:
        nonlocal offset, pending_wait, current_angles
        index = item["segment"]
        segment = segments[index]
        duration = float(item["duration"])
        segment_id = segment.get("segment_id", f"segment-{index + 1}")
        name = segment.get("name", segment_id)
        if current_angles is None:
            context = item["context"]
            current_angles = [float(value) for value in context["waypoints"][0]]
        start_time = offset
        frames.extend(
            _wait_frames(start_time, duration, current_angles, initial, item["context"], index)
        )
        wait_intervals.append((start_time, start_time + duration))
        offset += duration
        pending_wait += duration
        boundaries.append(offset)
        if item["role"] != "segment":
            return
        wait_before, wait_after, wait = _segment_waits(segment)
        digest = operation_digest(None, initial, "wait", wait_before, wait_after, wait)
        state = {
            "joint_angles": [float(value) for value in current_angles],
            "carried_object_id": initial,
            "entry_velocity": [0.0 for _ in current_angles],
        }
        per_segment.append({
            "segment_id": segment_id,
            "name": name,
            "digest": digest,
            "kind": "wait",
            "start_time": start_time,
            "end_time": offset,
            "wait": duration,
            "wait_before": 0.0,
            "wait_after": 0.0,
            "handover_in": state,
            "handover_out": {**state, "entry_velocity": [0.0 for _ in current_angles]},
            "status": "proven",
            "valid": True,
            "error": None,
            "report": None,
            "entry_velocity": [0.0 for _ in current_angles],
            "exit_velocity": [0.0 for _ in current_angles],
        })

    for item in items:
        if item["kind"] == "wait":
            record_wait(item)
            continue
        index = item["segment"]
        segment = segments[index]
        operation = segment["operation"]
        wait_before, wait_after, _ = _segment_waits(segment)
        constraint_version = operation.get("constraint_version", "1.1")
        digest = operation_digest(operation, initial, constraint_version, wait_before, wait_after)
        handover_in = {
            "joint_angles": [float(value) for value in operation["waypoints"][0]],
            "carried_object_id": initial,
        }
        try:
            report = _run_segment(operation, initial)
            error = None
        except ValueError as exc:
            report = None
            error = str(exc)

        name = segment.get("name", segment.get("segment_id", f"segment-{index + 1}"))
        segment_id = segment.get("segment_id", f"segment-{index + 1}")

        if report is None:
            if per_segment:
                seam_checks.append({
                    "between_segments": [index - 1, index],
                    "between_segment_ids": [per_segment[-1]["segment_id"], segment_id],
                    "boundary_time": offset,
                    "status": "unknown",
                    "confidence": "unknown",
                    "carried_object_id": initial,
                    "reason": f"segment {name} cannot start from the handed-over state: {error}",
                    "joints": [],
                })
            hazards.append({
                "start_time": offset,
                "end_time": offset,
                "segment": index,
                "inner_segment": None,
                "kind": "handover_state",
                "message": f"segment {index + 1} cannot start from the handed-over state: {error}",
                "confidence": "boundary",
            })
            per_segment.append({
                "segment_id": segment_id,
                "name": name,
                "digest": digest,
                "kind": "motion",
                "start_time": offset,
                "end_time": offset,
                "wait_before": wait_before,
                "wait_after": wait_after,
                "handover_in": handover_in,
                "handover_out": None,
                "status": "unknown",
                "valid": False,
                "error": error,
                "report": None,
                "entry_velocity": None,
                "exit_velocity": None,
            })
            boundaries.append(offset)
            break

        dof = len(operation["link_lengths"])
        acceleration, transition_time = _segment_constraints(operation, dof)
        velocities = _operation_velocities(operation, report)
        entry_velocity = velocities[0]
        exit_velocity = velocities[-1]

        entry_state = {
            "joint_angles": handover_in["joint_angles"],
            "carried_object_id": initial,
            "entry_velocity": [float(value) for value in entry_velocity],
        }

        if previous_motion is not None:
            previous = previous_motion
            seam_acceleration = np.minimum(previous["acceleration"], acceleration)
            from_velocity = np.asarray(previous["exit_velocity"], dtype=float)
            wait_time = pending_wait
            joint_results = []
            violated_joints = []
            for joint in range(dof):
                accel_limit = float(seam_acceleration[joint])
                if wait_time > 0:
                    stop_change = max(abs(float(from_velocity[joint])), abs(float(entry_velocity[joint])))
                    required_time = stop_change / accel_limit if accel_limit > 0 else float("inf")
                    allowed_time = wait_time
                    joint_ok = required_time <= allowed_time + 1e-9
                    velocity_change = stop_change
                else:
                    velocity_change = abs(float(entry_velocity[joint] - from_velocity[joint]))
                    allowed_time = min(previous["transition_time"], transition_time)
                    required_time = velocity_change / accel_limit if accel_limit > 0 else float("inf")
                    joint_ok = velocity_change <= accel_limit * allowed_time + 1e-9
                joint_results.append({
                    "joint": joint + 1,
                    "from_velocity": float(from_velocity[joint]),
                    "to_velocity": float(entry_velocity[joint]),
                    "velocity_change": velocity_change,
                    "required_transition_time": required_time,
                    "allowed_transition_time": allowed_time,
                    "acceleration_limit": accel_limit,
                    "ok": joint_ok,
                })
                if not joint_ok:
                    violated_joints.append((joint, velocity_change, required_time))
            seam_status = "violated" if violated_joints else "proven"
            seam_info = {
                "between_segments": [previous["index"], index],
                "between_segment_ids": [previous["segment_id"], segment_id],
                "between_names": [previous["name"], name],
                "boundary_time": offset,
                "transition_window_start": offset - wait_time if wait_time > 0 else offset - min(previous["transition_time"], transition_time),
                "transition_window_end": offset if wait_time > 0 else offset + min(previous["transition_time"], transition_time),
                "wait_time": wait_time,
                "status": seam_status,
                "confidence": "boundary",
                "carried_object_id": initial,
                "joint_angles": handover_in["joint_angles"],
                "joints": joint_results,
            }
            seam_checks.append(seam_info)
            for joint, velocity_change, required_time in violated_joints:
                allowed_time = joint_results[joint]["allowed_transition_time"]
                if wait_time > 0:
                    message = (
                        f"joint {joint + 1} must brake {abs(float(from_velocity[joint])):.3f} rad/s to a stop and "
                        f"re-accelerate to {abs(float(entry_velocity[joint])):.3f} rad/s across the "
                        f"{wait_time:.3f}s wait between segments {previous['index'] + 1} ('{previous['name']}') and "
                        f"{index + 1} ('{name}') ending at t={offset:.3f}s; requires {required_time:.3f}s but the "
                        f"wait provides {wait_time:.3f}s "
                        f"(acceleration limit {joint_results[joint]['acceleration_limit']:.3f} rad/s^2; "
                        f"carried object: {initial if initial is not None else 'none'})"
                    )
                else:
                    message = (
                        f"joint {joint + 1} velocity changes {velocity_change:.3f} rad/s between "
                        f"segments {previous['index'] + 1} ('{previous['name']}') and {index + 1} ('{name}') at t={offset:.3f}s; "
                        f"requires {required_time:.3f}s but only {allowed_time:.3f}s transition is allowed "
                        f"(acceleration limit {joint_results[joint]['acceleration_limit']:.3f} rad/s^2; "
                        f"carried object: {initial if initial is not None else 'none'})"
                    )
                hazards.append({
                    "start_time": offset - wait_time if wait_time > 0 else offset,
                    "end_time": offset,
                    "segment": index,
                    "inner_segment": None,
                    "kind": "velocity_transition",
                    "message": message,
                    "confidence": "boundary",
                    "detail": {
                        "joint": joint + 1,
                        "from_velocity": float(from_velocity[joint]),
                        "to_velocity": float(entry_velocity[joint]),
                        "velocity_change": velocity_change,
                        "required_transition_time": required_time,
                        "allowed_transition_time": allowed_time,
                        "acceleration_limit": joint_results[joint]["acceleration_limit"],
                        "wait_time": wait_time,
                        "transition_window_start": seam_info["transition_window_start"],
                        "transition_window_end": seam_info["transition_window_end"],
                        "carried_object_id": initial,
                    },
                })

        if previous_end_angles is not None:
            mismatch = max(
                abs(float(a) - float(b))
                for a, b in zip(previous_end_angles, operation["waypoints"][0])
            )
            if mismatch > handover_tolerance:
                hazards.append({
                    "start_time": offset,
                    "end_time": offset,
                    "segment": index,
                    "inner_segment": None,
                    "kind": "handover_mismatch",
                    "message": (
                        f"segment {index + 1} start pose differs from the previous segment "
                        f"end pose by {mismatch:.4f} rad"
                    ),
                    "confidence": "boundary",
                })

        for hazard in report["hazard_intervals"]:
            hazards.append({
                "start_time": float(hazard["start_time"]) + offset,
                "end_time": float(hazard["end_time"]) + offset,
                "segment": index,
                "inner_segment": hazard["segment"],
                "kind": hazard["kind"],
                "message": hazard["message"],
                "confidence": hazard["confidence"],
            })

        for frame in report["frames"]:
            frames.append({
                "time": float(frame["time"]) + offset,
                "local_time": float(frame["time"]),
                "segment": index,
                "joint_angles": frame["joint_angles"],
                "links": frame["links"],
                "carried_object_id": frame["carried_object_id"],
                "object_position": frame["object_position"],
            })

        carried_ranges.extend(
            item
            for item in _carried_ranges_for_segment(report, offset, index, initial)
            if item["end_time"] > item["start_time"] + 1e-9
        )

        duration = float(report["duration"])
        handover_out = report["end_state"]
        previous_end_angles = handover_out["joint_angles"]
        initial = handover_out["carried_object_id"]
        current_angles = [float(value) for value in previous_end_angles]
        previous_motion = {
            "index": index,
            "segment_id": segment_id,
            "name": name,
            "exit_velocity": [float(value) for value in exit_velocity],
            "acceleration": acceleration,
            "transition_time": transition_time,
        }
        pending_wait = 0.0
        per_segment.append({
            "segment_id": segment_id,
            "name": name,
            "digest": digest,
            "kind": "motion",
            "start_time": offset,
            "end_time": offset + duration,
            "wait_before": wait_before,
            "wait_after": wait_after,
            "handover_in": entry_state,
            "handover_out": handover_out,
            "status": "proven" if report["valid"] else "violated",
            "valid": bool(report["valid"]),
            "error": None,
            "report": report,
            "entry_velocity": [float(value) for value in entry_velocity],
            "exit_velocity": [float(value) for value in exit_velocity],
        })
        offset += duration
        boundaries.append(offset)

    hazards.sort(key=lambda item: (item["start_time"], item["end_time"], item["kind"]))
    unknown_segments = [
        index for index, item in enumerate(per_segment)
        if item.get("kind", "motion") != "wait" and (item["status"] == "unknown" or item["report"] is None)
    ]
    return {
        "segments": per_segment,
        "hazards": hazards,
        "frames": frames,
        "carried_ranges": _merge_carried_ranges(carried_ranges, wait_intervals),
        "seam_checks": seam_checks,
        "boundaries": boundaries,
        "duration": offset,
        "unknown_segments": unknown_segments,
        "valid": (
            not hazards
            and not unknown_segments
            and all(item["valid"] for item in per_segment)
            and len(per_segment) == len(segments)
        ),
    }


def _replay_window(first_hazard: dict, boundaries: list[float], duration: float) -> dict:
    start = max(0.0, first_hazard["start_time"] - REPLAY_PAD)
    end = min(duration, first_hazard["end_time"] + REPLAY_PAD)
    crossing = []
    for boundary in boundaries[:-1]:
        if start - REPLAY_PAD <= boundary <= end + REPLAY_PAD:
            start = min(start, max(0.0, boundary - REPLAY_PAD))
            end = max(end, min(duration, boundary + REPLAY_PAD))
            crossing.append(boundary)
    return {
        "start_time": start,
        "end_time": end,
        "segment": first_hazard["segment"],
        "crosses_boundary": bool(crossing),
        "boundaries": crossing,
    }


def _stale_segments(current: list[dict], previous: dict | None) -> list[dict]:
    if not previous:
        return []
    previous_segments = previous.get("segments", [])
    stale = []
    for index, segment in enumerate(current):
        if index >= len(previous_segments):
            stale.append({"segment": index, "reason": "new segment without a previous conclusion"})
            continue
        before = previous_segments[index]
        reason = None
        if before.get("segment_id") != segment["segment_id"]:
            reason = "segment was replaced"
        elif before.get("digest") != segment["digest"]:
            reason = "operation version changed"
        elif before.get("handover_in") != segment["handover_in"]:
            reason = "handover-in condition changed"
        elif index > 0 and any(item["segment"] == index - 1 for item in stale):
            reason = "predecessor entry state changed; seam conclusion must be rechecked"
        if reason is not None:
            stale.append({"segment": index, "reason": reason})
    return stale


def _conclusion_sources(per_segment: list[dict], constraint_version: str) -> dict:
    return {
        "constraint_version": constraint_version,
        "segments": [
            {
                "segment_id": item["segment_id"],
                "digest": item["digest"],
                "handover_in": item["handover_in"],
                "handover_out": item["handover_out"],
                "entry_velocity": item["entry_velocity"],
                "exit_velocity": item["exit_velocity"],
            }
            for item in per_segment
        ],
    }


def validate_task(
    task: dict,
    failures: list[dict] | None = None,
    previous: dict | None = None,
) -> dict:
    segments = task.get("segments", [])
    if len(segments) < 2:
        raise ValueError("a chained task requires at least two segments")
    handover_tolerance = float(task.get("handover_tolerance", DEFAULT_HANDOVER_TOLERANCE))
    if handover_tolerance < 0:
        raise ValueError("handover_tolerance must be non-negative")
    constraint_version = task.get("constraint_version", "1.1")

    normal = _run_chain(segments, handover_tolerance)
    stale = _stale_segments(
        [{"segment_id": item["segment_id"], "digest": item["digest"], "handover_in": item["handover_in"]} for item in normal["segments"]],
        previous,
    )

    first_hazard = normal["hazards"][0] if normal["hazards"] else None
    replay_window = (
        None if first_hazard is None else _replay_window(first_hazard, normal["boundaries"], normal["duration"])
    )

    abnormal = None
    unknown_intervals: list[dict] = []
    if failures:
        transformed = _apply_failures(segments, failures)
        abnormal_chain = _run_chain(transformed, handover_tolerance)
        abnormal_first = abnormal_chain["hazards"][0] if abnormal_chain["hazards"] else None
        for failure in failures:
            index = int(failure["segment_index"])
            if index >= len(normal["segments"]):
                continue
            segment_info = normal["segments"][index]
            failure_time = segment_info["start_time"]
            if segment_info["report"] is not None:
                for event in segment_info["report"]["planned_events"]:
                    if str(event["object_id"]) == str(failure["object_id"]) and (
                        (failure["type"] == "pick_failed" and event["type"] == "pick")
                        or (failure["type"] == "early_release" and event["type"] == "place")
                    ):
                        failure_time = segment_info["start_time"] + float(event["time"])
                        break
            unknown_intervals.append({
                "start_time": failure_time,
                "end_time": normal["duration"],
                "segment": index,
                "kind": "unconfirmed_object_state",
                "message": (
                    f"object {failure['object_id']} real state cannot be confirmed after "
                    f"{failure['type']} in segment {index + 1}"
                ),
                "confidence": "unknown",
            })
        abnormal = {
            "valid": abnormal_chain["valid"],
            "duration": abnormal_chain["duration"],
            "segments": [
                {key: value for key, value in item.items() if key != "report"}
                for item in abnormal_chain["segments"]
            ],
            "hazards": abnormal_chain["hazards"],
            "first_hazard": abnormal_first,
            "frames": abnormal_chain["frames"],
            "carried_ranges": abnormal_chain["carried_ranges"],
            "seam_checks": abnormal_chain["seam_checks"],
            "unknown_segments": abnormal_chain["unknown_segments"],
            "replay_window": (
                None if abnormal_first is None else _replay_window(abnormal_first, abnormal_chain["boundaries"], abnormal_chain["duration"])
            ),
        }

    concern_candidates = []
    if first_hazard is not None:
        concern_candidates.append((first_hazard["start_time"], "normal", first_hazard))
    if abnormal is not None and abnormal["first_hazard"] is not None:
        concern_candidates.append((abnormal["first_hazard"]["start_time"], "abnormal", abnormal["first_hazard"]))
    for interval in unknown_intervals:
        concern_candidates.append((interval["start_time"], "unknown", interval))
    concern_candidates.sort(key=lambda item: item[0])
    first_concern = concern_candidates[0] if concern_candidates else None

    return {
        "valid": normal["valid"],
        "duration": normal["duration"],
        "constraint_version": constraint_version,
        "handover_tolerance": handover_tolerance,
        "segments": normal["segments"],
        "hazards": normal["hazards"],
        "first_hazard": first_hazard,
        "replay_window": replay_window,
        "frames": normal["frames"],
        "carried_ranges": normal["carried_ranges"],
        "boundaries": normal["boundaries"],
        "seam_checks": normal["seam_checks"],
        "unknown_segments": normal["unknown_segments"],
        "stale_segments": stale,
        "recheck_required": [item["segment"] for item in stale],
        "conclusion_sources": _conclusion_sources(normal["segments"], constraint_version),
        "abnormal": abnormal,
        "unknown_intervals": unknown_intervals,
        "comparison": {
            "normal_first_hazard": first_hazard,
            "abnormal_first_hazard": None if abnormal is None else abnormal["first_hazard"],
            "first_unknown": unknown_intervals[0] if unknown_intervals else None,
            "first_concern_time": None if first_concern is None else first_concern[0],
            "first_concern_source": None if first_concern is None else first_concern[1],
            "first_concern": None if first_concern is None else first_concern[2],
        },
        "collision_check_dt": COLLISION_CHECK_DT,
    }
