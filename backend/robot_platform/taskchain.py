from __future__ import annotations

import copy
import hashlib
import json

import numpy as np

from .robotics import COLLISION_CHECK_DT, carried_object_pose, forward_kinematics, segment_velocities, validate_motion

DEFAULT_HANDOVER_TOLERANCE = 1e-6
REPLAY_PAD = 0.02


def operation_digest(operation: dict, initial_carried: str | None, constraint_version: str) -> str:
    payload = {
        "operation": operation,
        "initial_carried": initial_carried,
        "constraint_version": constraint_version,
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


def _expand_segment_items(segment: dict) -> list[tuple[str, object]]:
    operation = segment.get("operation")
    wait_value = segment.get("wait")
    if operation is None and wait_value is None:
        raise ValueError("each task segment requires an operation or a wait duration")
    if operation is None:
        duration = float(wait_value)
        if duration < 0:
            raise ValueError("wait duration must be non-negative")
        return [("wait", duration)]
    wait_before = float(operation.get("wait_before", 0.0))
    wait_after = float(operation.get("wait_after", 0.0))
    if wait_before < 0 or wait_after < 0:
        raise ValueError("wait durations must be non-negative")
    items: list[tuple[str, object]] = []
    if wait_before > 0:
        items.append(("wait", wait_before))
    items.append(("motion", operation))
    if wait_after > 0:
        items.append(("wait", wait_after))
    return items


def _hold_frames(
    angles: list[float],
    link_lengths: list[float],
    carried_id: str | None,
    objects_by_id: dict,
    start: float,
    duration: float,
    sample_dt: float,
    segment_index: int,
) -> list[dict]:
    if duration <= 0:
        return []
    vertices = forward_kinematics(np.asarray(angles, dtype=float), link_lengths)
    object_position = None
    if carried_id is not None and carried_id in objects_by_id:
        object_position = carried_object_pose(vertices, np.asarray(angles, dtype=float), objects_by_id[carried_id])
    count = max(1, int(np.floor(duration / sample_dt + 1e-9)))
    frames = []
    for step in range(count + 1):
        time_value = min(start + step * sample_dt, start + duration)
        frames.append({
            "time": float(time_value),
            "local_time": float(time_value - start),
            "segment": segment_index,
            "joint_angles": [float(value) for value in angles],
            "links": [[float(x), float(y)] for x, y in vertices],
            "carried_object_id": carried_id,
            "object_position": None if object_position is None else [float(object_position[0]), float(object_position[1])],
        })
    return frames


def _merge_carried_ranges(ranges: list[dict], wait_intervals: list[tuple[float, float]]) -> list[dict]:
    merged: list[dict] = []
    for item in sorted(ranges, key=lambda entry: (entry["start_time"], entry["object_id"])):
        if merged and merged[-1]["object_id"] == item["object_id"]:
            gap_start = merged[-1]["end_time"]
            gap_end = item["start_time"]
            gap = gap_end - gap_start
            if gap <= 1e-9:
                merged[-1]["end_time"] = max(merged[-1]["end_time"], item["end_time"])
                continue
            covered = 0.0
            for wait_start, wait_end in wait_intervals:
                covered += max(0.0, min(gap_end, wait_end) - max(gap_start, wait_start))
            if covered >= gap - 1e-9:
                merged[-1]["end_time"] = max(merged[-1]["end_time"], item["end_time"])
                continue
        merged.append(dict(item))
    return merged


def _apply_failures(segments: list[dict], failures: list[dict]) -> list[dict]:
    transformed = copy.deepcopy(segments)
    for failure in failures:
        index = int(failure["segment_index"])
        if not 0 <= index < len(transformed):
            raise ValueError("failure segment_index is out of range")
        operation = transformed[index].get("operation")
        if operation is None:
            raise ValueError("failure segment must reference a motion segment")
        object_id = str(failure["object_id"])
        if failure["type"] == "pick_failed":
            operation["pick_events"] = [
                event for event in operation.get("pick_events", []) if str(event["object_id"]) != object_id
            ]
            operation["place_events"] = [
                event for event in operation.get("place_events", []) if str(event["object_id"]) != object_id
            ]
            for later in transformed[index + 1:]:
                later_operation = later.get("operation")
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
                later_operation = later.get("operation")
                if later_operation is None:
                    continue
                later_operation["place_events"] = [
                    event for event in later_operation.get("place_events", []) if str(event["object_id"]) != object_id
                ]
        else:
            raise ValueError(f"unknown failure type {failure['type']}")
    return transformed


def _run_chain(segments: list[dict], handover_tolerance: float) -> dict:
    per_segment: list[dict] = []
    hazards: list[dict] = []
    frames: list[dict] = []
    carried_ranges: list[dict] = []
    seam_checks: list[dict] = []
    boundaries: list[float] = []
    wait_intervals: list[tuple[float, float]] = []
    offset = 0.0
    initial: str | None = None
    current_pose: list[float] | None = None
    current_link_lengths: list[float] | None = None
    pending_dwell = 0.0
    last_sample_dt = 0.02
    previous_motion: dict | None = None
    objects_by_id: dict = {}
    for segment in segments:
        operation = segment.get("operation")
        if operation:
            for carried in operation.get("carried_objects", []):
                objects_by_id[str(carried["id"])] = carried

    for index, segment in enumerate(segments):
        name = segment.get("name", segment.get("segment_id", f"segment-{index + 1}"))
        segment_id = segment.get("segment_id", f"segment-{index + 1}")
        segment_start = offset
        items = _expand_segment_items(segment)
        operation = segment.get("operation")
        constraint_version = (operation or {}).get("constraint_version", "1.1")
        digest_payload = operation if operation is not None else {"wait": float(segment.get("wait", 0.0))}
        digest = operation_digest(digest_payload, initial, constraint_version)

        if operation is None:
            wait_duration = float(segment.get("wait", 0.0))
            handover_in = {
                "joint_angles": None if current_pose is None else [float(value) for value in current_pose],
                "carried_object_id": initial,
            }
            if current_pose is None:
                hazards.append({
                    "start_time": offset,
                    "end_time": offset,
                    "segment": index,
                    "inner_segment": None,
                    "kind": "handover_state",
                    "message": f"wait segment {index + 1} has no preceding motion to hold its pose",
                    "confidence": "boundary",
                })
                per_segment.append({
                    "segment_id": segment_id,
                    "name": name,
                    "digest": digest,
                    "start_time": segment_start,
                    "end_time": offset,
                    "handover_in": handover_in,
                    "handover_out": None,
                    "status": "unknown",
                    "valid": False,
                    "error": "wait segment requires a preceding motion segment",
                    "report": None,
                    "entry_velocity": None,
                    "exit_velocity": None,
                    "kind": "wait",
                })
                boundaries.append(offset)
                break
            frames.extend(
                _hold_frames(current_pose, current_link_lengths, initial, objects_by_id, offset, wait_duration, last_sample_dt, index)
            )
            if wait_duration > 0:
                wait_intervals.append((offset, offset + wait_duration))
                pending_dwell += wait_duration
                offset += wait_duration
            per_segment.append({
                "segment_id": segment_id,
                "name": name,
                "digest": digest,
                "start_time": segment_start,
                "end_time": offset,
                "handover_in": handover_in,
                "handover_out": {
                    "joint_angles": [float(value) for value in current_pose],
                    "carried_object_id": initial,
                },
                "status": "proven",
                "valid": True,
                "error": None,
                "report": None,
                "entry_velocity": None,
                "exit_velocity": None,
                "kind": "wait",
                "wait_duration": wait_duration,
            })
            boundaries.append(offset)
            continue

        wait_before = float(operation.get("wait_before", 0.0))
        wait_after = float(operation.get("wait_after", 0.0))
        handover_in = {
            "joint_angles": [float(value) for value in operation["waypoints"][0]],
            "carried_object_id": initial,
        }

        if wait_before > 0:
            hold_pose = current_pose if current_pose is not None else [float(value) for value in operation["waypoints"][0]]
            frames.extend(
                _hold_frames(hold_pose, operation["link_lengths"], initial, objects_by_id, offset, wait_before, float(operation.get("sample_dt", 0.02)), index)
            )
            wait_intervals.append((offset, offset + wait_before))
            pending_dwell += wait_before
            offset += wait_before

        try:
            report = _run_segment(operation, initial)
            error = None
        except ValueError as exc:
            report = None
            error = str(exc)

        if report is None:
            if previous_motion is not None:
                seam_checks.append({
                    "between_segments": [index - 1, index],
                    "between_segment_ids": [previous_motion["segment_id"], segment_id],
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
                "start_time": segment_start,
                "end_time": offset,
                "handover_in": handover_in,
                "handover_out": None,
                "status": "unknown",
                "valid": False,
                "error": error,
                "report": None,
                "entry_velocity": None,
                "exit_velocity": None,
                "kind": "motion",
            })
            boundaries.append(offset)
            break

        dof = len(operation["link_lengths"])
        acceleration, transition_time = _segment_constraints(operation, dof)
        velocities = _operation_velocities(operation, report)
        entry_velocity = velocities[0]
        exit_velocity = velocities[-1]
        motion_start = offset

        entry_state = {
            "joint_angles": handover_in["joint_angles"],
            "carried_object_id": initial,
            "entry_velocity": [float(value) for value in entry_velocity],
        }

        if previous_motion is not None:
            previous = previous_motion
            previous_acceleration, previous_transition_time = _segment_constraints(
                previous["operation"], dof
            )
            seam_acceleration = np.minimum(previous_acceleration, acceleration)
            seam_transition_time = min(previous_transition_time, transition_time)
            allowed_transition = seam_transition_time + pending_dwell
            from_velocity = np.asarray(previous["exit_velocity"], dtype=float)
            joint_results = []
            violated_joints = []
            for joint in range(dof):
                velocity_change = abs(float(entry_velocity[joint] - from_velocity[joint]))
                allowed_change = float(seam_acceleration[joint]) * allowed_transition
                required_time = (
                    velocity_change / float(seam_acceleration[joint])
                    if float(seam_acceleration[joint]) > 0
                    else float("inf")
                )
                joint_ok = velocity_change <= allowed_change + 1e-9
                joint_results.append({
                    "joint": joint + 1,
                    "from_velocity": float(from_velocity[joint]),
                    "to_velocity": float(entry_velocity[joint]),
                    "velocity_change": velocity_change,
                    "required_transition_time": required_time,
                    "allowed_transition_time": allowed_transition,
                    "acceleration_limit": float(seam_acceleration[joint]),
                    "ok": joint_ok,
                })
                if not joint_ok:
                    violated_joints.append((joint, velocity_change, required_time))
            seam_status = "violated" if violated_joints else "proven"
            seam_info = {
                "between_segments": [previous["index"], index],
                "between_segment_ids": [previous["segment_id"], segment_id],
                "between_names": [previous["name"], name],
                "boundary_time": previous["motion_end"],
                "wait_duration": pending_dwell,
                "transition_window_start": previous["motion_end"] - seam_transition_time,
                "transition_window_end": motion_start + seam_transition_time,
                "status": seam_status,
                "confidence": "boundary",
                "carried_object_id": initial,
                "joint_angles": handover_in["joint_angles"],
                "joints": joint_results,
            }
            seam_checks.append(seam_info)
            for joint, velocity_change, required_time in violated_joints:
                hazards.append({
                    "start_time": previous["motion_end"],
                    "end_time": motion_start,
                    "segment": index,
                    "inner_segment": None,
                    "kind": "velocity_transition",
                    "message": (
                        f"joint {joint + 1} velocity changes {velocity_change:.3f} rad/s between "
                        f"segments {previous['index'] + 1} ('{previous['name']}') and {index + 1} ('{name}') "
                        f"at t={previous['motion_end']:.3f}s; "
                        f"requires {required_time:.3f}s but only {allowed_transition:.3f}s transition is allowed "
                        f"(base {seam_transition_time:.3f}s + wait {pending_dwell:.3f}s; "
                        f"(acceleration limit {float(seam_acceleration[joint]):.3f} rad/s^2; "
                        f"carried object: {initial if initial is not None else 'none'})"
                    ),
                    "confidence": "boundary",
                    "detail": {
                        "joint": joint + 1,
                        "from_velocity": float(from_velocity[joint]),
                        "to_velocity": float(entry_velocity[joint]),
                        "velocity_change": velocity_change,
                        "required_transition_time": required_time,
                        "allowed_transition_time": allowed_transition,
                        "wait_duration": pending_dwell,
                        "acceleration_limit": float(seam_acceleration[joint]),
                        "transition_window_start": previous["motion_end"] - seam_transition_time,
                        "transition_window_end": motion_start + seam_transition_time,
                        "carried_object_id": initial,
                    },
                })

        if current_pose is not None:
            mismatch = max(
                abs(float(a) - float(b))
                for a, b in zip(current_pose, operation["waypoints"][0])
            )
            if mismatch > handover_tolerance:
                hazards.append({
                    "start_time": motion_start,
                    "end_time": motion_start,
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
                "start_time": float(hazard["start_time"]) + motion_start,
                "end_time": float(hazard["end_time"]) + motion_start,
                "segment": index,
                "inner_segment": hazard["segment"],
                "kind": hazard["kind"],
                "message": hazard["message"],
                "confidence": hazard["confidence"],
            })

        for frame in report["frames"]:
            frames.append({
                "time": float(frame["time"]) + motion_start,
                "local_time": float(frame["time"]),
                "segment": index,
                "joint_angles": frame["joint_angles"],
                "links": frame["links"],
                "carried_object_id": frame["carried_object_id"],
                "object_position": frame["object_position"],
            })

        carried_ranges.extend(
            item
            for item in _carried_ranges_for_segment(report, motion_start, index, initial)
            if item["end_time"] > item["start_time"] + 1e-9
        )

        duration = float(report["duration"])
        handover_out = report["end_state"]
        current_pose = [float(value) for value in handover_out["joint_angles"]]
        current_link_lengths = [float(value) for value in operation["link_lengths"]]
        initial = handover_out["carried_object_id"]
        last_sample_dt = float(operation.get("sample_dt", 0.02))
        motion_end = motion_start + duration
        offset = motion_end
        previous_motion = {
            "index": index,
            "segment_id": segment_id,
            "name": name,
            "operation": operation,
            "exit_velocity": [float(value) for value in exit_velocity],
            "motion_end": motion_end,
        }
        pending_dwell = 0.0
        if wait_after > 0:
            frames.extend(
                _hold_frames(current_pose, current_link_lengths, initial, objects_by_id, offset, wait_after, last_sample_dt, index)
            )
            wait_intervals.append((offset, offset + wait_after))
            pending_dwell += wait_after
            offset += wait_after
        per_segment.append({
            "segment_id": segment_id,
            "name": name,
            "digest": digest,
            "start_time": segment_start,
            "end_time": offset,
            "handover_in": entry_state,
            "handover_out": handover_out,
            "status": "proven" if report["valid"] else "violated",
            "valid": bool(report["valid"]),
            "error": None,
            "report": report,
            "entry_velocity": [float(value) for value in entry_velocity],
            "exit_velocity": [float(value) for value in exit_velocity],
            "kind": "motion",
            "wait_before": wait_before,
            "wait_after": wait_after,
        })
        boundaries.append(offset)

    carried_ranges = _merge_carried_ranges(carried_ranges, wait_intervals)
    hazards.sort(key=lambda item: (item["start_time"], item["end_time"], item["kind"]))
    unknown_segments = [
        index for index, item in enumerate(per_segment)
        if item["status"] == "unknown"
    ]
    return {
        "segments": per_segment,
        "hazards": hazards,
        "frames": frames,
        "carried_ranges": carried_ranges,
        "seam_checks": seam_checks,
        "boundaries": boundaries,
        "wait_intervals": [
            {"start_time": start, "end_time": end} for start, end in wait_intervals
        ],
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
        "wait_intervals": normal["wait_intervals"],
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
