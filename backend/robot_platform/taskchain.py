from __future__ import annotations

import copy
import hashlib
import json

from .robotics import COLLISION_CHECK_DT, validate_motion

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
        "wait_duration": operation.get("wait_duration", 0.0),
    }


def _operation_limits(operation: dict) -> tuple[float, list[float]]:
    transition_time = float(operation.get("velocity_transition_time", 0.2))
    acceleration = operation.get("acceleration_limits")
    if acceleration is None:
        acceleration = [max(2.0, float(value)) for value in operation["speed_limits"]]
    return transition_time, [float(value) for value in acceleration]


def _initial_dynamic_state(operation: dict, carried_object_id: str | None) -> dict:
    return {
        "motion_kind": "plan_start",
        "joint_angles": [float(value) for value in operation["waypoints"][0]],
        "joint_velocities": [0.0 for _ in operation["speed_limits"]],
        "translation_velocity": [0.0, 0.0],
        "rotation_velocity": 0.0,
        "carried_object_id": carried_object_id,
        "available_buffer_time": 0.0,
    }


def _seam_constraint(previous: dict, current: dict, previous_operation: dict, current_operation: dict) -> dict | None:
    previous_time, previous_acceleration = _operation_limits(previous_operation)
    current_time, current_acceleration = _operation_limits(current_operation)
    previous_kind = previous.get("motion_kind")
    current_kind = current.get("motion_kind")
    if previous_kind in {None, "unconstrained_zero_duration"} or current_kind in {None, "unconstrained_zero_duration"}:
        return None

    previous_wait = previous_kind == "wait"
    current_wait = current_kind == "wait"
    if previous_wait or current_wait:
        if previous_wait and current_wait:
            return None
        if previous_wait:
            return {"available_time": current_time, "acceleration_limits": current_acceleration, "side": "leave"}
        return {"available_time": previous_time, "acceleration_limits": previous_acceleration, "side": "enter"}

    available_time = min(previous_time, current_time)
    acceleration_limits = [min(left, right) for left, right in zip(previous_acceleration, current_acceleration)]
    return {
        "available_time": available_time,
        "acceleration_limits": acceleration_limits,
        "side": "cross",
    }


def _dynamic_seam_hazard(
    previous: dict,
    current: dict,
    previous_index: int,
    current_index: int,
    previous_operation: dict,
    current_operation: dict,
    boundary_time: float,
) -> dict | None:
    constraint = _seam_constraint(previous, current, previous_operation, current_operation)
    if constraint is None:
        return None
    previous_velocity = previous.get("joint_velocities")
    current_velocity = current.get("joint_velocities")
    if previous_velocity is None or current_velocity is None:
        return {
            "start_time": boundary_time,
            "end_time": boundary_time,
            "segment": current_index,
            "inner_segment": None,
            "kind": "missing_dynamic_state",
            "message": f"segments {previous_index + 1} and {current_index + 1}: entering velocity or load state is missing",
            "confidence": "unknown",
            "details": {
                "from_segment": previous_index,
                "to_segment": current_index,
                "missing": "joint_velocities",
            },
        }
    if previous.get("carried_object_id") != current.get("carried_object_id") and current.get("motion_kind") != "wait":
        # A zero-duration pick/place event is expected to change load; validate_motion receives that state.
        pass

    available_time = float(constraint["available_time"])
    for joint, (left, right, acceleration_limit) in enumerate(
        zip(previous_velocity, current_velocity, constraint["acceleration_limits"])
    ):
        required_time = abs(float(right) - float(left)) / float(acceleration_limit)
        if required_time > available_time + 1e-9:
            start_time = boundary_time - available_time if constraint["side"] in {"cross", "enter"} else boundary_time
            end_time = boundary_time if constraint["side"] in {"cross", "enter"} else boundary_time + available_time
            return {
                "start_time": start_time,
                "end_time": end_time,
                "segment": current_index,
                "inner_segment": None,
                "kind": "velocity_transition",
                "message": (
                    f"joint {joint + 1} cannot change velocity between segments {previous_index + 1} and {current_index + 1}: "
                    f"{float(left):.6g} to {float(right):.6g} rad/s requires {required_time:.6g}s, "
                    f"but only {available_time:.6g}s is available under acceleration limit {float(acceleration_limit):.6g} rad/s^2"
                ),
                "confidence": "boundary",
                "details": {
                    "from_segment": previous_index,
                    "to_segment": current_index,
                    "joint": joint,
                    "from_joint_velocity": float(left),
                    "to_joint_velocity": float(right),
                    "from_translation_velocity": previous.get("translation_velocity"),
                    "to_translation_velocity": current.get("translation_velocity"),
                    "from_rotation_velocity": previous.get("rotation_velocity"),
                    "to_rotation_velocity": current.get("rotation_velocity"),
                    "from_carried_object_id": previous.get("carried_object_id"),
                    "to_carried_object_id": current.get("carried_object_id"),
                    "required_transition_time": required_time,
                    "allowed_transition_time": available_time,
                    "acceleration_limit": float(acceleration_limit),
                    "transition_side": constraint["side"],
                },
            }
    return None


def _grouped_wait_hazard(
    before: dict,
    after: dict,
    before_index: int,
    after_index: int,
    before_operation: dict,
    after_operation: dict,
    wait_time: float,
    start_time: float,
    end_time: float,
    connector_kind: str = "wait",
) -> dict | None:
    before_transition, before_acceleration = _operation_limits(before_operation)
    after_transition, after_acceleration = _operation_limits(after_operation)
    for joint, (start_velocity, end_velocity, deceleration, acceleration) in enumerate(
        zip(before["joint_velocities"], after["joint_velocities"], before_acceleration, after_acceleration)
    ):
        stop_required = abs(float(start_velocity)) / float(deceleration)
        start_required = abs(float(end_velocity)) / float(acceleration)
        acceleration_limit = min(float(deceleration), float(acceleration))
        required_time = stop_required + start_required
        available_time = float(before_transition) + float(wait_time) + float(after_transition)
        if required_time > available_time + 1e-9:
            return {
                "start_time": start_time,
                "end_time": end_time,
                "segment": after_index,
                "inner_segment": None,
                "kind": "velocity_transition",
                "message": (
                    f"joint {joint + 1} cannot complete the buffered transition between segments {before_index + 1} "
                    f"and {after_index + 1}: velocity {float(start_velocity):.6g} -> 0 -> {float(end_velocity):.6g} rad/s "
                    f"requires {required_time:.6g}s, but {before_transition:.6g}s + {wait_time:.6g}s {connector_kind} + "
                    f"{after_transition:.6g}s gives only {available_time:.6g}s"
                ),
                "confidence": "boundary",
                "details": {
                    "from_segment": before_index,
                    "to_segment": after_index,
                    "joint": joint,
                    "connector_kind": connector_kind,
                    "from_joint_velocity": float(start_velocity),
                    "to_joint_velocity": float(end_velocity),
                    "from_translation_velocity": before.get("translation_velocity"),
                    "to_translation_velocity": after.get("translation_velocity"),
                    "from_rotation_velocity": before.get("rotation_velocity"),
                    "to_rotation_velocity": after.get("rotation_velocity"),
                    "from_carried_object_id": before.get("carried_object_id"),
                    "to_carried_object_id": after.get("carried_object_id"),
                    "stop_required_time": stop_required,
                    "start_required_time": start_required,
                    "required_transition_time": required_time,
                    "wait_time": float(wait_time),
                    "allowed_transition_time": available_time,
                    "deceleration_limit": float(deceleration),
                    "acceleration_limit": float(acceleration),
                    "effective_acceleration_limit": acceleration_limit,
                },
            }
    return None


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
    boundaries: list[float] = []
    offset = 0.0
    initial: str | None = None
    previous_end_angles: list[float] | None = None
    incoming_dynamic: dict | None = None
    incoming_dynamic_index: int | None = None
    incoming_dynamic_operation: dict | None = None
    velocity_anchor_index: int | None = None
    velocity_anchor_operation: dict | None = None
    pending_wait_anchor: dict | None = None
    pending_wait_index: int | None = None
    pending_wait_operation: dict | None = None
    pending_wait_time = 0.0
    pending_wait_window_start: float | None = None

    for index, segment in enumerate(segments):
        operation = segment["operation"]
        constraint_version = operation.get("constraint_version", "1.1")
        digest = operation_digest(operation, initial, constraint_version)
        start_dynamic = incoming_dynamic or _initial_dynamic_state(operation, initial)
        handover_in = {
            "joint_angles": [float(value) for value in operation["waypoints"][0]],
            "carried_object_id": initial,
            "dynamic": start_dynamic,
        }
        try:
            report = _run_segment(operation, initial)
            error = None
        except ValueError as exc:
            report = None
            error = str(exc)

        if report is None:
            hazards.append({
                "start_time": offset,
                "end_time": offset,
                "segment": index,
                "inner_segment": None,
                "kind": "handover_state",
                "message": f"segment {index + 1} cannot start from the handed-over state or load definition: {error}",
                "confidence": "unknown",
                "details": {"reason": "required_entry_state_unavailable"},
            })
            per_segment.append({
                "segment_id": segment.get("segment_id", f"segment-{index + 1}"),
                "name": segment.get("name", f"segment-{index + 1}"),
                "digest": digest,
                "start_time": offset,
                "end_time": offset,
                "handover_in": handover_in,
                "handover_out": None,
                "valid": False,
                "execution_status": "unknown",
                "error": error,
                "report": None,
            })
            boundaries.append(offset)
            for unknown_index in range(index + 1, len(segments)):
                per_segment.append({
                    "segment_id": segments[unknown_index].get("segment_id", f"segment-{unknown_index + 1}"),
                    "name": segments[unknown_index].get("name", f"segment-{unknown_index + 1}"),
                    "digest": None,
                    "start_time": offset,
                    "end_time": offset,
                    "handover_in": None,
                    "handover_out": None,
                    "valid": False,
                    "execution_status": "unknown",
                    "error": "required predecessor state is unavailable",
                    "report": None,
                })
                boundaries.append(offset)
            break

        duration = float(report["duration"])
        report_start_dynamic = report["dynamic_state"]["at_start"]
        report_end_dynamic = report["dynamic_state"]["at_end"]
        zero_duration = duration <= 1e-9
        connector_kind = report_start_dynamic.get("motion_kind")
        current_is_connector = connector_kind in {"wait", "turn_in_place"}

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
                    "details": {
                        "from_segment": index - 1,
                        "to_segment": index,
                        "allowed_pose_error": handover_tolerance,
                        "actual_pose_error": mismatch,
                    },
                })
        else:
            mismatch = 0.0

        if mismatch <= handover_tolerance and not zero_duration and incoming_dynamic is not None:
            if current_is_connector:
                if pending_wait_anchor is None:
                    pending_wait_anchor = incoming_dynamic
                    pending_wait_index = velocity_anchor_index if velocity_anchor_index is not None else incoming_dynamic_index
                    pending_wait_operation = velocity_anchor_operation or incoming_dynamic_operation
                    prior_transition, _ = _operation_limits(incoming_dynamic_operation or operation)
                    pending_wait_window_start = offset - float(prior_transition)
                pending_wait_time += duration
            else:
                if pending_wait_anchor is not None:
                    seam_hazard = _grouped_wait_hazard(
                        pending_wait_anchor,
                        report_start_dynamic,
                        pending_wait_index if pending_wait_index is not None else 0,
                        index,
                        pending_wait_operation or operation,
                        operation,
                        pending_wait_time,
                        pending_wait_window_start if pending_wait_window_start is not None else offset,
                        offset + float(operation.get("velocity_transition_time", 0.2)),
                        connector_kind or "wait",
                    )
                    pending_wait_anchor = None
                    pending_wait_index = None
                    pending_wait_operation = None
                    pending_wait_time = 0.0
                    pending_wait_window_start = None
                else:
                    seam_hazard = _dynamic_seam_hazard(
                        incoming_dynamic,
                        report_start_dynamic,
                        (
                            velocity_anchor_index if velocity_anchor_index is not None
                            else (incoming_dynamic_index if incoming_dynamic_index is not None else 0)
                        ),
                        index,
                        velocity_anchor_operation or incoming_dynamic_operation or operation,
                        operation,
                        offset,
                    )
                if seam_hazard is not None:
                    hazards.append(seam_hazard)

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

        handover_out = report["end_state"]
        if zero_duration:
            outgoing_dynamic = copy.deepcopy(pending_wait_anchor or incoming_dynamic or report_end_dynamic)
            outgoing_dynamic["joint_angles"] = report_end_dynamic["joint_angles"]
            outgoing_dynamic["carried_object_id"] = handover_out["carried_object_id"]
        else:
            outgoing_dynamic = report_end_dynamic
        handover_out["dynamic"] = outgoing_dynamic
        previous_end_angles = handover_out["joint_angles"]
        initial = handover_out["carried_object_id"]
        per_segment.append({
            "segment_id": segment.get("segment_id", f"segment-{index + 1}"),
            "name": segment.get("name", f"segment-{index + 1}"),
            "digest": digest,
            "start_time": offset,
            "end_time": offset + duration,
            "handover_in": handover_in,
            "handover_out": handover_out,
            "valid": bool(report["valid"]),
            "execution_status": "executable" if report["valid"] else "violated",
            "error": None,
            "report": report,
        })
        offset += duration
        boundaries.append(offset)
        incoming_dynamic = outgoing_dynamic
        incoming_dynamic_index = index
        incoming_dynamic_operation = operation
        if not zero_duration and not current_is_connector:
            velocity_anchor_index = index
            velocity_anchor_operation = operation

    if pending_wait_anchor is not None and incoming_dynamic_operation is not None and pending_wait_index is not None and pending_wait_operation is not None:
        zero_dynamic = {
            **pending_wait_anchor,
            "joint_velocities": [0.0 for _ in pending_wait_anchor["joint_velocities"]],
            "translation_velocity": [0.0, 0.0],
            "rotation_velocity": 0.0,
            "motion_kind": "plan_end",
        }
        final_operation = copy.deepcopy(pending_wait_operation)
        final_operation["velocity_transition_time"] = 0.0
        seam_hazard = _grouped_wait_hazard(
            pending_wait_anchor,
            zero_dynamic,
            pending_wait_index,
            len(segments) - 1,
            pending_wait_operation,
            final_operation,
            pending_wait_time,
            boundaries[-1] - pending_wait_time,
            boundaries[-1],
            incoming_dynamic.get("motion_kind", "wait"),
        )
        if seam_hazard is not None:
            hazards.append(seam_hazard)

    hazards.sort(key=lambda item: (item["start_time"], item["end_time"], item["kind"]))
    return {
        "segments": per_segment,
        "hazards": hazards,
        "frames": frames,
        "carried_ranges": carried_ranges,
        "boundaries": boundaries,
        "duration": offset,
        "valid": not hazards and all(item["valid"] for item in per_segment) and len(per_segment) == len(segments),
        "execution_status": (
            "executable"
            if not hazards and all(item["valid"] for item in per_segment) and len(per_segment) == len(segments)
            else "unknown" if any(item.get("confidence") == "unknown" for item in hazards)
            else "violated"
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
    reasons: dict[int, str] = {}
    for index, segment in enumerate(current):
        if index >= len(previous_segments):
            reasons[index] = "new segment without a previous conclusion"
            continue
        before = previous_segments[index]
        if before.get("segment_id") != segment["segment_id"]:
            reasons[index] = "segment was replaced"
            continue
        if before.get("digest") != segment["digest"]:
            reasons[index] = "operation version changed"
            continue
        if before.get("handover_in") != segment["handover_in"]:
            reasons[index] = "handover-in condition changed"

    for index in range(len(current)):
        if index in reasons:
            for downstream in range(index + 1, len(current)):
                reasons.setdefault(downstream, "upstream entering velocity, duration, or load changed")
            break
    return [{"segment": index, "reason": reasons[index]} for index in sorted(reasons)]


def _conclusion_sources(per_segment: list[dict], constraint_version: str) -> dict:
    return {
        "constraint_version": constraint_version,
        "segments": [
            {
                "segment_id": item["segment_id"],
                "digest": item["digest"],
                "handover_in": item["handover_in"],
                "handover_out": item["handover_out"],
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
            "execution_status": abnormal_chain["execution_status"],
            "duration": abnormal_chain["duration"],
            "segments": [
                {key: value for key, value in item.items() if key != "report"}
                for item in abnormal_chain["segments"]
            ],
            "hazards": abnormal_chain["hazards"],
            "first_hazard": abnormal_first,
            "frames": abnormal_chain["frames"],
            "carried_ranges": abnormal_chain["carried_ranges"],
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
        "execution_status": normal["execution_status"],
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
