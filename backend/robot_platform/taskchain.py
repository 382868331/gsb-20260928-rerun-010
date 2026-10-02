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
    }


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

    for index, segment in enumerate(segments):
        operation = segment["operation"]
        constraint_version = operation.get("constraint_version", "1.1")
        digest = operation_digest(operation, initial, constraint_version)
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

        if report is None:
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
                "segment_id": segment.get("segment_id", f"segment-{index + 1}"),
                "name": segment.get("name", f"segment-{index + 1}"),
                "digest": digest,
                "start_time": offset,
                "end_time": offset,
                "handover_in": handover_in,
                "handover_out": None,
                "valid": False,
                "error": error,
                "report": None,
            })
            boundaries.append(offset)
            break

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
        per_segment.append({
            "segment_id": segment.get("segment_id", f"segment-{index + 1}"),
            "name": segment.get("name", f"segment-{index + 1}"),
            "digest": digest,
            "start_time": offset,
            "end_time": offset + duration,
            "handover_in": handover_in,
            "handover_out": handover_out,
            "valid": bool(report["valid"]),
            "error": None,
            "report": report,
        })
        offset += duration
        boundaries.append(offset)

    hazards.sort(key=lambda item: (item["start_time"], item["end_time"], item["kind"]))
    return {
        "segments": per_segment,
        "hazards": hazards,
        "frames": frames,
        "carried_ranges": carried_ranges,
        "boundaries": boundaries,
        "duration": offset,
        "valid": not hazards and all(item["valid"] for item in per_segment) and len(per_segment) == len(segments),
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
        if before.get("segment_id") != segment["segment_id"]:
            stale.append({"segment": index, "reason": "segment was replaced"})
            continue
        if before.get("digest") != segment["digest"]:
            stale.append({"segment": index, "reason": "operation version changed"})
            continue
        if before.get("handover_in") != segment["handover_in"]:
            stale.append({"segment": index, "reason": "handover-in condition changed"})
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
