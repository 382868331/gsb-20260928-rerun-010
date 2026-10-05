from __future__ import annotations

import copy

from .taskchain import (
    _run_chain,
    _segment_constraints,
    _segment_waits,
    validate_task,
)

SCHEME_SHORTEST = "shortest_completion"
SCHEME_FEWEST_STOPS = "fewest_new_stops"
SCHEMES = (SCHEME_SHORTEST, SCHEME_FEWEST_STOPS)


def _collect_dwell_pools(segments: list[dict], chain_segments: list[dict]) -> list[dict]:
    pools: list[dict] = []
    last_real: dict | None = None
    zero_motions: list[dict] = []
    pending_wait = 0.0
    for index, segment in enumerate(segments):
        wait_before, wait_after, wait = _segment_waits(segment)
        operation = segment.get("operation")
        if operation is None:
            pending_wait += wait
            continue
        pending_wait += wait_before
        info = chain_segments[index]
        if info.get("report") is None:
            raise ValueError(
                f"segment {index + 1} ('{info.get('name', index)}') cannot be validated, "
                f"so dwell requirements cannot be derived: {info.get('error')}"
            )
        dof = len(operation["link_lengths"])
        acceleration, transition_time = _segment_constraints(operation, dof)
        node = {
            "index": index,
            "entry_velocity": [float(value) for value in info["entry_velocity"]],
            "exit_velocity": [float(value) for value in info["exit_velocity"]],
            "acceleration": acceleration,
            "transition_time": transition_time,
        }
        duration = float(info["end_time"]) - float(info["start_time"])
        if duration > 0.0:
            if last_real is not None:
                pairs = list(zip([last_real, *zero_motions], [*zero_motions, node]))
                pools.append({
                    "pairs": pairs,
                    "existing_wait": pending_wait,
                    "left": last_real["index"],
                    "right": index,
                })
            last_real = node
            zero_motions = []
            pending_wait = 0.0
        elif last_real is not None:
            zero_motions.append(node)
        pending_wait += wait_after
    return pools


def _pair_dwell_requirement(left: dict, right: dict) -> tuple[float, bool]:
    required = 0.0
    no_wait_ok = True
    for joint in range(len(left["exit_velocity"])):
        exit_velocity = float(left["exit_velocity"][joint])
        entry_velocity = float(right["entry_velocity"][joint])
        accel_limit = min(float(left["acceleration"][joint]), float(right["acceleration"][joint]))
        exit_speed = abs(exit_velocity)
        entry_speed = abs(entry_velocity)
        if accel_limit > 0:
            required = max(required, (exit_speed + entry_speed) / accel_limit)
            allowed = min(float(left["transition_time"]), float(right["transition_time"]))
            velocity_change = abs(entry_velocity - exit_velocity)
            if velocity_change > accel_limit * allowed + 1e-9:
                no_wait_ok = False
        elif exit_speed > 0 or entry_speed > 0:
            required = float("inf")
            no_wait_ok = False
    return required, no_wait_ok


def _boundary_existing_wait(segments: list[dict], boundary: int) -> float:
    _, wait_after, wait = _segment_waits(segments[boundary])
    total = wait if segments[boundary].get("operation") is None else wait_after
    if boundary + 1 < len(segments):
        next_before, _, _ = _segment_waits(segments[boundary + 1])
        total += next_before
    return total


def _placement_field(segments: list[dict], boundary: int) -> str:
    return "wait" if segments[boundary].get("operation") is None else "wait_after"


def apply_placements(task: dict, placements: list[dict]) -> dict:
    revised = copy.deepcopy(task)
    for placement in placements:
        segment = revised["segments"][placement["segment_index"]]
        field = placement["field"]
        segment[field] = float(segment.get(field) or 0.0) + float(placement["added"])
    return revised


def _build_scheme(task: dict, pools: list[dict], baseline_duration: float, strategy: str) -> dict:
    segments = task["segments"]
    placements: list[dict] = []
    for pool in pools:
        deficit = pool["deficit"]
        if deficit is None or deficit <= 1e-9:
            continue
        usable = pool["usable_boundaries"]
        if strategy == SCHEME_FEWEST_STOPS:
            with_existing = [
                boundary for boundary in usable if _boundary_existing_wait(segments, boundary) > 1e-9
            ]
            boundary = with_existing[0] if with_existing else usable[0]
        else:
            boundary = usable[0]
        placements.append({
            "boundary": boundary,
            "segment_index": boundary,
            "field": _placement_field(segments, boundary),
            "added": deficit,
            "between_segment_ids": pool["between_segment_ids"],
            "between_names": pool["between_names"],
        })
    revised = apply_placements(task, placements)
    try:
        verification = validate_task(revised)
        validation = {"valid": bool(verification["valid"]), "duration": verification["duration"]}
    except ValueError as exc:
        validation = {"valid": False, "duration": None, "error": str(exc)}
    new_stops = sum(
        1 for placement in placements
        if _boundary_existing_wait(segments, placement["boundary"]) <= 1e-9
    )
    added_time = sum(placement["added"] for placement in placements)
    return {
        "placements": placements,
        "added_time": added_time,
        "new_duration": baseline_duration + added_time,
        "new_stops": new_stops,
        "stops": len(placements),
        "validation": validation,
    }


def build_wait_revision(task: dict, plan: dict, scheme: str) -> dict:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown wait allocation scheme {scheme}")
    if not plan["feasible"]:
        raise ValueError("no feasible wait allocation exists for the given constraints")
    revised = apply_placements(task, plan["schemes"][scheme]["placements"])
    revised["name"] = f"{task.get('name', 'task')}（等待分配修订）"
    return revised


def analyze_wait_allocation(
    task: dict,
    allowed_seams: list[int],
    locked_segments: list[int],
    max_duration: float | None,
) -> dict:
    segments = task.get("segments", [])
    if len(segments) < 2:
        raise ValueError("a chained task requires at least two segments")
    allowed = {int(value) for value in allowed_seams}
    locked = {int(value) for value in locked_segments}
    for boundary in allowed:
        if not 0 <= boundary < len(segments) - 1:
            raise ValueError(f"allowed seam {boundary} is out of range for {len(segments)} segments")
    for index in locked:
        if not 0 <= index < len(segments):
            raise ValueError(f"locked segment {index} is out of range for {len(segments)} segments")
    if max_duration is not None and max_duration <= 0:
        raise ValueError("max_duration must be positive")

    handover_tolerance = float(task.get("handover_tolerance", 1e-6))
    chain = _run_chain(segments, handover_tolerance)
    baseline_duration = float(chain["duration"])
    pools = []
    infeasible: list[dict] = []
    for pool in _collect_dwell_pools(segments, chain["segments"]):
        left, right = pool["left"], pool["right"]
        required = 0.0
        no_wait_ok = True
        for pair_left, pair_right in pool["pairs"]:
            pair_required, pair_no_wait_ok = _pair_dwell_requirement(pair_left, pair_right)
            required = max(required, pair_required)
            no_wait_ok = no_wait_ok and pair_no_wait_ok
        existing = pool["existing_wait"]
        if existing > 1e-9:
            deficit = max(0.0, required - existing)
        elif not no_wait_ok:
            deficit = required
        else:
            deficit = 0.0
        boundaries = list(range(left, right))
        usable = [boundary for boundary in boundaries if boundary in allowed and boundary not in locked]
        forbidden = [boundary for boundary in boundaries if boundary not in usable]
        entry = {
            "left_segment": left,
            "right_segment": right,
            "between_segments": [left, right],
            "between_segment_ids": [
                segments[left].get("segment_id", f"segment-{left + 1}"),
                segments[right].get("segment_id", f"segment-{right + 1}"),
            ],
            "between_names": [
                segments[left].get("name", f"segment-{left + 1}"),
                segments[right].get("name", f"segment-{right + 1}"),
            ],
            "existing_wait": existing,
            "required_dwell": required if required != float("inf") else None,
            "deficit": deficit if deficit != float("inf") else None,
            "no_wait_ok": no_wait_ok,
            "usable_boundaries": usable,
            "forbidden_boundaries": forbidden,
        }
        pools.append(entry)
        if deficit > 1e-9 and not usable:
            infeasible.append({
                "seam": {
                    "between_segments": entry["between_segments"],
                    "between_segment_ids": entry["between_segment_ids"],
                    "between_names": entry["between_names"],
                },
                "shortfall": None if deficit == float("inf") else deficit,
                "reason": (
                    "加速度上限为 0，无法在该接缝减速/再加速"
                    if deficit == float("inf")
                    else "该静止区间内没有允许停留且可修改的接缝"
                ),
            })

    total_added = sum(pool["deficit"] or 0.0 for pool in pools)
    if max_duration is not None and not infeasible:
        remaining = float(max_duration) - baseline_duration
        if total_added > remaining + 1e-9:
            cumulative = 0.0
            blocking = pools[-1] if pools else None
            for pool in pools:
                cumulative += pool["deficit"] or 0.0
                if cumulative > remaining + 1e-9:
                    blocking = pool
                    break
            infeasible.append({
                "seam": {
                    "between_segments": blocking["between_segments"],
                    "between_segment_ids": blocking["between_segment_ids"],
                    "between_names": blocking["between_names"],
                },
                "shortfall": baseline_duration + total_added - float(max_duration),
                "reason": (
                    f"任务完成上限 {float(max_duration):.3f}s 不足：当前 {baseline_duration:.3f}s，"
                    f"必需等待共 {total_added:.3f}s"
                ),
            })

    feasible = not infeasible
    schemes = {}
    if feasible:
        for strategy in SCHEMES:
            schemes[strategy] = _build_scheme(task, pools, baseline_duration, strategy)

    return {
        "feasible": feasible,
        "baseline_duration": baseline_duration,
        "max_duration": max_duration,
        "allowed_seams": sorted(allowed),
        "locked_segments": sorted(locked),
        "total_added_time": total_added,
        "pools": pools,
        "infeasible": infeasible,
        "schemes": schemes,
        "constraint_version": task.get("constraint_version", "1.1"),
    }
