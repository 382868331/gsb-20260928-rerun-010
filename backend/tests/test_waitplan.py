import pytest

from robot_platform import main
from robot_platform.schemas import TaskDefinition, WaitPlanRequest, WaitRevisionRequest
from robot_platform.taskchain import validate_task
from robot_platform.waitplan import analyze_wait_allocation, build_wait_revision


BASE = {
    "speed_limits": [0.8, 0.8, 0.8],
    "joint_limits": [[-2.8, 2.8], [-2.4, 2.4], [-2.4, 2.4]],
    "link_lengths": [1.4, 1.0, 0.7],
    "obstacles": [],
    "carried_objects": [],
}


def _op(waypoints, speed_limits):
    return dict(
        BASE,
        waypoints=waypoints,
        speed_limits=speed_limits,
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        pick_events=[],
        place_events=[],
    )


def _two_motion_task():
    return {
        "name": "two-motions",
        "segments": [
            {"segment_id": "move-a", "name": "move-a", "operation": _op([[0, 0, 0], [0.42, 0, 0]], [0.6, 1, 1])},
            {"segment_id": "move-b", "name": "move-b", "operation": _op([[0.42, 0, 0], [0.42, 0.42, 0]], [1, 0.6, 1])},
        ],
    }


def test_plan_derives_brake_and_restart_requirement_from_seam():
    plan = analyze_wait_allocation(_two_motion_task(), allowed_seams=[0], locked_segments=[], max_duration=None)

    assert plan["feasible"] is True
    assert plan["baseline_duration"] == pytest.approx(1.4)
    pool = plan["pools"][0]
    assert pool["between_segment_ids"] == ["move-a", "move-b"]
    assert pool["required_dwell"] == pytest.approx(0.6)
    assert pool["deficit"] == pytest.approx(0.6)
    assert pool["usable_boundaries"] == [0]
    shortest = plan["schemes"]["shortest_completion"]
    fewest = plan["schemes"]["fewest_new_stops"]
    assert shortest["added_time"] == pytest.approx(0.6)
    assert shortest["new_duration"] == pytest.approx(2.0)
    placement = shortest["placements"][0]
    assert placement["boundary"] == 0
    assert placement["field"] == "wait_after"
    assert placement["added"] == pytest.approx(0.6)
    assert shortest["validation"]["valid"] is True
    assert shortest["validation"]["duration"] == pytest.approx(2.0)
    assert fewest["new_stops"] == 1
    assert fewest["validation"]["valid"] is True


def test_revision_validation_matches_continuous_wait_conclusion():
    task = _two_motion_task()
    plan = analyze_wait_allocation(task, allowed_seams=[0], locked_segments=[], max_duration=None)
    revised = build_wait_revision(task, plan, "shortest_completion")

    result = validate_task(revised)
    assert result["valid"] is True
    assert result["seam_checks"][0]["wait_time"] == pytest.approx(0.6)
    assert result["duration"] == pytest.approx(2.0)

    continuous = _two_motion_task()
    continuous["segments"][0]["wait_after"] = 0.6
    assert validate_task(continuous)["valid"] is True


def test_forbidden_seam_cannot_fill_gap_and_reports_shortfall():
    plan = analyze_wait_allocation(_two_motion_task(), allowed_seams=[], locked_segments=[], max_duration=None)

    assert plan["feasible"] is False
    assert plan["schemes"] == {}
    entry = plan["infeasible"][0]
    assert entry["seam"]["between_segment_ids"] == ["move-a", "move-b"]
    assert entry["shortfall"] == pytest.approx(0.6)


def test_locked_segment_blocks_its_boundary():
    plan = analyze_wait_allocation(_two_motion_task(), allowed_seams=[0], locked_segments=[0], max_duration=None)

    assert plan["feasible"] is False
    assert plan["pools"][0]["usable_boundaries"] == []
    assert plan["pools"][0]["forbidden_boundaries"] == [0]
    assert plan["infeasible"][0]["shortfall"] == pytest.approx(0.6)


def test_duration_cap_reports_blocking_seam_and_shortfall_time():
    plan = analyze_wait_allocation(_two_motion_task(), allowed_seams=[0], locked_segments=[], max_duration=1.8)

    assert plan["feasible"] is False
    entry = plan["infeasible"][0]
    assert entry["seam"]["between_segment_ids"] == ["move-a", "move-b"]
    assert entry["shortfall"] == pytest.approx(0.2)

    fits = analyze_wait_allocation(_two_motion_task(), allowed_seams=[0], locked_segments=[], max_duration=2.0)
    assert fits["feasible"] is True


def test_zero_length_motion_adds_no_margin_and_split_waits_pool():
    task = _two_motion_task()
    task["segments"].insert(1, {
        "segment_id": "zero",
        "name": "zero-length",
        "operation": _op([[0.42, 0, 0], [0.42, 0, 0]], [1, 1, 1]),
    })
    plan = analyze_wait_allocation(task, allowed_seams=[0, 1], locked_segments=[], max_duration=None)

    assert plan["feasible"] is True
    assert len(plan["pools"]) == 1
    pool = plan["pools"][0]
    assert pool["existing_wait"] == pytest.approx(0.0)
    assert pool["required_dwell"] == pytest.approx(0.6)
    assert pool["deficit"] == pytest.approx(0.6)

    task["segments"][0]["wait_after"] = 0.25
    task["segments"][1]["wait_after"] = 0.15
    plan = analyze_wait_allocation(task, allowed_seams=[0, 1], locked_segments=[], max_duration=None)
    pool = plan["pools"][0]
    assert pool["existing_wait"] == pytest.approx(0.4)
    assert pool["deficit"] == pytest.approx(0.2)
    fewest = plan["schemes"]["fewest_new_stops"]
    assert fewest["new_stops"] == 0
    assert fewest["placements"][0]["added"] == pytest.approx(0.2)
    assert fewest["validation"]["valid"] is True


def test_fewest_stops_prefers_existing_dwell_boundary():
    task = _two_motion_task()
    task["segments"].insert(1, {
        "segment_id": "zero",
        "name": "zero-length",
        "operation": _op([[0.42, 0, 0], [0.42, 0, 0]], [1, 1, 1]),
        "wait_after": 0.1,
    })
    plan = analyze_wait_allocation(task, allowed_seams=[0, 1], locked_segments=[], max_duration=None)

    fewest = plan["schemes"]["fewest_new_stops"]
    assert fewest["placements"][0]["boundary"] == 1
    assert fewest["placements"][0]["added"] == pytest.approx(0.5)
    assert fewest["new_stops"] == 0
    shortest = plan["schemes"]["shortest_completion"]
    assert shortest["placements"][0]["boundary"] == 0
    assert shortest["new_stops"] == 1
    assert shortest["added_time"] == pytest.approx(fewest["added_time"])


def test_wait_plan_endpoint_and_revision_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "task_store", main.TaskStore(tmp_path))
    task = TaskDefinition.model_validate(_two_motion_task())
    saved = main.save_task(task)
    assert saved.executable is False

    plan = main.plan_wait_allocation(WaitPlanRequest(task=task, allowed_seams=[0], max_duration=3.0))
    assert plan["feasible"] is True

    response = main.create_wait_revision(
        saved.id,
        WaitRevisionRequest(allowed_seams=[0], max_duration=3.0, scheme="fewest_new_stops"),
    )
    revision = response["revision"]
    assert revision["id"] != saved.id
    assert revision["executable"] is True
    assert revision["segments"][0]["wait_after"] == pytest.approx(0.6)
    assert revision["last_task_validation"]["valid"] is True
    assert revision["last_task_validation"]["duration"] == pytest.approx(2.0)
    assert revision["last_task_validation"]["constraint_version"] == task.constraint_version

    original = main.task_store.get(saved.id)
    assert original.segments[0].wait_after == 0.0
    assert original.last_task_validation["duration"] == pytest.approx(1.4)


def test_revision_rejected_when_infeasible(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "task_store", main.TaskStore(tmp_path))
    task = TaskDefinition.model_validate(_two_motion_task())
    saved = main.save_task(task)

    with pytest.raises(main.HTTPException) as excinfo:
        main.create_wait_revision(saved.id, WaitRevisionRequest(allowed_seams=[], max_duration=None))
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert detail["infeasible"][0]["seam"]["between_segment_ids"] == ["move-a", "move-b"]
    assert detail["infeasible"][0]["shortfall"] == pytest.approx(0.6)
    assert len(main.task_store.list()) == 1


def test_smooth_seam_needs_no_allocation():
    task = {
        "name": "smooth",
        "segments": [
            {"segment_id": "a", "name": "a", "operation": _op([[0, 0, 0], [0.4, 0, 0]], [1, 1, 1])},
            {"segment_id": "b", "name": "b", "operation": _op([[0.4, 0, 0], [0.8, 0, 0]], [1, 1, 1])},
        ],
    }
    plan = analyze_wait_allocation(task, allowed_seams=[], locked_segments=[], max_duration=None)

    assert plan["feasible"] is True
    assert plan["pools"][0]["deficit"] == pytest.approx(0.0)
    assert plan["schemes"]["shortest_completion"]["placements"] == []
    assert plan["schemes"]["shortest_completion"]["added_time"] == pytest.approx(0.0)
