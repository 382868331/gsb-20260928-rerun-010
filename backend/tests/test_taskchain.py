from robot_platform import main
from robot_platform.robotics import validate_motion
from robot_platform.schemas import TaskDefinition, TaskValidationRequest
from robot_platform.taskchain import validate_task


BASE = {
    "speed_limits": [0.8, 0.8, 0.8],
    "joint_limits": [[-2.8, 2.8], [-2.4, 2.4], [-2.4, 2.4]],
    "link_lengths": [1.4, 1.0, 0.7],
    "obstacles": [],
    "carried_objects": [{"id": "block", "name": "block", "radius": 0.08, "offset_x": 0.12, "offset_y": 0.0}],
}


def make_task(place_waypoint=1, seg2_start=None, obstacle=None):
    seg1 = dict(
        BASE,
        waypoints=[[-0.65, 0, 0], [-0.2, 0.05, 0]],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[],
    )
    seg2 = dict(
        BASE,
        waypoints=[seg2_start or [-0.2, 0.05, 0], [0.45, 0.08, 0]],
        pick_events=[],
        place_events=[{"waypoint_index": place_waypoint, "object_id": "block"}],
    )
    if obstacle:
        seg2["obstacles"] = [obstacle]
    return {
        "name": "chained",
        "segments": [
            {"segment_id": "s1", "name": "pick", "operation": seg1},
            {"segment_id": "s2", "name": "place", "operation": seg2},
        ],
    }


def test_chain_of_two_safe_segments_is_valid_and_carries_state_over():
    result = validate_task(make_task())

    assert result["valid"] is True
    assert result["segments"][0]["handover_out"]["carried_object_id"] == "block"
    assert result["segments"][1]["handover_in"]["carried_object_id"] == "block"
    assert result["segments"][1]["handover_in"]["joint_angles"] == [-0.2, 0.05, 0]
    assert result["duration"] > 0
    assert {frame["segment"] for frame in result["frames"]} == {0, 1}
    carried = result["carried_ranges"]
    assert len(carried) == 1
    assert carried[0]["object_id"] == "block"
    assert carried[0]["segment_index"] == 1
    assert carried[0]["start_time"] == result["segments"][0]["end_time"]


def test_single_segment_report_is_preserved_unchanged():
    task = make_task()
    result = validate_task(task)

    standalone = validate_motion(
        task["segments"][0]["operation"]["waypoints"],
        BASE["speed_limits"],
        [tuple(pair) for pair in BASE["joint_limits"]],
        BASE["link_lengths"],
        [],
        carried_objects=BASE["carried_objects"],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[],
    )
    report = result["segments"][0]["report"]
    assert report["valid"] == standalone["valid"]
    assert report["duration"] == standalone["duration"]
    assert report["hazard_intervals"] == standalone["hazard_intervals"]
    assert report["frames"] == standalone["frames"]


def test_handover_pose_mismatch_is_rejected_at_boundary():
    result = validate_task(make_task(seg2_start=[-0.1, 0.05, 0]))

    assert result["valid"] is False
    first = result["first_hazard"]
    assert first["kind"] == "handover_mismatch"
    assert first["segment"] == 1
    assert first["start_time"] == result["segments"][0]["end_time"]
    assert result["replay_window"]["crosses_boundary"] is True


def test_handover_state_error_when_object_missing_in_next_segment():
    task = make_task()
    task["segments"][1]["operation"]["carried_objects"] = []
    task["segments"][1]["operation"]["place_events"] = []

    result = validate_task(task)

    assert result["valid"] is False
    assert result["first_hazard"]["kind"] == "handover_state"
    assert result["segments"][1]["report"] is None


def test_hazard_in_second_segment_uses_full_task_time():
    obstacle = {"name": "late", "x": 2.16, "y": 2.46, "radius": 0.05}
    task = make_task(obstacle=obstacle)
    task["segments"][1]["operation"]["waypoints"] = [[-0.2, 0.05, 0], [0.75, 0, 0], [1.0, 0, 0]]
    task["segments"][1]["operation"]["place_events"] = [{"waypoint_index": 2, "object_id": "block"}]

    result = validate_task(task)

    assert result["valid"] is False
    first = result["first_hazard"]
    assert first["segment"] == 1
    assert first["start_time"] >= result["segments"][0]["end_time"]
    assert result["replay_window"]["start_time"] < first["start_time"]


def test_replacing_segment_marks_downstream_conclusions_stale():
    original = validate_task(make_task())
    previous = original["conclusion_sources"]

    changed = make_task()
    changed["segments"][0]["operation"]["pick_events"] = []
    result = validate_task(changed, previous=previous)

    stale_segments = {item["segment"] for item in result["stale_segments"]}
    assert 0 in stale_segments
    assert 1 in stale_segments
    assert result["recheck_required"] == sorted(stale_segments)

    unchanged = validate_task(make_task(), previous=previous)
    assert unchanged["stale_segments"] == []


def test_pick_failure_keeps_normal_and_abnormal_plans_separate():
    result = validate_task(make_task(), failures=[{"segment_index": 0, "type": "pick_failed", "object_id": "block"}])

    assert result["valid"] is True
    assert result["carried_ranges"][0]["object_id"] == "block"
    abnormal = result["abnormal"]
    assert abnormal is not None
    assert abnormal["valid"] is True
    assert abnormal["carried_ranges"] == []
    assert result["unknown_intervals"]
    unknown = result["unknown_intervals"][0]
    assert unknown["confidence"] == "unknown"
    assert unknown["end_time"] == result["duration"]
    assert result["comparison"]["first_concern_source"] == "unknown"


def test_early_release_shortens_actual_carried_range():
    result = validate_task(
        make_task(),
        failures=[{"segment_index": 1, "type": "early_release", "object_id": "block", "waypoint_index": 0}],
    )

    abnormal = result["abnormal"]
    assert abnormal is not None
    normal_range = result["carried_ranges"][0]
    assert abnormal["carried_ranges"] == [] or all(
        item["end_time"] <= normal_range["start_time"] + 1e-9 for item in abnormal["carried_ranges"]
    )
    assert result["unknown_intervals"][0]["start_time"] <= normal_range["end_time"]


def test_task_api_roundtrip_keeps_conclusion_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "task_store", main.TaskStore(tmp_path))
    task = TaskDefinition.model_validate(make_task())

    saved = main.save_task(task)
    assert saved.executable is True
    sources = saved.last_task_validation["conclusion_sources"]
    assert len(sources["segments"]) == 2
    assert sources["segments"][1]["handover_in"]["carried_object_id"] == "block"

    loaded = main.get_task(saved.id)
    assert loaded.last_task_validation["conclusion_sources"] == sources
    assert loaded.last_task_validation["stale_segments"] == []

    request = TaskValidationRequest(task=task, previous=sources)
    revalidated = main.validate_chained_task(request)
    assert revalidated["valid"] is True
    assert revalidated["stale_segments"] == []


def _speed_step_task():
    fast = dict(
        BASE,
        waypoints=[[0, 0, 0], [0.4, 0, 0]],
        segment_speed_limits=[[1, 1, 1]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        speed_limits=[1, 1, 1],
        pick_events=[],
        place_events=[],
    )
    slow = dict(
        BASE,
        waypoints=[[0.4, 0, 0], [0.45, 0, 0]],
        segment_speed_limits=[[0.05, 1, 1]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        speed_limits=[1, 1, 1],
        pick_events=[],
        place_events=[],
    )
    return {
        "name": "speed-step",
        "segments": [
            {"segment_id": "fast", "name": "fast", "operation": fast},
            {"segment_id": "slow", "name": "slow", "operation": slow},
        ],
    }


def test_cross_segment_speed_step_is_rejected_at_seam():
    result = validate_task(_speed_step_task())

    assert result["valid"] is False
    seam_hazards = [item for item in result["hazards"] if item["kind"] == "velocity_transition"]
    assert len(seam_hazards) == 1
    hazard = seam_hazards[0]
    assert hazard["segment"] == 1
    assert hazard["start_time"] == result["boundaries"][0] == 0.4
    detail = hazard["detail"]
    assert detail["joint"] == 1
    assert detail["from_velocity"] == 1.0
    assert detail["to_velocity"] == 0.05
    assert 0.94 < detail["required_transition_time"] < 0.96
    assert detail["allowed_transition_time"] == 0.2
    assert detail["acceleration_limit"] == 1.0
    assert abs(detail["transition_window_start"] - 0.2) < 1e-9
    assert abs(detail["transition_window_end"] - 0.6) < 1e-9
    assert "fast" in hazard["message"] and "slow" in hazard["message"]
    seam = result["seam_checks"][0]
    assert seam["status"] == "violated"
    assert seam["between_segment_ids"] == ["fast", "slow"]


def test_smooth_seam_is_explicitly_proven():
    result = validate_task(make_task())

    assert result["valid"] is True
    assert result["unknown_segments"] == []
    seam = result["seam_checks"][0]
    assert seam["status"] == "proven"
    assert seam["carried_object_id"] == "block"
    assert all(item["ok"] for item in seam["joints"])


def test_zero_length_action_at_boundary_provides_no_transition_time():
    task = _speed_step_task()
    task["segments"].insert(1, {
        "segment_id": "pause",
        "name": "zero-length pause",
        "operation": dict(
            BASE,
            waypoints=[[0.4, 0, 0], [0.4, 0, 0]],
            acceleration_limits=[1, 1, 1],
            velocity_transition_time=0.2,
            speed_limits=[1, 1, 1],
            pick_events=[],
            place_events=[],
        ),
    })
    result = validate_task(task)

    assert result["valid"] is False
    violated = [seam for seam in result["seam_checks"] if seam["status"] == "violated"]
    assert any(seam["between_segment_ids"] == ["fast", "pause"] for seam in violated)
    assert any(seam["between_segment_ids"] == ["pause", "slow"] and seam["status"] == "proven" for seam in result["seam_checks"])


def test_splitting_one_trajectory_does_not_add_transition_budget():
    operation = dict(
        BASE,
        waypoints=[[0, 0, 0], [0.4, 0, 0], [0.45, 0, 0]],
        segment_speed_limits=[[1, 1, 1], [0.05, 1, 1]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        speed_limits=[1, 1, 1],
        pick_events=[],
        place_events=[],
    )
    whole = validate_motion(
        operation["waypoints"],
        operation["speed_limits"],
        [tuple(pair) for pair in BASE["joint_limits"]],
        BASE["link_lengths"],
        [],
        segment_speed_limits=operation["segment_speed_limits"],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
    )
    assert whole["valid"] is False

    split = {
        "name": "split",
        "segments": [
            {"segment_id": "a", "name": "a", "operation": {
                **operation,
                "waypoints": [[0, 0, 0], [0.4, 0, 0]],
                "segment_speed_limits": [[1, 1, 1]],
            }},
            {"segment_id": "b", "name": "b", "operation": {
                **operation,
                "waypoints": [[0.4, 0, 0], [0.45, 0, 0]],
                "segment_speed_limits": [[0.05, 1, 1]],
            }},
        ],
    }
    result = validate_task(split)
    assert result["valid"] is False
    assert result["seam_checks"][0]["status"] == "violated"


def test_continuous_in_place_rotation_seam_is_checked():
    first = dict(
        BASE,
        waypoints=[[0, 0, 0], [0.4, 0, 0]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        speed_limits=[1, 1, 1],
        pick_events=[],
        place_events=[],
    )
    rotate = dict(
        BASE,
        waypoints=[[0.4, 0, 0], [0.4, 0.3, 0]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
        speed_limits=[1, 1, 1],
        pick_events=[],
        place_events=[],
    )
    result = validate_task({
        "name": "rotation",
        "segments": [
            {"segment_id": "move", "name": "move", "operation": first},
            {"segment_id": "turn", "name": "turn", "operation": rotate},
        ],
    })

    seam = result["seam_checks"][0]
    assert seam["status"] == "violated"
    joint_statuses = {item["joint"]: item["ok"] for item in seam["joints"]}
    assert joint_statuses[1] is False
    assert joint_statuses[2] is False


def test_missing_required_state_marks_later_segments_unknown():
    task = make_task()
    task["segments"][1]["operation"]["carried_objects"] = []
    task["segments"][1]["operation"]["place_events"] = []
    result = validate_task(task)

    assert result["valid"] is False
    assert result["unknown_segments"] == [1]
    assert result["segments"][1]["status"] == "unknown"
    assert result["segments"][0]["status"] == "proven"
    assert result["seam_checks"][0]["status"] == "unknown"


def test_speed_change_in_middle_segment_invalidates_downstream_seam_conclusion():
    original = validate_task(_speed_step_task())
    sources = original["conclusion_sources"]
    assert sources["segments"][0]["exit_velocity"] == [1.0, 0.0, 0.0]
    assert sources["segments"][1]["entry_velocity"] == [0.05, 0.0, 0.0]

    slower_first = _speed_step_task()
    slower_first["segments"][0]["operation"]["segment_speed_limits"] = [[0.25, 1, 1]]
    result = validate_task(slower_first, previous=sources)
    assert {item["segment"] for item in result["stale_segments"]} == {0, 1}
    assert result["valid"] is True
    assert result["seam_checks"][0]["status"] == "proven"
