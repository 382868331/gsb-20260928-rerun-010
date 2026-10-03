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


SINGLE_AXIS = dict(
    BASE,
    speed_limits=[0.8, 0.8, 0.8],
    acceleration_limits=[2.0, 2.0, 2.0],
    velocity_transition_time=0.2,
    carried_objects=[],
    pick_events=[],
    place_events=[],
)


def _single_axis_operation(waypoints, **overrides):
    operation = dict(SINGLE_AXIS, waypoints=waypoints)
    operation.update(overrides)
    return operation


def _reversal_task(wait_after=0.0, wait_before=0.0, middle_wait=None):
    first = _single_axis_operation([[0, 0, 0], [0.56, 0, 0]], wait_after=wait_after)
    second = _single_axis_operation([[0.56, 0, 0], [0, 0, 0]], wait_before=wait_before)
    segments = [
        {"segment_id": "out", "name": "out", "operation": first},
    ]
    if middle_wait is not None:
        segments.append({"segment_id": "dwell", "name": "dwell", "operation": None, "wait": middle_wait})
    segments.append({"segment_id": "back", "name": "back", "operation": second})
    return {"name": "reversal", "segments": segments}


def test_real_wait_counts_toward_duration_and_seam_transition():
    no_wait = validate_task(_reversal_task())
    assert no_wait["valid"] is False
    assert abs(no_wait["duration"] - 1.4) < 1e-9
    seam = no_wait["seam_checks"][0]
    assert seam["status"] == "violated"
    assert seam["wait_duration"] == 0.0
    joint = seam["joints"][0]
    assert abs(joint["required_transition_time"] - 0.8) < 1e-9
    assert abs(joint["allowed_transition_time"] - 0.2) < 1e-9

    enough = validate_task(_reversal_task(wait_after=0.71))
    assert abs(enough["duration"] - 2.11) < 1e-9
    assert enough["valid"] is True
    seam = enough["seam_checks"][0]
    assert seam["status"] == "proven"
    assert abs(seam["wait_duration"] - 0.71) < 1e-9
    assert abs(seam["joints"][0]["allowed_transition_time"] - 0.91) < 1e-9
    assert seam["between_segment_ids"] == ["out", "back"]
    wait_frames = [
        frame for frame in enough["frames"] if 0.7 < frame["time"] < 1.41
    ]
    assert wait_frames
    assert all(frame["joint_angles"] == [0.56, 0, 0] for frame in wait_frames)

    short = validate_task(_reversal_task(wait_after=0.51))
    assert abs(short["duration"] - 1.91) < 1e-9
    assert short["valid"] is False
    seam = short["seam_checks"][0]
    assert seam["status"] == "violated"
    assert abs(seam["joints"][0]["allowed_transition_time"] - 0.71) < 1e-9
    hazard = [item for item in short["hazards"] if item["kind"] == "velocity_transition"][0]
    assert abs(hazard["start_time"] - 0.7) < 1e-9
    assert abs(hazard["end_time"] - 1.21) < 1e-9
    assert hazard["detail"]["wait_duration"] == 0.51


def test_wait_position_variants_give_same_conclusion():
    at_end = validate_task(_reversal_task(wait_after=0.71))
    at_start = validate_task(_reversal_task(wait_before=0.71))
    middle = validate_task(_reversal_task(middle_wait=0.71))

    for result in (at_end, at_start, middle):
        assert result["valid"] is True
        assert abs(result["duration"] - 2.11) < 1e-9
        assert result["seam_checks"][0]["status"] == "proven"
        assert abs(result["seam_checks"][0]["wait_duration"] - 0.71) < 1e-9

    middle_segment = middle["segments"][1]
    assert middle_segment["kind"] == "wait"
    assert abs(middle_segment["end_time"] - middle_segment["start_time"] - 0.71) < 1e-9
    assert middle["segments"][2]["start_time"] == middle_segment["end_time"]


def test_split_wait_does_not_create_extra_transition_margin():
    split_enough = _reversal_task(wait_after=0.36, middle_wait=0.35)
    result = validate_task(split_enough)
    assert result["valid"] is True
    assert abs(result["duration"] - 2.11) < 1e-9
    assert abs(result["seam_checks"][0]["wait_duration"] - 0.71) < 1e-9

    split_short = _reversal_task(wait_after=0.26, middle_wait=0.25)
    result = validate_task(split_short)
    assert result["valid"] is False
    assert abs(result["seam_checks"][0]["wait_duration"] - 0.51) < 1e-9
    assert result["seam_checks"][0]["status"] == "violated"


def test_zero_length_action_does_not_relay_wait_credit():
    task = _reversal_task(wait_after=0.71)
    task["segments"].insert(1, {
        "segment_id": "noop",
        "name": "zero-length",
        "operation": _single_axis_operation([[0.56, 0, 0], [0.56, 0, 0]]),
    })
    result = validate_task(task)

    assert result["valid"] is False
    first_seam = result["seam_checks"][0]
    assert first_seam["between_segment_ids"] == ["out", "noop"]
    assert first_seam["status"] == "proven"
    second_seam = result["seam_checks"][1]
    assert second_seam["between_segment_ids"] == ["noop", "back"]
    assert second_seam["status"] == "violated"
    assert second_seam["wait_duration"] == 0.0


def test_wait_preserves_carry_state_and_extends_carried_range():
    first = dict(
        BASE,
        waypoints=[[-0.65, 0, 0], [-0.2, 0.05, 0], [0.1, 0, 0]],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[],
        wait_after=0.6,
    )
    second = dict(
        BASE,
        waypoints=[[0.1, 0, 0], [0.45, 0.08, 0]],
        pick_events=[],
        place_events=[{"waypoint_index": 1, "object_id": "block"}],
    )
    task = {
        "name": "carry-wait",
        "segments": [
            {"segment_id": "s1", "name": "pick", "operation": first},
            {"segment_id": "s2", "name": "place", "operation": second},
        ],
    }
    result = validate_task(task)

    assert result["valid"] is True
    wait_interval = result["wait_intervals"][0]
    assert abs(wait_interval["end_time"] - wait_interval["start_time"] - 0.6) < 1e-9
    carried = result["carried_ranges"]
    assert len(carried) == 1
    assert carried[0]["start_time"] < wait_interval["start_time"]
    assert carried[0]["end_time"] > wait_interval["end_time"]
    wait_frames = [
        frame
        for frame in result["frames"]
        if wait_interval["start_time"] < frame["time"] < wait_interval["end_time"]
    ]
    assert wait_frames
    assert all(frame["carried_object_id"] == "block" for frame in wait_frames)


def test_wait_segment_without_preceding_motion_is_unknown():
    task = _reversal_task()
    task["segments"].insert(0, {"segment_id": "dwell", "name": "dwell", "operation": None, "wait": 0.5})
    result = validate_task(task)

    assert result["valid"] is False
    assert result["unknown_segments"] == [0]
    assert result["segments"][0]["status"] == "unknown"
    assert result["first_hazard"]["kind"] == "handover_state"


def test_changing_wait_marks_segment_and_downstream_stale():
    original = validate_task(_reversal_task(wait_after=0.71))
    assert original["valid"] is True
    sources = original["conclusion_sources"]

    shortened = validate_task(_reversal_task(wait_after=0.51), previous=sources)
    assert shortened["valid"] is False
    assert {item["segment"] for item in shortened["stale_segments"]} == {0, 1}

    unchanged = validate_task(_reversal_task(wait_after=0.71), previous=sources)
    assert unchanged["stale_segments"] == []
    assert unchanged["valid"] is True


def test_split_chain_and_whole_motion_with_wait_agree():
    whole_operation = _single_axis_operation(
        [[0, 0, 0], [0.56, 0, 0], [0, 0, 0]],
        segment_speed_limits=[[0.8, 0.8, 0.8], [0.8, 0.8, 0.8]],
    )

    def run_whole(wait):
        return validate_motion(
            whole_operation["waypoints"],
            whole_operation["speed_limits"],
            [tuple(pair) for pair in BASE["joint_limits"]],
            BASE["link_lengths"],
            [],
            segment_speed_limits=whole_operation["segment_speed_limits"],
            acceleration_limits=[2.0, 2.0, 2.0],
            velocity_transition_time=0.2,
            wait_events=[{"waypoint_index": 1, "duration": wait}],
        )

    whole_enough = run_whole(0.71)
    assert whole_enough["valid"] is True
    assert abs(whole_enough["duration"] - 2.11) < 1e-9
    whole_short = run_whole(0.51)
    assert whole_short["valid"] is False
    assert whole_short["hazard_intervals"][0]["kind"] == "velocity_transition"

    chain_enough = validate_task(_reversal_task(middle_wait=0.71))
    chain_short = validate_task(_reversal_task(middle_wait=0.51))
    assert chain_enough["valid"] == whole_enough["valid"] is True
    assert chain_short["valid"] == whole_short["valid"] is False
    assert abs(chain_enough["duration"] - whole_enough["duration"]) < 1e-9


def test_waited_task_survives_save_and_reload_revalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "task_store", main.TaskStore(tmp_path))
    task = TaskDefinition.model_validate(_reversal_task(middle_wait=0.71))

    saved = main.save_task(task)
    assert saved.executable is True
    assert abs(saved.last_task_validation["duration"] - 2.11) < 1e-9

    loaded = main.get_task(saved.id)
    assert loaded.executable is True
    revalidated = loaded.last_task_validation
    assert abs(revalidated["duration"] - 2.11) < 1e-9
    assert revalidated["seam_checks"][0]["status"] == "proven"
    assert revalidated["stale_segments"] == []

    short_task = TaskDefinition.model_validate(_reversal_task(middle_wait=0.51))
    saved_short = main.save_task(short_task)
    assert saved_short.executable is False
    assert saved_short.last_task_validation["seam_checks"][0]["status"] == "violated"
