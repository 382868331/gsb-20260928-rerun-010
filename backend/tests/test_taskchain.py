import pytest

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


def joint_step_task(wait_duration=None, include_zero=False):
    common = dict(
        speed_limits=[1, 1, 1],
        joint_limits=[[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]],
        link_lengths=[1, 1, 1],
        obstacles=[],
        carried_objects=[],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
    )
    first_waypoints = [[0, 0, 0], [0.4, 0, 0]]
    second_waypoints = [[0.4, 0, 0], [0.45, 0, 0]]
    first_limits = [[1, 1, 1]]
    second_limits = [[0.05, 1, 1]]
    if wait_duration is not None:
        wait = dict(common, waypoints=[[0.4, 0, 0], [0.4, 0, 0]], wait_duration=wait_duration,
                    segment_speed_limits=[[1, 1, 1]])
        middle = [dict(seg_type="wait", operation=wait)]
    elif include_zero:
        zero = dict(common, waypoints=[[0.4, 0, 0], [0.4, 0, 0]], segment_speed_limits=[[1, 1, 1]])
        middle = [dict(seg_type="zero", operation=zero)]
    else:
        middle = []
    segments = [
        {"segment_id": "fast", "name": "fast", "operation": dict(
            common, waypoints=first_waypoints, segment_speed_limits=first_limits,
        )},
        *middle,
        {"segment_id": "slow", "name": "slow", "operation": dict(
            common, waypoints=second_waypoints, segment_speed_limits=second_limits,
        )},
    ]
    return {"name": "joint-step", "segments": segments}


def test_cross_segment_velocity_step_is_rejected_with_time_and_constraint_details():
    result = validate_task(joint_step_task())

    assert result["valid"] is False
    assert result["execution_status"] == "violated"
    seam = next(item for item in result["hazards"] if item["kind"] == "velocity_transition")
    assert seam["details"]["from_segment"] == 0
    assert seam["details"]["to_segment"] == 1
    assert seam["details"]["joint"] == 0
    assert seam["details"]["from_joint_velocity"] == 1.0
    assert seam["details"]["to_joint_velocity"] == 0.05
    assert seam["details"]["allowed_transition_time"] == 0.2
    assert seam["details"]["required_transition_time"] == 0.95
    assert seam["start_time"] == 0.2
    assert seam["end_time"] == 0.4
    assert "segments 1 and 2" in seam["message"]


def test_zero_duration_or_gripper_boundary_does_not_create_transition_buffer():
    result = validate_task(joint_step_task(include_zero=True))

    seam = next(item for item in result["hazards"] if item["kind"] == "velocity_transition")
    assert seam["details"]["from_segment"] == 0
    assert seam["details"]["to_segment"] == 2
    assert seam["details"]["allowed_transition_time"] == 0.2


def test_real_wait_duration_completes_the_required_velocity_change():
    assert validate_task(joint_step_task(wait_duration=0.55))["valid"] is False
    result = validate_task(joint_step_task(wait_duration=0.65))

    assert result["valid"] is True
    assert result["duration"] == pytest.approx(2.05)
    assert result["segments"][1]["report"]["wait_duration"] == 0.65


def test_split_of_same_trajectory_has_same_transition_conclusion_as_one_operation():
    common = dict(
        speed_limits=[1, 1, 1],
        joint_limits=[[-3.14, 3.14]] * 3,
        link_lengths=[1, 1, 1],
        obstacles=[],
        carried_objects=[],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
    )
    split = validate_task({
        "name": "split",
        "segments": [
            {"segment_id": "a", "name": "a", "operation": dict(
                common, waypoints=[[0, 0, 0], [0.4, 0, 0]], segment_speed_limits=[[1, 1, 1]])},
            {"segment_id": "b", "name": "b", "operation": dict(
                common, waypoints=[[0.4, 0, 0], [0.45, 0, 0]], segment_speed_limits=[[0.05, 1, 1]])},
        ],
    })
    whole = validate_motion(
        [[0, 0, 0], [0.4, 0, 0], [0.45, 0, 0]],
        [1, 1, 1],
        [[-3.14, 3.14]] * 3,
        [1, 1, 1],
        [],
        segment_speed_limits=[[1, 1, 1], [0.05, 1, 1]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
    )

    assert split["valid"] is whole["valid"] is False
    assert split["hazards"][0]["kind"] == whole["hazard_intervals"][0]["kind"]


def test_positive_duration_turn_in_place_provides_real_buffer():
    task = joint_step_task(include_zero=True)
    task["segments"][1]["operation"]["waypoints"] = [[0.4, 0, 0], [0.4, 0, 0]]
    task["segments"][1]["operation"]["segment_speed_limits"] = [[1, 1, 1]]
    task["segments"][1]["operation"]["velocity_transition_time"] = 0.2
    assert validate_task(task)["valid"] is False

    task["segments"][1]["operation"]["waypoints"] = [[0.4, 0, 0], [1.4, 0, 0], [0.4, 0, 0]]
    task["segments"][1]["operation"]["segment_speed_limits"] = [[1, 1, 1], [1, 1, 1]]
    task["segments"][1]["operation"]["velocity_transition_time"] = 2.0
    turned = validate_task(task)

    assert turned["valid"] is True
    assert turned["segments"][1]["report"]["dynamic_state"]["at_start"]["motion_kind"] == "turn_in_place"
    assert turned["segments"][1]["report"]["duration"] >= 0.2


def test_dynamic_boundary_report_includes_translation_rotation_and_load():
    task = make_task()
    result = validate_task(task)

    boundary = result["segments"][1]["handover_in"]["dynamic"]
    assert boundary["carried_object_id"] == "block"
    assert len(boundary["translation_velocity"]) == 2
    assert isinstance(boundary["rotation_velocity"], float)
    assert boundary["joint_velocities"] == result["segments"][0]["report"]["segment_velocities"][-1]


def test_middle_speed_or_load_change_marks_downstream_for_recheck():
    original = validate_task(make_task())
    previous = original["conclusion_sources"]

    changed = make_task()
    changed["segments"][0]["operation"]["segment_speed_limits"] = [[0.4, 0.8, 0.8]]
    result = validate_task(changed, previous=previous)

    assert {item["segment"] for item in result["stale_segments"]} == {0, 1}
    assert result["segments"][1]["handover_in"]["dynamic"]["joint_velocities"] != \
        previous["segments"][1]["handover_in"]["dynamic"]["joint_velocities"]


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
