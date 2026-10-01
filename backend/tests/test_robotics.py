from robot_platform.robotics import interpolate_trajectory, validate_motion


LINK_LENGTHS = [1.4, 1.0, 0.7]
JOINT_LIMITS = [[-2.8, 2.8], [-2.4, 2.4], [-2.4, 2.4]]
WAYPOINTS = [
    [-0.65, 0.0, 0.0],
    [0.65, 0.0, 0.0],
]
OBSTACLE = {"name": "middle-sweep", "x": 1.75, "y": 0.12, "radius": 0.1}


def test_intermediate_obstacle_collision_is_rejected_while_endpoints_are_safe():
    result = validate_motion(
        WAYPOINTS,
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [OBSTACLE],
        link_radius=0.08,
        sample_dt=0.01,
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "obstacle"
    assert 0.25 < result["first_collision"]["time"] < 1.35
    assert all(
        collision["time"] not in (result["frames"][0]["time"], result["frames"][-1]["time"])
        for collision in result["collisions"]
    )

    start_result = validate_motion(
        [WAYPOINTS[0], WAYPOINTS[0]],
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [OBSTACLE],
    )
    end_result = validate_motion(
        [WAYPOINTS[1], WAYPOINTS[1]],
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [OBSTACLE],
    )
    assert start_result["valid"] is True
    assert end_result["valid"] is True


def test_display_step_size_does_not_change_swept_collision_conclusion():
    fine = validate_motion(
        WAYPOINTS,
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [OBSTACLE],
        link_radius=0.08,
        sample_dt=0.01,
    )
    coarse = validate_motion(
        WAYPOINTS,
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [OBSTACLE],
        link_radius=0.08,
        sample_dt=2.0,
    )

    assert fine["valid"] is False
    assert coarse["valid"] is False
    assert coarse["first_collision"]["kind"] == fine["first_collision"]["kind"]
    assert coarse["replay_window"]["start_time"] < coarse["first_collision"]["time"]
    assert coarse["replay_window"]["end_time"] > coarse["first_collision"]["end_time"]
    assert coarse["display_sample_dt"] == 2.0
    assert coarse["collision_check_dt"] <= 0.01


def test_speed_limit_changes_motion_time():
    fast_times, _, _ = interpolate_trajectory(WAYPOINTS, [1.0, 1.0, 1.0], sample_dt=0.01)
    slow_times, _, _ = interpolate_trajectory(WAYPOINTS, [0.5, 1.0, 1.0], sample_dt=0.01)

    assert abs(fast_times[-1] - 1.3) < 0.015
    assert abs(slow_times[-1] - 2.6) < 0.015


def test_joint_limit_is_checked_along_motion():
    result = validate_motion(
        [[0.0, 0.0, 0.0], [3.2, 0.0, 0.0]],
        [1.0, 1.0, 1.0],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "joint_limit"


def test_carried_object_collision_is_checked_after_pick():
    carried = {"id": "block", "name": "block", "radius": 0.08, "offset_x": 0.18, "offset_y": 0}
    result = validate_motion(
        [[0.4, 0, 0], [0.75, 0, 0], [1.0, 0, 0]],
        [0.5, 0.5, 0.5],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [{"name": "object-path", "x": 2.16, "y": 2.46, "radius": 0.05}],
        segment_speed_limits=[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]],
        carried_objects=[carried],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[{"waypoint_index": 2, "object_id": "block"}],
        sample_dt=0.01,
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "carried_object_obstacle"
    assert result["planned_events"][0] == {"type": "pick", "waypoint_index": 1, "object_id": "block", "time": 0.7}


def test_speed_change_at_continuous_segment_boundary_is_checked():
    result = validate_motion(
        [[0, 0, 0], [1, 0, 0], [1.05, 0, 0]],
        [1, 1, 1],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        segment_speed_limits=[[1, 1, 1], [0.05, 1, 1]],
        acceleration_limits=[1, 1, 1],
        velocity_transition_time=0.2,
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "velocity_transition"


def test_two_safe_operations_are_checked_as_one_continuous_task():
    carried = {"id": "block", "name": "block", "radius": 0.05, "offset_x": 0.1}
    result = validate_motion(
        [[0, 0, 0], [0.4, 0, 0], [0.45, 0, 0]],
        [1, 1, 1],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        segment_speed_limits=[[1, 1, 1], [0.05, 1, 1]],
        acceleration_limits=[1, 1, 1],
        carried_objects=[carried],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[{"waypoint_index": 2, "object_id": "block"}],
        task_operations=[
            {"id": "op-1", "name": "pick", "version": "v1", "start_waypoint_index": 0, "end_waypoint_index": 1},
            {"id": "op-2", "name": "place", "version": "v1", "start_waypoint_index": 1, "end_waypoint_index": 2},
        ],
        handoff_conditions=[{
            "id": "handoff-1",
            "waypoint_index": 1,
            "source_operation_id": "op-1",
            "target_operation_id": "op-2",
            "pose": [0.4, 0, 0],
            "carried_object_id": "block",
            "gripper_state": "closed",
            "confidence": "confirmed",
        }],
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "velocity_transition"
    assert result["first_collision"]["time"] == result["segment_durations"][0]
    assert result["task_review"]["segment_provenance"][1]["operation_version"] == "v1"
    assert result["carried_ranges"][0]["object_id"] == "block"


def test_replacing_later_operation_marks_downstream_segments_for_recheck():
    operations = [
        {"id": "op-1", "name": "first", "version": "v1", "start_waypoint_index": 0, "end_waypoint_index": 1},
        {"id": "op-2", "name": "second", "version": "v1", "start_waypoint_index": 1, "end_waypoint_index": 2},
    ]
    kwargs = dict(
        speed_limits=[1, 1, 1],
        joint_limits=JOINT_LIMITS,
        link_lengths=LINK_LENGTHS,
        obstacles=[],
    )
    baseline = validate_motion([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], task_operations=operations, **kwargs)
    operations[1]["version"] = "v2"
    updated = validate_motion([[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], task_operations=operations, baseline_snapshot=baseline, **kwargs)

    assert updated["task_review"]["changed_segments"] == [1]
    assert updated["task_review"]["revalidation_required_segments"] == [1]
    pose_changed = validate_motion([[0, 0, 0], [0.12, 0, 0], [0.2, 0, 0]], task_operations=operations, baseline_snapshot=updated, **kwargs)
    assert pose_changed["task_review"]["changed_segments"] == [0, 1]
    assert pose_changed["task_review"]["revalidation_required_segments"] == [0, 1]


def test_pick_failure_and_early_release_keep_separate_actual_carry_and_unknown_ranges():
    carried = {"id": "block", "name": "block", "radius": 0.05, "offset_x": 0.1}
    result = validate_motion(
        [[0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]],
        [1, 1, 1],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        carried_objects=[carried],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        place_events=[{"waypoint_index": 3, "object_id": "block"}],
        task_failure_events=[
            {"id": "failed-pick", "name": "pick failed", "kind": "pick_failure", "waypoint_index": 1, "object_id": "block"},
            {"id": "early-open", "name": "early open", "kind": "early_release", "waypoint_index": 2, "object_id": "block"},
        ],
    )

    failed_pick, early_release = result["abnormal_scenarios"]
    assert failed_pick["carried_ranges"] == []
    assert failed_pick["unknown_ranges"][0]["start_time"] == 0.2
    assert early_release["carried_ranges"] == [{
        "object_id": "block",
        "start_time": 0.2,
        "end_time": 0.4,
        "confidence": "confirmed",
    }]
    assert early_release["first_collision"]["kind"] == "carry_state_unknown"
    assert early_release["unknown_ranges"][0]["start_time"] == 0.4


def test_handoff_condition_is_checked_at_task_boundary():
    carried = {"id": "block", "name": "block", "radius": 0.05}
    result = validate_motion(
        [[0, 0, 0], [0.2, 0, 0]],
        [1, 1, 1],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        carried_objects=[carried],
        pick_events=[{"waypoint_index": 1, "object_id": "block"}],
        handoff_conditions=[{
            "id": "handoff-1",
            "waypoint_index": 1,
            "source_operation_id": "op-1",
            "target_operation_id": "op-2",
            "pose": [0.3, 0, 0],
            "carried_object_id": None,
            "gripper_state": "open",
            "confidence": "unknown",
        }],
    )

    kinds = {item["kind"] for item in result["hazard_intervals"]}
    assert "handoff_pose_mismatch" in kinds
    assert "handoff_carry_unknown" in kinds
    assert "handoff_gripper_mismatch" in kinds
