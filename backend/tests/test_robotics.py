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
