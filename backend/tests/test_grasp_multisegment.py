from robot_platform.robotics import validate_motion


LINK_LENGTHS = [1.4, 1.0, 0.7]
JOINT_LIMITS = [[-2.8, 2.8], [-2.4, 2.4], [-2.4, 2.4]]


def test_segment_speed_limits_set_each_duration():
    result = validate_motion(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [1.0, 1.0, 1.0],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        segment_speed_limits=[[1.0, 1.0, 1.0], [0.25, 1.0, 1.0]],
        sample_dt=0.02,
    )

    assert abs(result["segment_durations"][0] - 0.5) < 0.02
    assert abs(result["segment_durations"][1] - 2.0) < 0.02


def test_speed_jump_at_junction_is_rejected():
    result = validate_motion(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [1.0, 1.0, 1.0],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        segment_speed_limits=[[1.0, 1.0, 1.0], [0.2, 1.0, 1.0]],
        junction_speed_tolerance=0.2,
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "speed_jump"
    assert result["segments_solved"][1] is False
    assert result["segments_solved"][0] is True


def test_continuous_speeds_pass_the_junction():
    result = validate_motion(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [1.0, 1.0, 1.0],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        segment_speed_limits=[[0.5, 1.0, 1.0], [0.5, 1.0, 1.0]],
    )
    assert result["valid"] is True


def test_carried_object_collision_only_counts_after_grasp():
    waypoints = [
        [0.7761, -2.4, 1.6057],
        [1.1761, -2.1472, 1.9557],
        [1.3261, -2.1472, 2.1057],
    ]
    objects = [
        {
            "name": "block",
            "x": 1.49,
            "y": 1.05,
            "radius": 0.12,
            "hold_dx": 0.12,
        }
    ]
    actions = [
        {"type": "grasp", "waypoint_index": 1, "object": "block"},
        {"type": "release", "waypoint_index": 2, "object": "block"},
    ]
    safe = validate_motion(
        waypoints,
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [{"name": "shelf", "x": 1.95, "y": 1.25, "radius": 0.12}],
        carried_objects=objects,
        actions=actions,
        grasp_tolerance=0.25,
        junction_speed_tolerance=0.9,
        sample_dt=0.01,
    )
    assert safe["valid"] is True
    assert safe["grasp_events"][0]["object"] == "block"
    held_frames = [frame for frame in safe["frames"] if frame["objects"][0]["state"] == "held"]
    assert held_frames

    blocked = validate_motion(
        waypoints,
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [{"name": "shelf", "x": 1.62, "y": 1.22, "radius": 0.1}],
        carried_objects=objects,
        actions=actions,
        grasp_tolerance=0.25,
        junction_speed_tolerance=0.9,
        sample_dt=0.01,
    )
    assert blocked["valid"] is False
    collision = blocked["first_collision"]
    assert collision["kind"] == "carried_obstacle"
    grasp_time = blocked["grasp_events"][0]["time"]
    assert collision["time"] >= grasp_time - 1e-9


def test_grasp_far_from_object_rejects_but_earlier_segments_remain_solved():
    result = validate_motion(
        [[0.0, 0.0, 0.0], [0.4, 0.2, 0.0], [0.4, 0.9, 0.0]],
        [0.8, 0.8, 0.8],
        JOINT_LIMITS,
        LINK_LENGTHS,
        [],
        carried_objects=[{"name": "block", "x": 0.2, "y": 0.2, "radius": 0.1}],
        actions=[{"type": "grasp", "waypoint_index": 1, "object": "block"}],
        grasp_tolerance=0.1,
    )

    assert result["valid"] is False
    assert result["first_collision"]["kind"] == "grasp_unreachable"
    assert result["segments_solved"][0] is True
