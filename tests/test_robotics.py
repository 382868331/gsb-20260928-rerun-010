import pytest

from backend.robotics import Obstacle, collision_report, default_project, segment_duration, simulate


def test_default_path_has_safe_endpoints_but_rejects_midpath_obstacle_collision():
    project = default_project()
    robot = project["robot"]
    obstacles = [Obstacle(**item) for item in project["obstacles"]]

    start_report = collision_report(
        project["waypoints"][0]["angles"],
        robot["link_lengths"],
        robot["link_radius"],
        robot["joint_limits"],
        obstacles,
    )
    end_report = collision_report(
        project["waypoints"][1]["angles"],
        robot["link_lengths"],
        robot["link_radius"],
        robot["joint_limits"],
        obstacles,
    )

    assert start_report is None
    assert end_report is None

    result = simulate(project)

    assert result["accepted"] is False
    assert result["rejection"]["type"] == "obstacle"
    assert 0 < result["rejection"]["time"] < result["duration"]
    assert result["rejection"]["link"] == 1


def test_safe_path_is_accepted():
    project = default_project()
    project["obstacles"] = []

    result = simulate(project)

    assert result["accepted"] is True
    assert result["rejection"] is None
    assert result["samples"][0]["time"] == 0
    assert result["samples"][-1]["time"] == pytest.approx(2.333333, abs=1e-5)


def test_speed_limits_change_duration():
    project = default_project()
    project["obstacles"] = []

    fast = simulate(project)
    project["max_joint_speeds"] = [120, 120, 120]
    faster = simulate(project)

    assert faster["duration"] == pytest.approx(fast["duration"] / 2)


def test_joint_limits_are_checked_for_every_configuration():
    project = default_project()
    project["obstacles"] = []
    project["waypoints"][0]["angles"] = [180, 20, 20]
    project["waypoints"][1]["angles"] = [120, 20, 20]

    result = simulate(project)

    assert result["accepted"] is False
    assert result["rejection"]["type"] == "joint_limit"
    assert result["rejection"]["joint"] == 0
    assert result["rejection"]["time"] == 0


def test_segment_duration_uses_slowest_joint_speed():
    duration = segment_duration([0, 0], [30, -90], [60, 45])
    assert duration == pytest.approx(2.0)
