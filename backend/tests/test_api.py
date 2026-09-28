from robot_platform.main import validate_project
from robot_platform.schemas import Project


def project_payload():
    return {
        "name": "api-test",
        "waypoints": [[-0.65, 0, 0], [0.65, 0, 0]],
        "speed_limits": [0.8, 0.8, 0.8],
        "joint_limits": [[-2.8, 2.8], [-2.4, 2.4], [-2.4, 2.4]],
        "link_lengths": [1.4, 1, 0.7],
        "obstacles": [{"name": "middle-sweep", "x": 1.75, "y": 0.12, "radius": 0.1}],
        "link_radius": 0.08,
        "sample_dt": 0.02,
    }


def test_validate_route_rejects_mid_path_collision():
    data = validate_project(Project.model_validate(project_payload()))
    assert data["valid"] is False
    assert data["first_collision"]["kind"] == "obstacle"
