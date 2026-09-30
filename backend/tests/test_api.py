from robot_platform import main
from robot_platform.main import save_project, validate_project
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


def test_saved_project_keeps_incomplete_editor_data_but_is_not_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", main.ProjectStore(tmp_path))
    invalid = Project.model_validate(project_payload())
    saved_invalid = save_project(invalid)
    assert saved_invalid.executable is False
    assert saved_invalid.last_validation["valid"] is False
    assert main.store.get(saved_invalid.id).name == invalid.name

    payload = project_payload()
    payload.update(
        {
            "waypoints": [[0, 0, 0], [0.1, 0, 0]],
            "obstacles": [],
            "carried_objects": [{"id": "block", "name": "block", "radius": 0.08}],
            "pick_events": [{"waypoint_index": 1, "object_id": "block"}],
        }
    )
    saved_valid = save_project(Project.model_validate(payload))
    assert saved_valid.executable is True
    assert saved_valid.planned_events[0]["type"] == "pick"
    assert saved_valid.constraint_version == "1.1"
