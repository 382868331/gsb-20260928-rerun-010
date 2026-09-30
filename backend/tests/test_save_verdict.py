from robot_platform.main import save_project, validate_project
from robot_platform.schemas import Project
from robot_platform.storage import ProjectStore


def base_payload(obstacles):
    return {
        "name": "verdict",
        "waypoints": [[0, 0, 0], [0.1, 0.1, 0.1]],
        "speed_limits": [1, 1, 1],
        "joint_limits": [[-1, 1], [-1, 1], [-1, 1]],
        "link_lengths": [1, 1, 1],
        "obstacles": obstacles,
    }


def test_incomplete_plan_is_not_marked_executable(tmp_path, monkeypatch):
    import robot_platform.main as main_module

    monkeypatch.setattr(main_module, "store", ProjectStore(tmp_path))
    blocked = Project.model_validate(
        base_payload([{"name": "wall", "x": 0.5, "y": 0.5, "radius": 0.4}])
    )
    saved = save_project(blocked)
    assert saved.executable is False
    assert saved.constraint_version is not None
    assert saved.segments_solved == [False]
    assert main_module.store.get(saved.id).executable is False

    clear = Project.model_validate(base_payload([]))
    ok = main_module.store.save(clear, saved.id, verdict=validate_project(clear))
    assert ok.executable is True
    assert ok.segments_solved == [True]
    assert ok.grasp_events == []


def test_replay_revalidates_with_saved_constraint_version(tmp_path, monkeypatch):
    import robot_platform.main as main_module

    monkeypatch.setattr(main_module, "store", ProjectStore(tmp_path))
    saved = save_project(Project.model_validate(base_payload([])))
    verdict = validate_project(Project.model_validate(saved.model_dump(exclude={"id"})))
    assert verdict["valid"] is True
    assert verdict["constraint_version"] == saved.constraint_version
