from robot_platform.schemas import Project
from robot_platform.storage import ProjectStore


def test_project_store_saves_loads_and_updates(tmp_path):
    store = ProjectStore(tmp_path)
    project = Project.model_validate(
        {
            "name": "stored",
            "waypoints": [[0, 0, 0], [0.1, 0.1, 0.1]],
            "speed_limits": [1, 1, 1],
            "joint_limits": [[-1, 1], [-1, 1], [-1, 1]],
            "link_lengths": [1, 1, 1],
            "obstacles": [],
        }
    )

    saved = store.save(project)
    loaded = store.get(saved.id)
    assert loaded.name == "stored"

    updated = store.save(saved.model_copy(update={"name": "renamed"}), saved.id)
    assert updated.id == saved.id
    assert store.get(saved.id).name == "renamed"
