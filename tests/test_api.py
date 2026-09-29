from backend import main
from backend.main import SaveRequest, get_project, list_projects, run_simulation
from backend.robotics import default_project


def test_simulation_route_rejects_midpath_collision():
    result = run_simulation(SaveRequest(**default_project()))

    assert result["accepted"] is False
    assert result["rejection"]["type"] == "obstacle"


def test_project_persistence_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECTS_DIR", tmp_path)
    payload = SaveRequest(**default_project())

    saved = main.create_project(payload)
    loaded = get_project(saved["id"])

    assert loaded["name"] == saved["name"]
    assert len(list_projects()) == 1
    assert (tmp_path / f"{saved['id']}.json").exists()
