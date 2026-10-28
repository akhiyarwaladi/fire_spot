import pytest

from fire_spot.cli import main


def test_analysis_missing_file_exits_with_failure(capsys):
    with pytest.raises(SystemExit) as error:
        main(["analyze", "--data", "missing-file.csv"])
    assert error.value.code == 2
    assert "CSV tidak ditemukan" in capsys.readouterr().err


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2


def test_default_boundary_path_works_outside_repository(fire_data, tmp_path, monkeypatch):
    source = tmp_path / "points.csv"
    fire_data.to_csv(source, index=False)
    monkeypatch.chdir(tmp_path)
    destination = tmp_path / "filtered"
    assert (
        main(
            [
                "filter",
                "--regions",
                "Jambi",
                "--no-maps",
                "--data",
                str(source),
                "--output-dir",
                str(destination),
            ]
        )
        == 0
    )
    assert (destination / "jambi.csv").is_file()


def test_conflicting_region_selections_are_rejected():
    with pytest.raises(SystemExit) as error:
        main(["filter", "--sumatra", "--all"])
    assert error.value.code == 2
