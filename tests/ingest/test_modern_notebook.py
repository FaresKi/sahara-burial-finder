import json
from pathlib import Path

import matplotlib
import pytest

from sbf.ingest.modern import FakeExporter, build_stack, grid_from_bounds
from sbf.ingest.modern.process import raw_dir

NB = Path(__file__).resolve().parents[2] / "notebooks" / "02_modern_detectability.ipynb"


def _run(capsys):
    matplotlib.use("Agg")
    ns: dict = {}
    nb = json.loads(NB.read_text())
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            exec("".join(c["source"]), ns)  # noqa: S102
    return capsys.readouterr().out


def test_notebook_does_not_error_without_data(tmp_project, capsys):
    out = _run(capsys)
    assert "no stack" in out and "skipped" in out


@pytest.mark.filterwarnings("ignore")
def test_notebook_runs_with_synthetic_stack(tmp_project, capsys):
    grid = grid_from_bounds((-0.0640, 26.3050, -0.0500, 26.3180), "EPSG:32631", 10.0)
    FakeExporter().export_aoi("REG", raw_dir("REG", tmp_project), grid=grid)
    build_stack("REG", settings=tmp_project, grid=grid)
    out = _run(capsys)
    assert "[REG] 16 layers" in out
    assert "no stack" in out  # INE still absent
    assert "markdown" in {c["cell_type"] for c in json.loads(NB.read_text())["cells"]}
    assert "ambiguous" in NB.read_text()
