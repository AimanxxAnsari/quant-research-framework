"""Smoke test: the Pillar 5 study functions run end to end with tiny repetition counts."""

from quant_research.studies import pillar5


def test_pillar5_study_functions_run(tmp_path):
    hl = pillar5.half_life_study(tmp_path, reps=5)
    assert len(hl) == 5 and (tmp_path / "figures" / "half_life_bias.png").exists()
    co = pillar5.cointegration_study(reps=5, n=200)
    assert 0 <= co["eg_false_positive"] <= 1
    vr = pillar5.variance_ratio_size_study(reps=5, n=300)
    assert set(vr) == {"iid normal", "GARCH(1,1)"}
    var = pillar5.var_study(tmp_path, reps=2, n=600)
    assert len(var) == 3
    table, ma = pillar5.overfitting_study(tmp_path, reps=3, ma_reps=2)
    assert ma["grid"] == 25 and "1 strategy" in table.index
