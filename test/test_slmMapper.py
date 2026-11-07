import datetime

import slmMapper as smp


def test_filtra_regiao_keeps_points_inside_bounds():
    ts = [1, 2, 3]
    lats = [0.0, 10.0, -1.0]
    lngs = [-50.0, -50.0, -10.0]

    out_ts, out_lats, out_lngs = smp.filtra_regiao(ts, lats, lngs, "BR")

    assert out_ts == [1]
    assert out_lats == [0.0]
    assert out_lngs == [-50.0]


def test_le_input_skips_malformed_rows(tmp_path):
    csv_path = tmp_path / "raios.csv"
    csv_path.write_text(
        "timestamp,lat,lng\n"
        "1700000000,1.0,-2.0\n"
        "1700000010,not-a-number,-2.0\n"
        "1700000020,3.0,-4.0\n"
    )

    ts, lats, lngs = smp.le_input(csv_path)

    assert len(ts) == 2
    assert lats == [1.0, 3.0]
    assert lngs == [-2.0, -4.0]
    assert all(t.tzinfo == datetime.timezone.utc for t in ts)


def test_main_sem_raios_na_regiao_retorna_rc_3(tmp_path):
    # [fix] sem raios na região, plot_mapa_raios levantava RuntimeError
    csv_path = tmp_path / "raios.csv"
    csv_path.write_text("timestamp,lat,lng\n1700000000,50.0,10.0\n")

    li_rc = smp.main(["-r", "BR", "-s", str(tmp_path / "out"), "-f", "m.png", str(csv_path)])

    assert li_rc == smp.CI_RC_NO_LIGHTNING
    assert not (tmp_path / "out" / "m.png").exists()
