from __future__ import annotations

from chimera_v2c.tools.backfill_moneypuck_pregame import _cell_blank


def test_backfill_moneypuck_pregame_treats_nr_as_fillable() -> None:
    assert _cell_blank("") is True
    assert _cell_blank("NR") is True
    assert _cell_blank(" nr ") is True
    assert _cell_blank(".55") is False
