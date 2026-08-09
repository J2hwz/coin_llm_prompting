import sys
from pathlib import Path

import pytest

_ALGO_DIR = Path(__file__).resolve().parent.parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from inv_planning import _corridor_dead_end_fixture


@pytest.fixture
def dead_end_fixture():
    """(layout, paths, true_coin_json) for a 4x7 corridor with one dead-end branch.

    Reused from inv_planning.py rather than redefined — see its docstring for
    why a true dead-end guarantees a strict, non-tied argmax.
    """
    return _corridor_dead_end_fixture()


@pytest.fixture
def wall_bump_paths():
    """Two JSON-coord paths on the dead_end_fixture layout/grid that are
    identical except one has an extra wall-bump-style back-and-forth detour
    at the same cell (col=3, row_top=1) before continuing on.

    Non-baseline algorithms (which drop zero-net-movement steps) should score
    these two paths identically; Cell Visit Freq should not.
    """
    clean = [[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]]
    with_detour = [[1, 1], [2, 1], [3, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]]
    return clean, with_detour
