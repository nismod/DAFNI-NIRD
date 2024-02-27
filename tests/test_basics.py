"""Initial test file
"""

from nird import add_one


def test_add_one() -> None:
    assert add_one(3) == 4


def test_add_one_float() -> None:
    assert add_one(3.5) == 4.5
