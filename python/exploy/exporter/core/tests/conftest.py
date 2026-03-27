# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import pathlib

import pytest


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Modify collected test items to add custom markers based on their file paths.
    """

    # Add 'core' marker to tests in the same directory as this conftest.py.
    for item in items:
        if str(pathlib.Path(__file__).parent) in str(item.path):
            item.add_marker(marker="core")
