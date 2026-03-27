# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import logging
import pathlib

import pytest

try:
    from isaaclab.app import AppLauncher

    HAS_ISAACLAB = True
except ImportError:
    HAS_ISAACLAB = False

logger = logging.getLogger(__name__)


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """
    Modify collected test items to add custom markers based on their file paths.

    Note:
        Isaaclab tests are skipped in the global test suite if `isaaclab` is not found.
    """

    # Add 'isaaclab' marker to tests in the same directory as this conftest.py.
    for item in items:
        if str(pathlib.Path(__file__).parent) in str(item.path):
            item.add_marker(marker="isaaclab")

    if HAS_ISAACLAB:
        return

    skip_isaaclab = pytest.mark.skip(
        reason="Optional dependency 'isaaclab' not found, skipping test",
    )

    # Skip tests with 'isaaclab' mark.
    for item in items:
        if "isaaclab" in item.keywords:
            item.add_marker(skip_isaaclab)


@pytest.fixture(scope="session")
def sim_app():
    """
    Session-level fixture to start the SimulationApp.
    Isaac Sim cannot be restarted in the same process, so we keep it alive.
    """
    # Initialize the launcher with AppLauncher arguments
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    yield simulation_app

    # Clean teardown at the end of the test session
    logger.info("Closing Simulation App...")
    simulation_app.close()


@pytest.fixture(scope="function")
def sim_setup(sim_app):
    """
    Function-level fixture to ensure app is ready for tests.
    Each test handles its own stage setup/teardown as needed.
    """
    return sim_app
