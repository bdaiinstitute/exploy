# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from isaaclab.envs import ManagerBasedRLEnv


class TermResetter:
    """Manager environment elements that are replaced with exporter-friendly versions.

    This class encapsulates how to replace `ManagerTerm` objects in environment managers with exporter-friendly versions
    and how to set them back to their original type.

    For example, the `EELimbMotionGenerator` generates commands from trajectories that are loaded from file.
    The functionality that we want to export is just how to get/set commands, which is abstracted into
    `EELimbMotionGeneratorExporter`. This class handles replacing the former type with the latter for the export phase,
    and setting the term back to a `EELimbMotionGenerator` when export is done.
    """

    def __init__(self, env: ManagerBasedRLEnv):
        self._resetters = []

    def add(self, reset_func: callable):
        self._resetters.append(reset_func)

    def __call__(self):
        """Call all reset functions to set back objects to their original type."""
        for func in self._resetters:
            func()
