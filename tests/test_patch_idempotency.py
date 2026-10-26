"""
tests/test_patch_idempotency.py - Verify FrictionDetector patch is applied exactly once.
"""

import importlib
import pytest
from gymnasium.envs.box2d import car_racing
from f1sim.env import apply_finish_line_patch


def test_sentinel_attribute_exists():
    """Verify sentinel attribute _f1sim_patched exists on FrictionDetector."""
    assert hasattr(car_racing.FrictionDetector, "_f1sim_patched")
    assert car_racing.FrictionDetector._f1sim_patched is True


def test_patch_is_idempotent():
    """Verify repeated patch invocations do not duplicate or stack wrappers."""
    method_before = car_racing.FrictionDetector.BeginContact

    # Call again
    reapplied = apply_finish_line_patch()
    assert reapplied is False, "apply_finish_line_patch should return False when already patched"

    method_after = car_racing.FrictionDetector.BeginContact
    assert method_before is method_after, "BeginContact function pointer changed on redundant patch call"


def test_module_reload_safety():
    """Verify re-importing env module maintains patch integrity."""
    import f1sim.env
    importlib.reload(f1sim.env)
    assert getattr(car_racing.FrictionDetector, "_f1sim_patched", False) is True
