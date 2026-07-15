"""Natural-drop diagnostic contracts for packaging."""

from __future__ import annotations

import math

import pytest

from ufactory.config import load_runtime_config
from ufactory.manipulation.packaging.drop import measure_natural_drop


def test_natural_drop_rejects_invalid_windows() -> None:
    config = load_runtime_config("xarm6", task="packaging_showcase")
    with pytest.raises(ValueError, match="positive"):
        measure_natural_drop(config, max_steps=0)


@pytest.mark.gpu
def test_natural_drop_records_contact_without_rebound_threshold() -> None:
    config = load_runtime_config("xarm6", task="packaging_showcase")
    report = measure_natural_drop(config)

    assert report.samples > 0
    assert report.impact_time_s > 0.0
    assert report.impact_velocity_m_s < 0.0
    assert math.isfinite(report.post_impact_velocity_m_s)
    assert report.maximum_post_impact_velocity_m_s >= 0.0
    assert report.rebound_height_m >= 0.0
    assert report.settled_time_s is None or report.settled_time_s >= report.impact_time_s
