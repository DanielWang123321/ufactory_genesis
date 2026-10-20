"""Table loads from entity-filtered Genesis contact arrays, including padding."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def table_contact_loads(
    contacts: Mapping[str, Any], *, table_link_idx: int, object_link_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return upward object support and total robot load on the table (N).

    The caller passes table.get_contacts(). Select the force on the non-table
    side, independent of A/B ordering, and separate object and robot pairs.
    """
    a, b = contacts["link_a"], contacts["link_b"]
    fa, fb = contacts["force_a"], contacts["force_b"]
    valid = contacts.get("valid_mask", torch.ones_like(a, dtype=torch.bool))
    if a.ndim == 1:
        a, b, fa, fb, valid = (x.unsqueeze(0) for x in (a, b, fa, fb, valid))
    table_a, table_b = a == table_link_idx, b == table_link_idx
    valid = valid & (table_a | table_b)
    other = torch.where(table_a, b, a)
    force = torch.where(table_a[..., None], fb, fa)
    obj_mask = valid & (other == object_link_idx)
    robot_mask = valid & (other != object_link_idx)
    support = torch.where(obj_mask, force[..., 2], 0).sum(-1).clamp(min=0)
    robot_load = torch.where(robot_mask, torch.linalg.vector_norm(force, dim=-1), 0).sum(-1)
    return support, robot_load
