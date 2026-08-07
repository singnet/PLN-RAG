"""SENF: a sidecar entity/frame representation over canonicalized PLN atoms.

Nothing in the running system imports this package yet. It is introduced ahead of
its consumers so the deterministic layers can be tested in isolation before any
parser depends on them.
"""

from core.senf.types import (
    Filler,
    Literal,
    Mention,
    MentionType,
    Role,
    SENF,
    SENFFrame,
)

__all__ = [
    "Filler",
    "Literal",
    "Mention",
    "MentionType",
    "Role",
    "SENF",
    "SENFFrame",
]
