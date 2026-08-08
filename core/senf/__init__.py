"""SENF: a sidecar entity/frame representation over canonicalized PLN atoms.

Nothing in the running system imports this package yet. It is introduced ahead of
its consumers so the deterministic layers can be tested in isolation before any
parser depends on them.
"""

from core.senf.identity import (
    DEFAULT_IDENTITY_THRESHOLD,
    IdentityEdge,
    IdentityGraph,
    IdentityResolver,
    IdentityWeights,
    resolve_identity,
)
from core.senf.types import (
    Filler,
    Literal,
    Mention,
    MentionType,
    Role,
    SENF,
    SENF_PAYLOAD_KEY,
    SENF_PAYLOAD_VERSION,
    SENFFrame,
    senf_from_payload,
    senf_to_payload,
)

__all__ = [
    "DEFAULT_IDENTITY_THRESHOLD",
    "Filler",
    "IdentityEdge",
    "IdentityGraph",
    "IdentityResolver",
    "IdentityWeights",
    "Literal",
    "Mention",
    "MentionType",
    "Role",
    "SENF",
    "SENF_PAYLOAD_KEY",
    "SENF_PAYLOAD_VERSION",
    "SENFFrame",
    "resolve_identity",
    "senf_from_payload",
    "senf_to_payload",
]
