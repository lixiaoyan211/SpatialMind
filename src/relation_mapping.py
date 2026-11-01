"""Project-defined analysis groups for VSR spatial relations.

These groups are not official VSR categories. Each original relation belongs to
exactly one group; ambiguous or path-like relations use ``general_location``.
"""

from collections import Counter


RELATION_GROUPS = {
    "left_right": (
        "at the left side of", "at the right side of", "left of", "right of",
    ),
    "up_down_support": (
        "above", "below", "beneath", "down", "down from", "off", "on",
        "on top of", "over", "under",
    ),
    "front_back": (
        "ahead of", "at the back of", "behind", "beyond", "in front of",
    ),
    "orientation_alignment": (
        "congruent", "facing", "facing away from", "opposite to",
        "parallel to", "perpendicular to", "toward",
    ),
    "distance": (
        "away from", "close to", "far away from", "far from", "near",
    ),
    "adjacency_contact": (
        "adjacent to", "against", "alongside", "at the side of",
        "attached to", "beside", "by", "connected to", "detached from",
        "next to", "touching", "with",
    ),
    "containment_part_whole": (
        "among", "around", "between", "consists of", "contains",
        "enclosed by", "has as a part", "in", "in the middle of", "inside",
        "into", "out of", "outside", "part of", "surrounding", "within",
    ),
    "general_location": (
        "across", "across from", "along", "at", "at the edge of", "past",
        "through",
    ),
}

CATEGORY_LABELS = {
    "left_right": "Left / right",
    "up_down_support": "Up / down and support",
    "front_back": "Front / back",
    "orientation_alignment": "Orientation and alignment",
    "distance": "Distance",
    "adjacency_contact": "Adjacency and contact",
    "containment_part_whole": "Containment and part-whole",
    "general_location": "General location and path",
}

_counts = Counter(relation for relations in RELATION_GROUPS.values() for relation in relations)
_duplicates = sorted(relation for relation, count in _counts.items() if count != 1)
if _duplicates:
    raise ValueError(f"Relations assigned to multiple groups: {_duplicates}")
if set(RELATION_GROUPS) != set(CATEGORY_LABELS):
    raise ValueError("Category labels must cover every relation group")

RELATION_TO_CATEGORY = {
    relation: category
    for category, relations in RELATION_GROUPS.items()
    for relation in relations
}


def category_for_relation(relation: str) -> str:
    """Return the analysis group, rejecting relations absent from the mapping."""
    try:
        return RELATION_TO_CATEGORY[relation]
    except KeyError as exc:
        raise ValueError(f"Unknown VSR relation: {relation!r}") from exc
