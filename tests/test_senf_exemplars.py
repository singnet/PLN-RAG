from core.senf.exemplars import exemplar_distance, score_exemplars
from core.senf.extractor import extract_senf


def _game(sentence_id: str, text: str):
    return score_exemplars(
        extract_senf(sentence_id, text, [f"(: {sentence_id} (IsA game Game) (STV 1 1))"])
    )


def test_case_c_same_canonical_kind_has_different_active_exemplars():
    strategic = _game("s1", "The game required deep strategy.")
    physical = _game("s2", "The game was physically exhausting.")
    left, right = strategic.mentions[0], physical.mentions[0]

    assert strategic.kind_for(left) == physical.kind_for(right) == "Game"
    assert strategic.active_exemplars_for(left) == ("chess_game",)
    assert physical.active_exemplars_for(right) == ("football_game",)
    assert exemplar_distance(strategic, left, physical, right) == 0.8


def test_weak_evidence_keeps_close_active_alternatives_without_a_nearest_winner():
    senf = score_exemplars(extract_senf(
        "s1", "The camera lasted three hours.",
        ["(: a (Lasted camera three_hours) (STV 1 1))"],
    ))
    camera = next(mention for mention in senf.mentions if mention.canonical_symbol == "camera")

    assert len(senf.active_exemplars_for(camera)) == 4
    assert senf.nearest_exemplar_for(camera) is None
    assert camera.mention_id not in senf.nearest_exemplars
    assert senf.kind_assertions == [], "lexical applicability must not synthesize IsA"


def test_exemplar_context_does_not_leak_across_source_units():
    senf = score_exemplars(extract_senf(
        "doc",
        "The game required deep strategy. The game was physically exhausting.",
        [
            "(: a (IsA game Game) (STV 1 1))",
            "(: b (IsA game Game) (STV 1 1))",
        ],
    ))
    games = [mention for mention in senf.mentions if mention.canonical_symbol == "game"]
    assert [senf.active_exemplars_for(mention) for mention in games] == [
        ("chess_game",), ("football_game",),
    ]
