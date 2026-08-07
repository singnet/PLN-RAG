"""Characterization tests for symbol normalization primitives.

These are the canonicalization functions that SENF must route through.
Changing any of these invalidates every atom already on disk, so their current
behavior is pinned by these tests.
"""

import pytest

from core.symbol_normalization import canonical_symbol, normalize_text, pluralize, singularize


class TestSingularize:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("dogs", "dog"),
            ("cats", "cat"),
            ("properties", "property"),
            ("losses", "loss"),
            ("status", "status"),
            ("bus", "bus"),
            ("octopus", "octopus"),
            ("analysis", "analysis"),
            ("fish", "fish"),
            ("candidates", "candidate"),
            ("man", "man"),
            # Lossy: the "ses" rule strips two chars regardless of stem.
            ("horses", "hors"),
            # Lossy: no "es" rule, so only the trailing "s" comes off.
            ("boxes", "boxe"),
        ],
    )
    def test_singularize(self, word, expected):
        assert singularize(word) == expected

    def test_short_word_preserved(self):
        assert singularize("is") == "is"
        assert singularize("as") == "as"


class TestPluralize:
    @pytest.mark.parametrize(
        "word,expected",
        [
            ("dog", "dogs"),
            ("property", "properties"),
            ("box", "boxes"),
            ("bus", "buses"),
            # Naive: adds "es" to "sh" ending, doesn't know irregulars.
            ("fish", "fishes"),
        ],
    )
    def test_pluralize(self, word, expected):
        assert pluralize(word) == expected

    def test_not_inverse_of_singularize(self):
        """pluralize(singularize(w)) is not a round-trip. Never chain them."""
        # Irregulars break it: "fish" is its own plural, but pluralize adds "es".
        assert pluralize(singularize("fish")) == "fishes"
        # Plural-only stems get mangled even when the round-trip looks safe:
        # "horses" -> "hors" -> "horses" only coincides because "hors" ends in "s".
        assert singularize("horses") == "hors"


class TestCanonicalSymbol:
    @pytest.mark.parametrize(
        "token,expected",
        [
            ("HelloWorld", "hello_world"),
            ("Hello-World", "hello_world"),
            ("hello world", "hello_world"),
            ("  hello  ", "hello"),
            ("candidates", "candidate"),
            ("properties", "property"),
            ("IsA", "is_a"),
            ("FishEater", "fish_eater"),
            ("kebede", "kebede"),
            ("HbA1c", "hb_a1c"),
            ("type 2", "type_2"),
            ("", ""),
        ],
    )
    def test_canonical_symbol(self, token, expected):
        assert canonical_symbol(token) == expected

    def test_disable_lemmatize(self):
        assert canonical_symbol("axes", lemmatize=False) == "axes"

    def test_protect_disables_lemmatize(self):
        assert canonical_symbol("Google", protect=True) == "google"

    def test_idempotent_on_already_canonical(self):
        first = canonical_symbol("HelloWorld")
        second = canonical_symbol(first)
        assert first == second


class TestNormalizeText:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Hello World", "hello world"),
            ("Hello-World", "hello world"),
            ("Is Kebede smart?", "is kebede smart"),
            ("Hello, World!", "hello world"),
            ("  extra   spaces  ", "extra spaces"),
        ],
    )
    def test_normalize_text(self, text, expected):
        assert normalize_text(text) == expected
