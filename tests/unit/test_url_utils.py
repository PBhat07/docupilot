
import pytest

from docupilot.crawler.url_utils import (
    is_within_scope,
    normalize_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "HTTPS://DOCS.EXAMPLE.COM:443/guide#intro",
            "https://docs.example.com/guide",
        ),
        (
            "http://docs.example.com:80/guide",
            "http://docs.example.com/guide",
        ),
        (
            "https://docs.example.com/guide?version=2",
            "https://docs.example.com/guide?version=2",
        ),
        (
            "https://docs.example.com/guide?utm_source=test&version=2",
            "https://docs.example.com/guide?version=2",
        ),
        (
            "https://docs.example.com/guide/",
            "https://docs.example.com/guide/",
        ),
        ("mailto:help@example.com", None),
        ("javascript:alert(1)", None),
        ("https://user:pass@docs.example.com/", None),
        ("", None),
    ],
)
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


def test_resolves_relative_url():
    assert normalize_url(
        "../api/",
        "https://docs.example.com/guide/start/",
    ) == "https://docs.example.com/guide/api/"


def test_same_document_with_different_fragments_normalizes_identically():
    first = normalize_url("https://docs.example.com/guide#intro")
    second = normalize_url("https://docs.example.com/guide#examples")

    assert first == second


def test_scope_accepts_same_documentation_directory():
    root = "https://docs.example.com/v2/"

    assert is_within_scope(
        "https://docs.example.com/v2/guide",
        root,
    )


def test_scope_rejects_other_host():
    assert not is_within_scope(
        "https://other.example.com/v2/guide",
        "https://docs.example.com/v2/",
    )


def test_scope_rejects_similar_but_outside_path():
    assert not is_within_scope(
        "https://docs.example.com/v2-other/guide",
        "https://docs.example.com/v2/",
    )


def test_scope_rejects_different_scheme():
    assert not is_within_scope(
        "http://docs.example.com/v2/guide",
        "https://docs.example.com/v2/",
    )


def test_invalid_port_is_rejected():
    assert normalize_url("https://docs.example.com:99999/") is None
