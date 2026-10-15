
import pytest

from docupilot.crawler.policy import CrawlPolicy


def test_admits_root_at_depth_zero():
    policy = CrawlPolicy(
        "https://docs.example.com/v2/",
        max_pages=5,
        max_depth=2,
    )

    result = policy.admit("https://docs.example.com/v2/", depth=0)

    assert result.accepted
    assert result.reason == "accepted"
    assert policy.admitted_count == 1


def test_rejects_duplicate_urls():
    policy = CrawlPolicy("https://docs.example.com/v2/")

    first = policy.admit(
        "https://docs.example.com/v2/guide#intro", 0
    )
    second = policy.admit(
        "https://docs.example.com/v2/guide#examples", 1
    )

    assert first.accepted
    assert not second.accepted
    assert second.reason == "duplicate"
    assert policy.admitted_count == 1


def test_rejects_url_outside_scope():
    policy = CrawlPolicy("https://docs.example.com/v2/")

    result = policy.admit(
        "https://docs.example.com/v3/guide", 1
    )

    assert not result.accepted
    assert result.reason == "out_of_scope"


def test_rejects_depth_beyond_limit():
    policy = CrawlPolicy(
        "https://docs.example.com/v2/",
        max_depth=2,
    )

    result = policy.admit(
        "https://docs.example.com/v2/deep/page", 3
    )

    assert not result.accepted
    assert result.reason == "depth_limit"


def test_allows_depth_equal_to_limit():
    policy = CrawlPolicy(
        "https://docs.example.com/v2/",
        max_depth=2,
    )

    result = policy.admit(
        "https://docs.example.com/v2/deep/page", 2
    )

    assert result.accepted


def test_page_limit_includes_reserved_urls():
    policy = CrawlPolicy(
        "https://docs.example.com/v2/",
        max_pages=2,
    )

    assert policy.admit("https://docs.example.com/v2/", 0).accepted
    assert policy.admit("https://docs.example.com/v2/guide", 1).accepted

    result = policy.admit(
        "https://docs.example.com/v2/reference", 1
    )

    assert not result.accepted
    assert result.reason == "page_limit"
    assert policy.admitted_count == 2


def test_invalid_url_is_rejected():
    policy = CrawlPolicy("https://docs.example.com/v2/")

    result = policy.admit("javascript:alert(1)", 1)

    assert not result.accepted
    assert result.reason == "invalid_url"


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        CrawlPolicy("https://docs.example.com/", max_pages=0)

    with pytest.raises(ValueError):
        CrawlPolicy("https://docs.example.com/", max_depth=-1)


def test_rejection_counts_are_reported():
    policy = CrawlPolicy("https://docs.example.com/v2/")

    policy.admit("https://other.example.com/page", 1)
    policy.admit("https://docs.example.com/v2/guide", 3)

    assert policy.rejection_counts == {
        "out_of_scope": 1,
        "depth_limit": 1,
    }
