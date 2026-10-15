
from __future__ import annotations

from dataclasses import dataclass, field

from docupilot.crawler.url_utils import (
    is_within_scope,
    normalize_url,
)


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    url: str | None
    reason: str


@dataclass
class CrawlPolicy:
    root_url: str
    max_pages: int = 100
    max_depth: int = 2

    _seen_urls: set[str] = field(default_factory=set, init=False)
    _admitted_count: int = field(default=0, init=False)
    _rejections: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        normalized_root = normalize_url(self.root_url)

        if normalized_root is None:
            raise ValueError("root_url must be a valid HTTP(S) URL")

        if self.max_pages < 1:
            raise ValueError("max_pages must be at least 1")

        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")

        self.root_url = normalized_root

    @property
    def admitted_count(self) -> int:
        return self._admitted_count

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(self._rejections)

    @property
    def seen_urls(self) -> frozenset[str]:
        return frozenset(self._seen_urls)

    def _reject(
        self,
        reason: str,
        url: str | None = None,
    ) -> AdmissionDecision:
        self._rejections[reason] = self._rejections.get(reason, 0) + 1
        return AdmissionDecision(False, url, reason)

    def admit(
        self,
        url: str,
        depth: int,
    ) -> AdmissionDecision:
        """
        Reserve a URL for one crawl attempt.

        Depth 0 is the root page. Admitted URLs consume the page
        budget even if their subsequent HTTP request fails.
        """
        normalized = normalize_url(url)

        if normalized is None:
            return self._reject("invalid_url")

        if depth < 0:
            return self._reject("invalid_depth", normalized)

        if not is_within_scope(normalized, self.root_url):
            return self._reject("out_of_scope", normalized)

        if depth > self.max_depth:
            return self._reject("depth_limit", normalized)

        if normalized in self._seen_urls:
            return self._reject("duplicate", normalized)

        if self._admitted_count >= self.max_pages:
            return self._reject("page_limit", normalized)

        # Reserve before scheduling so duplicate links cannot
        # consume the budget multiple times.
        self._seen_urls.add(normalized)
        self._admitted_count += 1

        return AdmissionDecision(True, normalized, "accepted")
