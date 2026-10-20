
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from urllib.parse import urljoin

import scrapy
from scrapy import Request
from scrapy.http import Response
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.python.failure import Failure

from docupilot.crawler.extractor import extract_page
from docupilot.crawler.policy import CrawlPolicy
from docupilot.crawler.url_utils import normalize_url


class DocsSpider(scrapy.Spider):
    name = "docupilot_docs"

    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_TIMEOUT": 20,
        "DOWNLOAD_MAXSIZE": 5 * 1024 * 1024,
        "RETRY_ENABLED": True,
        "RETRY_TIMES": 2,
        "CONCURRENT_REQUESTS": 4,
        "DOWNLOAD_DELAY": 0.25,
        "USER_AGENT": "DocuPilot/0.1 (+documentation indexing)",
        "LOG_LEVEL": "INFO",
        # MongoDB persistence
        "ITEM_PIPELINES": {
            "docupilot.crawler.pipelines.MongoPersistencePipeline": 300,
        },
        "MONGODB_URI": "mongodb://mongodb:27017",
        "MONGODB_DATABASE": "docupilot",
    }

    def __init__(
        self,
        start_url: str | None = None,
        max_pages: str = "10",
        max_depth: str = "1",
        crawl_id: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not start_url:
            raise ValueError("Provide start_url when launching the spider")

        root_url = normalize_url(start_url)
        if root_url is None:
            raise ValueError("start_url must be a valid HTTP(S) URL")

        self.policy = CrawlPolicy(
            root_url=root_url,
            max_pages=int(max_pages),
            max_depth=int(max_depth),
        )

        self.root_url = root_url
        self.crawl_id = crawl_id or uuid.uuid4().hex

    async def start(self):
        """Admit and schedule the root URL."""
        decision = self.policy.admit(self.root_url, depth=0)

        if not decision.accepted:
            self.logger.error("Root URL rejected: %s", decision.reason)
            return

        yield self.make_request(decision.url, depth=0)

    def make_request(self, url: str, depth: int) -> Request:
        return Request(
            url=url,
            callback=self.parse,
            errback=self.handle_failure,
            meta={
                "crawl_depth": depth,
                # We inspect redirects ourselves before following them.
                "dont_redirect": True,
                "handle_httpstatus_all": True,
            },
        )

    def parse(self, response: Response):
        depth = response.meta["crawl_depth"]

        # Validate redirects before issuing another request.
        if 300 <= response.status < 400:
            location = response.headers.get("Location")

            if not location:
                yield self.error_record(
                    response.url,
                    depth,
                    f"redirect_without_location:{response.status}",
                )
                return

            target = urljoin(
                response.url,
                location.decode("latin-1"),
            )

            decision = self.policy.admit(target, depth=depth)

            if not decision.accepted:
                yield self.error_record(
                    response.url,
                    depth,
                    f"redirect_rejected:{decision.reason}",
                )
                return

            yield self.make_request(decision.url, depth=depth)
            return

        if response.status >= 400:
            yield self.error_record(
                response.url,
                depth,
                f"http_status:{response.status}",
            )
            return

        content_type = response.headers.get(
            "Content-Type", b""
        ).decode("latin-1").lower()

        if "text/html" not in content_type:
            yield self.error_record(
                response.url,
                depth,
                f"unsupported_content_type:{content_type}",
            )
            return

        page = extract_page(response.text, response.url)

        yield {
            "record_type": "page",
            "crawl_id": self.crawl_id,
            "root_url": self.root_url,
            "requested_url": response.request.url,
            "final_url": response.url,
            "canonical_url": normalize_url(response.url),
            "depth": depth,
            "http_status": response.status,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "title": page.title,
            "text": page.text,
            "headings": list(page.headings),
            "code_blocks": list(page.code_blocks),
            "links": list(page.links),
            "content_hash": page.content_hash,
            "extraction_method": page.extraction_method,
        }

        # Admit each link before scheduling it. The policy reserves
        # its budget slot, preventing duplicate links from being queued.
        if depth >= self.policy.max_depth:
            return

        for link in page.links:
            normalized = normalize_url(link, base_url=response.url)

            if normalized is None:
                continue

            decision = self.policy.admit(
                normalized,
                depth=depth + 1,
            )

            if decision.accepted:
                yield self.make_request(
                    decision.url,
                    depth=depth + 1,
                )

    def error_record(
        self,
        url: str,
        depth: int,
        reason: str,
    ) -> dict:
        return {
            "record_type": "crawl_error",
            "crawl_id": self.crawl_id,
            "url": url,
            "depth": depth,
            "reason": reason,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def handle_failure(self, failure: Failure):
        request = failure.request
        depth = request.meta.get("crawl_depth", 0)

        if failure.check(HttpError):
            status = failure.value.response.status
            reason = f"http_status:{status}"
        else:
            reason = f"request_failed:{failure.getErrorMessage()}"

        self.logger.warning(
            "Request failed: %s (%s)",
            request.url,
            reason,
        )

        yield self.error_record(request.url, depth, reason)
