
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

from pymongo import MongoClient
from scrapy import signals


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_source_id(root_url: str) -> str:
    """Stable identifier for a registered documentation root."""
    return hashlib.sha256(root_url.encode("utf-8")).hexdigest()


class MongoPersistencePipeline:
    def __init__(self, mongo_uri: str, database_name: str) -> None:
        self.client = MongoClient(
            mongo_uri,
            serverSelectionTimeoutMS=5000,
        )
        self.db = self.client[database_name]

    @classmethod
    def from_crawler(cls, crawler):
        pipeline = cls(
            mongo_uri=crawler.settings.get(
                "MONGODB_URI",
                os.getenv("MONGODB_URI", "mongodb://mongodb:27017"),
            ),
            database_name=crawler.settings.get(
                "MONGODB_DATABASE",
                os.getenv("MONGODB_DATABASE", "docupilot"),
            ),
        )

        crawler.signals.connect(
            pipeline.spider_closed,
            signal=signals.spider_closed,
        )
        return pipeline

    def open_spider(self, spider) -> None:
        # Fail clearly if MongoDB is unavailable.
        self.client.admin.command("ping")

        self.db.pages.create_index(
            [("source_id", 1), ("canonical_url", 1)],
            unique=True,
        )
        self.db.page_versions.create_index(
            [
                ("source_id", 1),
                ("canonical_url", 1),
                ("content_hash", 1),
            ],
            unique=True,
        )
        self.db.crawls.create_index("source_id")
        self.db.crawl_errors.create_index("crawl_id")

        self.source_id = make_source_id(spider.root_url)
        self.crawl_id = spider.crawl_id

        self.db.sources.update_one(
            {"_id": self.source_id},
            {
                "$set": {
                    "root_url": spider.root_url,
                    "updated_at": utc_now(),
                },
                "$setOnInsert": {
                    "created_at": utc_now(),
                    "status": "registered",
                },
            },
            upsert=True,
        )

        self.db.crawls.insert_one(
            {
                "_id": self.crawl_id,
                "source_id": self.source_id,
                "root_url": spider.root_url,
                "status": "running",
                "started_at": utc_now(),
                "finished_at": None,
                "max_pages": spider.policy.max_pages,
                "max_depth": spider.policy.max_depth,
                "pages_seen": 0,
                "errors_seen": 0,
            }
        )

    def process_item(self, item, spider):
        record = dict(item)
        record_type = record.get("record_type")

        if record_type == "page":
            self._save_page(record)
        elif record_type == "crawl_error":
            self._save_error(record)
        else:
            spider.logger.warning(
                "Ignoring unknown record type: %r",
                record_type,
            )

        return item

    def _save_page(self, record: dict) -> None:
        now = utc_now()

        page_key = {
            "source_id": self.source_id,
            "canonical_url": record["canonical_url"],
        }

        page_data = {
            key: value
            for key, value in record.items()
            if key != "record_type"
        }
        page_data["source_id"] = self.source_id
        page_data["last_seen_at"] = now

        # Current page: update content when it changes.
        self.db.pages.update_one(
            page_key,
            {
                "$set": page_data,
                "$setOnInsert": {"first_seen_at": now},
            },
            upsert=True,
        )

        # Version history: retain each distinct content hash.
        version_data = dict(page_data)
        version_data["version_recorded_at"] = now

        self.db.page_versions.update_one(
            {
                **page_key,
                "content_hash": record["content_hash"],
            },
            {"$setOnInsert": version_data},
            upsert=True,
        )

        self.db.crawls.update_one(
            {"_id": self.crawl_id},
            {"$inc": {"pages_seen": 1}},
        )

    def _save_error(self, record: dict) -> None:
        error_data = {
            **record,
            "source_id": self.source_id,
            "recorded_at": utc_now(),
        }

        self.db.crawl_errors.insert_one(error_data)

        self.db.crawls.update_one(
            {"_id": self.crawl_id},
            {"$inc": {"errors_seen": 1}},
        )

    def spider_closed(self, spider, reason) -> None:
        try:
            stats = spider.crawler.stats.get_stats()

            self.db.crawls.update_one(
                {"_id": self.crawl_id},
                {
                    "$set": {
                        "status": (
                            "completed" if reason == "finished" else "stopped"
                        ),
                        "close_reason": reason,
                        "finished_at": utc_now(),
                        "scrapy_stats": {
                            key: value
                            for key, value in stats.items()
                            if isinstance(value, (str, int, float, bool))
                        },
                    }
                },
            )
        finally:
            self.client.close()
