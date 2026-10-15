
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString, Tag


CONTENT_SELECTORS = (
    "main",
    "article",
    '[role="main"]',
    "div.document",
    ".theme-doc-markdown",
    ".markdown",
    ".md-content",
    ".content",
)

REMOVE_SELECTORS = (
    "script",
    "style",
    "noscript",
    "svg",
    "nav",
    "footer",
    "header",
    "aside",
    '[role="navigation"]',
    '[aria-hidden="true"]',
    ".breadcrumbs",
    ".breadcrumb",
    ".sidebar",
    ".table-of-contents",
    ".toc",
    ".theme-doc-toc",
    ".edit-this-page",
    ".pagination-nav",
)

BLOCK_TAGS = {
    "p", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "pre", "blockquote", "dt", "dd",
}

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]


@dataclass(frozen=True)
class ExtractedPage:
    title: str
    text: str
    headings: tuple[str, ...]
    code_blocks: tuple[str, ...]
    links: tuple[str, ...]
    content_hash: str
    extraction_method: str


def normalize_text(value: str) -> str:
    """Normalize whitespace while preserving meaningful block boundaries."""
    lines = []
    for line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", line)
        line = re.sub(r"(?<=\w)\s+\.\s+(?=\w)", ".", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _table_to_text(table: Tag) -> str:
    """Represent an HTML table as pipe-separated rows."""
    rows = []

    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if cells:
            values = [
                re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
                for cell in cells
            ]
            rows.append(" | ".join(values))

    return "\n".join(rows)


def _extract_blocks(node: Tag) -> list[str]:
    """Extract semantic blocks without flattening code examples."""
    blocks: list[str] = []
    inline_buffer: list[str] = []

    def flush_inline() -> None:
        if inline_buffer:
            value = normalize_text("".join(inline_buffer))
            if value:
                blocks.append(value)
            inline_buffer.clear()

    for child in node.children:
        if isinstance(child, NavigableString):
            inline_buffer.append(str(child))
            continue

        if not isinstance(child, Tag):
            continue

        if child.name == "table":
            flush_inline()
            value = _table_to_text(child)
            if value:
                blocks.append(value)
            continue

        if child.name == "pre":
            flush_inline()
            # Preserve indentation and line breaks in code examples.
            code = child.get_text("", strip=False).strip("\n")
            if code.strip():
                blocks.append(code)
            continue

        if child.name in BLOCK_TAGS:
            flush_inline()
            value = normalize_text(child.get_text(" ", strip=True))
            if value:
                blocks.append(value)
            continue

        # Containers such as section, div and span may wrap useful content.
        inline_buffer.append(" ")
        flush_inline()
        blocks.extend(_extract_blocks(child))
        inline_buffer.append(" ")

    flush_inline()
    return blocks


def extract_page(html: str, page_url: str) -> ExtractedPage:
    """Extract useful documentation content and source metadata from HTML."""
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    container: Tag | None = None
    method = "body_fallback"

    for selector in CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and candidate.get_text(" ", strip=True):
            container = candidate
            method = f"selector:{selector}"
            break

    if container is None:
        container = soup.body or soup

    # Work on a copy, leaving the original document untouched.
    content = BeautifulSoup(str(container), "html.parser")

    for element in content.select(", ".join(REMOVE_SELECTORS)):
        element.decompose()

    headings = tuple(
        heading.get_text(" ", strip=True)
        for heading in content.find_all(HEADING_TAGS)
        if heading.get_text(" ", strip=True)
    )

    code_blocks = tuple(
        element.get_text("", strip=False).strip("\n")
        for element in content.find_all("pre")
        if element.get_text("", strip=True)
    )

    links = tuple(
        dict.fromkeys(
            urljoin(page_url, anchor["href"])
            for anchor in content.find_all("a", href=True)
            if anchor["href"].strip()
        )
    )

    text = normalize_text("\n".join(_extract_blocks(content)))

    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    return ExtractedPage(
        title=title,
        text=text,
        headings=headings,
        code_blocks=code_blocks,
        links=links,
        content_hash=content_hash,
        extraction_method=method,
    )
