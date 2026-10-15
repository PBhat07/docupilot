from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
import trafilatura
from bs4 import BeautifulSoup, Tag


# Try these selectors in order. Keep them configurable because
# documentation sites use different HTML layouts.
CONTENT_SELECTORS = (
    "main",
    "article",
    '[role="main"]',
    "div.document",          # Sphinx documentation
    ".theme-doc-markdown",   # Docusaurus
    ".markdown",
    ".md-content",           # Common documentation layout
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

EXPECTED_TERMS = (
    "install",
    "example",
    "configuration",
    "api",
    "return",
    "parameter",
)


@dataclass
class ExtractionResult:
    method: str
    text: str
    elapsed_seconds: float
    headings: int
    code_blocks: int
    tables: int
    links: int
    content_hash: str


def clean_text(value: str) -> str:
    """Normalize whitespace while retaining meaningful line boundaries."""
    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in value.splitlines()
    ]
    return "\n".join(line for line in lines if line)


def remove_boilerplate(soup: BeautifulSoup | Tag) -> None:
    """Remove common non-content elements in place."""
    for element in soup.select(", ".join(REMOVE_SELECTORS)):
        element.decompose()


def generic_beautifulsoup(html: str) -> ExtractionResult:
    start = time.perf_counter()

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    remove_boilerplate(soup)

    headings = len(soup.find_all(["h1", "h2", "h3", "h4"]))
    code_blocks = len(soup.find_all(["pre", "code"]))
    tables = len(soup.find_all("table"))
    links = len(soup.find_all("a", href=True))

    text = clean_text(soup.get_text("\n", strip=True))
    if title and title not in text:
        text = f"{title}\n{text}"

    elapsed = time.perf_counter() - start
    return make_result(
        "generic_beautifulsoup",
        text,
        elapsed,
        headings,
        code_blocks,
        tables,
        links,
    )


def container_beautifulsoup(html: str) -> ExtractionResult:
    start = time.perf_counter()

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""

    container: Tag | BeautifulSoup | None = None
    selector_used = "body_fallback"

    for selector in CONTENT_SELECTORS:
        candidate = soup.select_one(selector)
        if candidate and candidate.get_text(" ", strip=True):
            container = candidate
            selector_used = selector
            break

    if container is None:
        container = soup.body or soup

    # Copy the selected container so extraction doesn't mutate the original.
    content = BeautifulSoup(str(container), "html.parser")
    remove_boilerplate(content)

    headings = len(content.find_all(["h1", "h2", "h3", "h4"]))
    code_blocks = len(content.find_all(["pre", "code"]))
    tables = len(content.find_all("table"))
    links = len(content.find_all("a", href=True))

    text = clean_text(content.get_text("\n", strip=True))
    if title and title not in text:
        text = f"{title}\n{text}"

    elapsed = time.perf_counter() - start
    return make_result(
        f"container_beautifulsoup[{selector_used}]",
        text,
        elapsed,
        headings,
        code_blocks,
        tables,
        links,
    )


def trafilatura_extraction(html: str, url: str) -> ExtractionResult:
    start = time.perf_counter()

    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        include_links=True,
        include_formatting=True,
        favor_precision=True,
    ) or ""

    elapsed = time.perf_counter() - start

    # Trafilatura returns extracted text, not the original DOM.
    # These structural counts are therefore unavailable from its output.
    return make_result(
        "trafilatura",
        clean_text(text),
        elapsed,
        headings=0,
        code_blocks=0,
        tables=0,
        links=0,
    )


def make_result(
    method: str,
    text: str,
    elapsed: float,
    headings: int,
    code_blocks: int,
    tables: int,
    links: int,
) -> ExtractionResult:
    return ExtractionResult(
        method=method,
        text=text,
        elapsed_seconds=elapsed,
        headings=headings,
        code_blocks=code_blocks,
        tables=tables,
        links=links,
        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def main() -> None:
    url = input("Documentation URL: ").strip()
    if not url.startswith(("https://", "http://")):
        raise SystemExit("Please enter a valid HTTP or HTTPS URL.")

    terms_input = input(
        "Expected technical terms, comma-separated "
        "(press Enter for defaults): "
    ).strip()

    expected_terms = (
        [term.strip() for term in terms_input.split(",") if term.strip()]
        if terms_input
        else list(EXPECTED_TERMS)
    )

    print("\nFetching page once...")

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "DocuPilot-Extractor-Experiment/0.1"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise SystemExit(f"HTTP fetch failed: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.lower():
        raise SystemExit(f"Expected HTML but received: {content_type}")

    html = response.text
    final_url = str(response.url)

    print(f"Requested URL: {url}")
    print(f"Final URL:     {final_url}")
    print(f"HTTP status:   {response.status_code}")
    print(f"Content type:  {content_type}")
    print(f"Response size: {len(response.content):,} bytes")

    results = [
        generic_beautifulsoup(html),
        container_beautifulsoup(html),
        trafilatura_extraction(html, final_url),
    ]

    print("\n" + "=" * 78)
    print("EXTRACTION COMPARISON")
    print("=" * 78)

    for result in results:
        normalized = result.text.casefold()
        matched = [
            term for term in expected_terms
            if term.casefold() in normalized
        ]
        missing = [
            term for term in expected_terms
            if term.casefold() not in normalized
        ]

        print(f"\n--- {result.method} ---")
        print(f"Text characters:      {len(result.text):,}")
        print(f"Extraction time:      {result.elapsed_seconds:.4f} seconds")
        print(f"Headings in DOM:      {result.headings}")
        print(f"Code/pre elements:    {result.code_blocks}")
        print(f"Tables in DOM:        {result.tables}")
        print(f"Links in DOM:         {result.links}")
        print(f"Content hash:         {result.content_hash}")
        print(f"Expected terms found: {matched}")
        print(f"Expected terms absent:{missing}")

        # Simple indicators of possible navigation leakage.
        nav_terms = (
            "skip to content",
            "table of contents",
            "edit this page",
            "previous page",
            "next page",
            "sign in",
        )
        leakage = [term for term in nav_terms if term in normalized]
        print(f"Navigation indicators:{leakage}")

        preview = result.text[:700].replace("\n", " | ")
        print(f"Preview: {preview}")

    # Re-run the container approach to check deterministic output.
    repeat = container_beautifulsoup(html)
    first_container = results[1]

    print("\n" + "=" * 78)
    print("REPEATABILITY CHECK")
    print("=" * 78)
    print(f"Same content hash: {first_container.content_hash == repeat.content_hash}")
    print(f"Same extracted text: {first_container.text == repeat.text}")

    # Save all outputs for closer inspection if a result looks suspicious.
    for index, result in enumerate(results, start=1):
        filename = f"/tmp/docupilot_extraction_{index}.txt"
        with open(filename, "w", encoding="utf-8") as file:
            file.write(result.text)
        print(f"Saved {result.method} output to {filename}")


if __name__ == "__main__":
    main()
