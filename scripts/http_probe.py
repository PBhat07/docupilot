
from __future__ import annotations

import argparse
import time
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


def inspect_page(url: str, timeout: float = 15.0) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Provide a valid HTTP or HTTPS URL.")

    headers = {
        "User-Agent": "DocuPilotResearchBot/0.1 "
        "(documentation ingestion experiment)"
    }

    started = time.perf_counter()

    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers=headers,
    ) as client:
        with client.stream("GET", url) as response:
            status = response.status_code
            final_url = str(response.url)
            content_type = response.headers.get("content-type", "")
            response_bytes = 0
            body = bytearray()

            # Bound the experiment's response size.
            for chunk in response.iter_bytes():
                response_bytes += len(chunk)

                if response_bytes > 2 * 1024 * 1024:
                    raise ValueError("Response exceeded the 2 MiB probe limit.")

                body.extend(chunk)

    elapsed = time.perf_counter() - started

    print(f"Requested URL: {url}")
    print(f"Final URL:     {final_url}")
    print(f"HTTP status:   {status}")
    print(f"Content type:  {content_type}")
    print(f"Response size: {response_bytes:,} bytes")
    print(f"Fetch time:    {elapsed:.2f} seconds")

    if "text/html" not in content_type.lower():
        print("Assessment: response is not HTML; inspect content type.")
        return

    soup = BeautifulSoup(bytes(body), "html.parser")
        # Structural checks: what did the HTTP response actually contain?
    print(f"HTML body present: {soup.body is not None}")
    print(f"Main element:      {soup.find('main') is not None}")
    print(f"Article element:   {soup.find('article') is not None}")
    print(f"Headings found:    {len(soup.find_all(['h1', 'h2', 'h3']))}")

    expected_terms = [
        "ArgumentParser",
        "add_argument",
        "parse_args",
    ]
    raw_html = bytes(body).decode("utf-8", errors="replace")

    for term in expected_terms:
        print(f"Expected term {term!r}: {term in raw_html}")

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()

    text = " ".join(soup.stripped_strings)

    print(f"Page title:    {title or '(missing)'}")
    print(f"Text length:   {len(text):,} characters")
    print(f"Text preview:  {text[:500]}")
    print(
        "Assessment: inspect the preview and verify expected documentation "
        "sections before deciding whether browser rendering is needed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Documentation page URL")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    inspect_page(args.url, args.timeout)
