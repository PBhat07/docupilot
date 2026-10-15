
from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

URL = "https://docs.python.org/3/library/argparse.html"

response = httpx.get(
    URL,
    timeout=15.0,
    follow_redirects=True,
)
response.raise_for_status()

soup = BeautifulSoup(response.content, "html.parser")

print("Title:", soup.title.get_text(" ", strip=True) if soup.title else "(none)")

print("\nElements with useful IDs:")
for element_id in [
    "content",
    "main-content",
    "document",
    "body",
    "furo-main-content",
]:
    element = soup.find(id=element_id)
    if element:
        print(
            f"id={element_id!r}, "
            f"tag={element.name!r}, "
            f"text_length={len(element.get_text(' ', strip=True)):,}"
        )

print("\nLargest candidate content containers:")
candidates = soup.find_all(["div", "section", "main", "article"])
candidates.sort(
    key=lambda element: len(element.get_text(" ", strip=True)),
    reverse=True,
)

for element in candidates[:8]:
    print(
        f"tag={element.name}, "
        f"id={element.get('id')}, "
        f"class={element.get('class')}, "
        f"text_length={len(element.get_text(' ', strip=True)):,}"
    )
