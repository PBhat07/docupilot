
from __future__ import annotations

import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


PAGES = {
    "/robots.txt": (
        "text/plain",
        "User-agent: *\nAllow: /\n",
    ),
    "/index.html": (
        "text/html",
        """
        <html>
          <head><title>Fixture Documentation</title></head>
          <body>
            <nav>Home | Navigation</nav>
            <main>
              <h1>Fixture Documentation</h1>
              <p>Welcome to the local documentation fixture.</p>
              <a href="/guide.html#intro">Guide</a>
              <a href="/guide.html#examples">Guide duplicate</a>
              <a href="/reference.html">Reference</a>
              <a href="http://example.org/outside">External link</a>
            </main>
          </body>
        </html>
        """,
    ),
    "/guide.html": (
        "text/html",
        """
        <html>
          <head><title>Guide</title></head>
          <body>
            <main>
              <h1>Guide</h1>
              <p>Follow these steps to install the package.</p>
              <pre><code>pip install example-package</code></pre>
              <a href="/deep/page.html">Deep page</a>
            </main>
          </body>
        </html>
        """,
    ),
    "/reference.html": (
        "text/html",
        """
        <html>
          <head><title>Reference</title></head>
          <body>
            <main>
              <h1>Reference</h1>
              <p>The reference describes the public API.</p>
            </main>
          </body>
        </html>
        """,
    ),
    "/deep/page.html": (
        "text/html",
        """
        <html>
          <head><title>Deep Page</title></head>
          <body>
            <main><h1>Deep Page</h1><p>Should not be crawled.</p></main>
          </body>
        </html>
        """,
    ),
}


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        entry = PAGES.get(path)

        if entry is None:
            self.send_error(404)
            return

        content_type, body = entry
        payload = body.encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def main() -> None:
    output_dir = Path("/app/data/crawls")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "fixture.jsonl"

    server = ThreadingHTTPServer(("127.0.0.1", 8765), FixtureHandler)
   

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        root_url = "http://127.0.0.1:8765/index.html"

        command = [
            "scrapy", "runspider",
            "src/docupilot/crawler/spiders/docs_spider.py",
            "-a", f"start_url={root_url}",
            "-a", "max_pages=10",
            "-a", "max_depth=1",
            "-O", f"{output_file}:jsonlines",
        ]

        print("Running local fixture crawl...")
        result = subprocess.run(
            command,
            cwd="/app",
            text=True,
            capture_output=True,
            timeout=60,
        )

        print(result.stdout)
        print(result.stderr)

        if result.returncode != 0:
            raise RuntimeError(
                f"Scrapy exited with code {result.returncode}"
            )

        records = [
            json.loads(line)
            for line in output_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        pages = [r for r in records if r["record_type"] == "page"]
        errors = [r for r in records if r["record_type"] == "crawl_error"]

        urls = [r["canonical_url"] for r in pages]
        paths = {urlsplit(url).path for url in urls}

        print("\n--- Fixture validation ---")
        print("Page records:", len(pages))
        print("Error records:", len(errors))
        print("Unique URLs:", len(set(urls)))
        print("Crawled paths:", sorted(paths))
        print("Depths:", sorted({r["depth"] for r in pages}))

        assert len(pages) == 3, f"Expected 3 pages, got {len(pages)}"
        assert len(set(urls)) == 3, "Duplicate canonical URLs found"
        assert "/index.html" in paths
        assert "/guide.html" in paths
        assert "/reference.html" in paths
        assert "/deep/page.html" not in paths
        assert all(r["depth"] <= 1 for r in pages)
        assert all(r["text"].strip() for r in pages)
        assert all(r["content_hash"] for r in pages)

        guide = next(r for r in pages if urlsplit(r["canonical_url"]).path == "/guide.html")
        assert "pip install example-package" in guide["text"]
        assert "Navigation" not in pages[0]["text"]

        print("\nPASS: fixture crawl behaved as expected.")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
    
'''We'll now test whether the spider actually handles duplicate links, external links and depth limits correctly. 
This test will run a tiny HTTP server inside the same Docker container, so we don't need another service or an 
internet connection for the fixture pages.'''