
from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx


DEFAULT_API_URL = "http://127.0.0.1:8000"


def print_json(data: Any) -> None:
    """Print structured API responses in a readable format."""
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def request_api(
    client: httpx.Client,
    method: str,
    path: str,
    **kwargs: Any,
) -> Any | None:
    """Make an API request and report expected errors to the user."""
    try:
        response = client.request(method, path, **kwargs)
        response.raise_for_status()

        if not response.content:
            print("Request completed successfully.")
            return None

        return response.json()

    except httpx.HTTPStatusError as exc:
        print(f"API error ({exc.response.status_code}):")
        try:
            print_json(exc.response.json())
        except ValueError:
            print(exc.response.text or "No error details returned.")
    except httpx.RequestError as exc:
        print(f"Could not connect to DocuPilot API: {exc}")
        print("Check that the API is running and the API URL is correct.")
    except ValueError:
        print("The API returned a response that was not valid JSON.")

    return None


def prompt_integer(
    message: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Prompt for an integer, using a default when input is blank."""
    while True:
        value = input(f"{message} [{default}]: ").strip()

        if not value:
            return default

        try:
            number = int(value)
            if minimum <= number <= maximum:
                return number
        except ValueError:
            pass

        print(f"Enter a whole number from {minimum} to {maximum}.")


def add_source(client: httpx.Client) -> None:
    url = input("Documentation URL: ").strip()
    if not url:
        print("URL cannot be empty.")
        return

    max_pages = prompt_integer("Maximum pages", 10, 1, 500)
    max_depth = prompt_integer("Maximum crawl depth", 1, 0, 10)

    result = request_api(
        client,
        "POST",
        "/sources",
        json={
            "url": url,
            "max_pages": max_pages,
            "max_depth": max_depth,
        },
    )
    if result is not None:
        print_json(result)


def list_sources(client: httpx.Client) -> None:
    result = request_api(
        client,
        "GET",
        "/sources",
        params={"limit": 50, "offset": 0},
    )
    if result is not None:
        print_json(result)


def inspect_source(client: httpx.Client) -> None:
    source_id = input("Source ID: ").strip()
    if not source_id:
        print("Source ID cannot be empty.")
        return

    result = request_api(client, "GET", f"/sources/{source_id}")
    if result is not None:
        print_json(result)


def refresh_source(client: httpx.Client) -> None:
    source_id = input("Source ID to refresh: ").strip()
    if not source_id:
        print("Source ID cannot be empty.")
        return

    max_pages = prompt_integer("Maximum pages", 10, 1, 500)
    max_depth = prompt_integer("Maximum crawl depth", 1, 0, 10)

    result = request_api(
        client,
        "POST",
        f"/sources/{source_id}/refresh",
        json={
            "max_pages": max_pages,
            "max_depth": max_depth,
        },
    )
    if result is not None:
        print_json(result)


def check_job(client: httpx.Client) -> None:
    job_id = input("Job ID: ").strip()
    if not job_id:
        print("Job ID cannot be empty.")
        return

    result = request_api(client, "GET", f"/jobs/{job_id}")
    if result is not None:
        print_json(result)



def ask_question(client: httpx.Client) -> None:
    """Run a multi-turn conversation until the user exits to the menu."""
    conversation_id = None

    print("Chat started. Type /new for a new conversation or /menu to exit.")

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nReturning to the main menu.")
            return

        if not question:
            continue

        if question.lower() in {"/menu", "/exit"}:
            print("Returning to the main menu.")
            return

        if question.lower() == "/new":
            conversation_id = None
            print("Started a new conversation.")
            continue

        payload = {"question": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        result = request_api(
            client,
            "POST",
            "/questions",
            json=payload,
        )

        if not isinstance(result, dict):
            continue

        # Reuse the ID returned by the API on subsequent turns.
        returned_id = result.get("conversation_id")
        if returned_id:
            conversation_id = returned_id

        answer = result.get("answer")
        if answer:
            print(f"\nDocuPilot:\n{answer}")

        citations = result.get("citations", [])
        if citations:
            print("\nSources:")
            for citation in citations:
                print(
                    f"- [{citation.get('ref', '?')}] "
                    f"{citation.get('title', 'Untitled')}"
                )
                if citation.get("url"):
                    print(f"  {citation['url']}")

        if result.get("abstained"):
            print("\nDocuPilot could not verify a reliable answer.")

        # Display other retrieved pages for further investigation.
        sources = result.get("sources", [])
        cited_urls = {
            item.get("url") for item in citations if item.get("url")
        }
        related = [
            source for source in sources
            if source.get("url") and source["url"] not in cited_urls
        ]

        if related:
            print("\nRelated documentation:")
            for source in related:
                print(
                    f"- {source.get('title', 'Documentation')}: "
                    f"{source['url']}"
                )



def main() -> None:
    api_url = os.getenv("DOCUPILOT_API_URL", DEFAULT_API_URL).rstrip("/")

    actions = {
        "1": ("Add a documentation source", add_source),
        "2": ("List sources", list_sources),
        "3": ("Inspect a source", inspect_source),
        "4": ("Refresh a source", refresh_source),
        "5": ("Check ingestion job status", check_job),
        "6": ("Ask a question", ask_question),
    }

    print("DocuPilot — Technical Documentation Assistant")
    print(f"API: {api_url}")

    try:
        with httpx.Client(
            base_url=api_url,
            timeout=httpx.Timeout(120.0, connect=5.0),
        ) as client:
            while True:
                print("\nMain menu")
                for key, (label, _) in actions.items():
                    print(f"{key}. {label}")
                print("0. Exit")

                choice = input("\nChoose an option: ").strip()

                if choice == "0":
                    print("Goodbye.")
                    break

                action = actions.get(choice)
                if action is None:
                    print("Invalid option. Choose one of the listed numbers.")
                    continue

                print(f"\n--- {action[0]} ---")
                action[1](client)

    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye.")


if __name__ == "__main__":
    main()
