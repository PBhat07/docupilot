
from pymongo import MongoClient

from docupilot.indexing.chunker import chunk_page


MONGODB_URI = "mongodb://mongodb:27017"
DATABASE_NAME = "docupilot"
MAX_PREVIEWS = 5


def main():
    with MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=5000,
    ) as client:
        db = client[DATABASE_NAME]
        pages = list(db.pages.find({"text": {"$type": "string"}}))

    if not pages:
        print("No pages with text found in MongoDB.")
        return

    # Inspect the longest page first so we can evaluate splitting.
    pages.sort(key=lambda page: len(page["text"]), reverse=True)
    page = pages[0]

    print("=" * 70)
    print("SOURCE PAGE")
    print("=" * 70)
    print(f"Title: {page.get('title', 'Untitled')}")
    print(f"URL: {page.get('canonical_url', 'Unknown')}")
    print(f"Content hash: {page.get('content_hash', 'Unknown')}")
    print(f"Original text length: {len(page['text'])} characters")
    print(f"Headings: {page.get('headings', [])}")
    print(f"Pages available: {len(pages)}")

    chunks = chunk_page(page)

    print("\n" + "=" * 70)
    print("CHUNK SUMMARY")
    print("=" * 70)
    print(f"Chunks generated: {len(chunks)}")

    if not chunks:
        print("This page produced no chunks.")
        return

    for chunk in chunks[:MAX_PREVIEWS]:
        print("\n" + "-" * 70)
        print(f"Chunk index: {chunk['chunk_index']}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Character count: {len(chunk['chunk_text'])}")
        print(f"Source URL: {chunk['canonical_url']}")
        print(f"Content hash: {chunk['content_hash']}")
        print("Text preview:")
        print(chunk["chunk_text"][:300])

    if len(chunks) > MAX_PREVIEWS:
        print(f"\n... {len(chunks) - MAX_PREVIEWS} more chunks not displayed.")


if __name__ == "__main__":
    main()

# The script:
# - Reads pages from MongoDB without modifying them.
# - Selects the longest page containing text.
# - Runs your existing chunk_page() function.
# - Prints chunk IDs, lengths, source metadata, and short previews.
# - Limits output to five chunks so your terminal stays readable.