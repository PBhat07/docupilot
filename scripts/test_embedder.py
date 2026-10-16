
import numpy as np

from docupilot.indexing.embedder import Embedder


def main():
    embedder = Embedder(device="cpu")

    texts = [
        "Install the package using pip.",
        "Run pip install example-package to install the package.",
        "Configure the database connection string.",
    ]

    vectors = embedder.embed_texts(texts)

    print(f"Model: {embedder.model_name}")
    print(f"Number of vectors: {len(vectors)}")
    print(f"Vector dimension: {len(vectors[0])}")

    norms = [np.linalg.norm(vector) for vector in vectors]
    print(f"Vector norms: {norms}")

    similarity = np.dot(vectors[0], vectors[1])
    print(f"Similarity of related sentences: {similarity:.4f}")

    assert len(vectors) == len(texts)
    assert all(len(vector) == 384 for vector in vectors)
    assert np.allclose(norms, 1.0, atol=1e-5)
    assert -1.0 <= similarity <= 1.0

    print("Embedding smoke test passed.")


if __name__ == "__main__":
    main()
