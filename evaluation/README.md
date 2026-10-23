# DocuPilot Evaluation

## Purpose

DocuPilot includes an **optional preliminary evaluation** to compare RAG answer generation with reranking disabled and enabled.

The evaluation records generated answers, retrieved evidence, and end-to-end latency. It is intended as a small engineering experiment, not a comprehensive benchmark or proof that reranking always improves quality.

## 1. Run the Evaluation

Run these commands from the project root.

**Build the evaluation image** (when the evaluation code, dependencies, or Docker configuration changes):

```bash
docker compose --profile evaluation build evaluation
```

**Run the evaluation:**

```bash
docker compose --profile evaluation run --rm evaluation
```

This runs the evaluation service's configured default command, which should execute the evaluation runner.

**Generate the plots and review report:**

```bash
docker compose --profile evaluation run --rm evaluation python -m evaluation.plot_results
```

The evaluation service must have access to the required environment variables, application configuration, and running MongoDB and Weaviate services.

The `--rm` flag removes the temporary evaluation container after execution. Ensure the output directory is mounted as a Docker volume if results need to persist on the host.

## 2. Output Files

Results are written to `data/evaluation/results/`, unless overridden by the evaluation configuration.

| File | Contents |
|---|---|
| `results.json` | Raw evaluation records, including questions, reference answers, generated answers, errors, latency, retrieved chunks, citations, and run metadata. |
| `manual_review.csv` | Side-by-side reranking-off/on answers and retrieved evidence, latency values, and fields for recording a manual quality verdict and notes. |
| `visuals/latency_by_question.png` | Paired end-to-end latency comparison. Only questions with successful answers in both configurations are included. |
| `findings.md` | Run summary, success counts, paired latency observations, interpretation guidance, and evaluation limitations. |

The manual-review CSV includes `quality_winner` and `review_notes`. These fields are intended for human assessment; the evaluation script does not automatically determine which answer is better.

**Note:** Rerunning the evaluation overwrites `results.json`. The plot-generation command also regenerates the review CSV and findings report. Back up any results or manual annotations you want to preserve.

## 3. Dataset

The benchmark questions are defined in `evaluation/dataset.json`.

Each entry can include:

- `id`: unique question identifier, such as `Q001`.
- `category`: question type, such as `basic_api` or `unsupported_question`.
- `question`: the question sent to DocuPilot.
- `reference`: expected answer used for manual comparison.
- `source_ids`: optional restriction to particular indexed sources; use `null` when unrestricted.
- `expected_evidence`: key facts or evidence expected in a satisfactory answer.
- `answerable`: whether the available documentation is expected to support an answer.
- `evaluation_focus`: the specific behavior the question is intended to examine.

To extend the dataset, add entries using the same structure, assign unique IDs, and keep reference answers grounded in the indexed documentation. Include straightforward questions, questions combining multiple concepts, similar API terms, and questions that the documentation cannot answer.

The current five-question dataset is deliberately small. Changes to the dataset affect subsequent runs, so preserve the dataset version associated with each reported result.

## 4. How to Interpret the Results

- **Latency:** compares the time taken for successful paired answers. A lower value indicates a faster response in that run.
- **Retrieved evidence:** inspect chunk IDs, previews, and citations to understand which evidence was used. A changed ranking does not automatically mean the evidence is better.
- **Answer quality:** compare each answer with its reference and retrieved evidence, then manually record `off`, `on`, `tie`, or `unclear` in `quality_winner`.

The current experiment is exploratory. With only five questions, results may be sensitive to question selection, model variability, and API conditions.

## 5. Limitations and Future Work

A larger evaluation is needed to determine whether reranking improves retrieval and answer quality enough to justify its additional computational cost for the intended deployment workload.

Future work could expand the dataset, manually label relevant retrieved chunks, compare retrieval-ranking quality and final answers, repeat latency measurements, and evaluate different candidate and final-context limits.

Based on those results, reranking can remain enabled or be disabled when its benefit does not justify the overhead. Resource constraints limited the current project to a small preliminary comparison; no claim is made that reranking is universally beneficial.

## 6. Optional Component

Keep the evaluation scripts and dataset for reproducibility, and run them when investigating retrieval changes, reranking behavior, or deployment performance. Evaluation outputs can remain local and need not be committed to the repository unless selected results are included as documented project evidence.