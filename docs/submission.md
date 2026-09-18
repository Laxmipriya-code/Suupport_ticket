# Assessment submission and walkthrough

[README](../README.md) · [Architecture](architecture.md) · [Evaluation](evaluation.md)

This checklist follows the supplied **End-to-End AI System Sprint — Technical Assessment — AI Engineer Role**, DOTMappers IT Pvt. Ltd. The deliverable is a GitHub repository containing the working Python application, both interfaces, documentation and reproducible dependencies.

## Deliverables and technical constraints

| Brief requirement | Submission evidence |
|---|---|
| Ingest the provided 500-row CSV and make it queryable | `data/support_tickets.csv`, `src/support_ticket_ai/dataset.py`, `query.py`; original bytes retained |
| Answer natural-language data questions with an LLM | `agent/execution.py`, `agent/prompts.py`, `agent/tools.py`; Groq and LangChain, typed tool arguments and final outcomes |
| Flag long resolutions and unresolved priority tickets older than 24 hours | `anomalies.py`; IQR and age rules with threshold, excess and explanation |
| REST API with NL query, anomalies and health endpoints | `api.py`, `main.py`; `/api/v1/query`, `/api/v1/anomalies`, `/health` plus stats/search |
| Minimal UI covering the functionality | `ui/app.py`; Overview, Questions and Anomalies tabs, tables and charts |
| README setup, architecture, model/tools, query outputs and limitations | All five sections are in `README.md`, with linked design/evaluation details |
| `requirements.txt` or equivalent | `pyproject.toml` and `uv.lock`; `uv sync --locked` |
| Python only | Python API, analytics, LLM orchestration, UI and launcher |
| No paid APIs/services; zero-cost local run | Local services plus a Groq Free Plan key; free quota/network needed for NL queries; no paid fallback |
| Start the system with a single command | After setup: `uv run --locked python run.py` |
| GitHub link and README handoff | Repository publication and the final link must be completed before sending |

**API/UI interpretation:** Section 2 says both are required; Section 4 item 2 says API “OR” UI. Both are implemented to satisfy the explicit Section 2 requirement.

## Weighted evaluation criteria

| Criterion | Weight | What to inspect or demonstrate |
|---|---:|---|
| Functionality | 30% | 500 records loaded, the five sample question intents, both anomaly rules, API and UI working together |
| Architecture and design | 25% | [Component decisions](architecture.md#component-decisions), request lifecycle, shared calculations and bounded model context |
| Code quality | 20% | Small modules, typed schemas, validation, safe error handling, offline tests, Ruff/mypy and CI |
| LLM integration quality | 15% | Prompt rules, native tools, structured outcomes, evidence IDs, clarification/unsupported behavior and honest live-evaluation status |
| README and documentation | 10% | Reproducible setup, example outputs, date assumptions, known limitations and supporting docs |

The brief does **not** specify a 90% query-accuracy pass mark. The 36/40 target in the evaluation runner is this project's own regression target. The submitted live evidence is partial, with all five example intents recorded as passed; see [evaluation status](evaluation.md#delivery-status-and-remaining-verification).

## Before sending

- [x] Include the unchanged dataset, source, UI, tests, `.env.example`, dependency manifest/lock, CI configuration and documentation.
- [x] Provide one command that starts and stops both services.
- [x] Include `evaluation-results.json` in the submission; it is no longer excluded by `.gitignore`.
- [x] Show sample outputs with the fixed historical reference time and explain the estimated resolution timestamp.
- [x] Keep the incomplete live run visible: 12 passed, 7 quota-blocked, 21 not attempted; no claim of full live acceptance.
- [x] Pass local verification: 104 offline tests, Ruff lint/format, mypy, dependency lock validation, document links and dataset/report integrity.
- [ ] Create or update the GitHub repository with these files and verify the evaluator can access it.
- [ ] Confirm a fresh checkout can run `uv sync --locked` and `uv run --locked python run.py` with a Free Plan key. Review the repository's GitHub Actions result after publication.
- [ ] Confirm the candidate name, final repository URL, assessment receipt time and corresponding deadline.
- [ ] Send the repository link using the required address and subject below.

Commit the **contents of `support-ticket-ai/` as the repository root** so the README and `.github/workflows/ci.yml` are in the expected locations. Do not include `.env`, credentials, `.venv`, caches, downloaded assessment archives or generated local evaluation files. Keep the reviewed `evaluation-results.json`; scratch reports ending in `.local.json` are ignored. The separately supplied submission ZIP is a convenience copy; the brief requires a GitHub link.

## Handoff details

- **Recipient:** `RajathKumar@dotmappers.in`
- **Subject:** `[AI Engineer Assessment] — Your Name`, replacing `Your Name` with the candidate's name.
- **Required content:** the accessible GitHub repository link; point the evaluator to the README for setup.
- **Deadline:** 48 hours from receipt of the assessment document. The receipt timestamp is not established by the project files; calculate the actual deadline from the original message.
- **Post-submission:** a 30-minute architecture walkthrough will be scheduled.

The local handoff does not establish that a GitHub repository has been published or that an email has been sent.

## Thirty-minute architecture walkthrough

| Minutes | Demonstration and discussion |
|---|---|
| 0–3 | State the support-ticket problem and four requirements. Start both services with one command and show `/health` and the UI. |
| 3–8 | Show the 500-row overview and unchanged CSV. Set **Historical demonstration**. Run the open-ticket count and Technical rating questions; inspect sample counts and evidence. |
| 8–13 | Show the March agent tie, Critical 12-hour target and weekly resolution anomalies. Explain recorded statuses, estimated resolution dates and the fixed reference time. |
| 13–18 | Trace API → agent → typed tool → deterministic calculation → result ID → rendered answer. Explain why pandas and bounded tools fit the dataset. |
| 18–22 | Show both anomaly rules. Explain the global 48.15-hour IQR cutoff, the >24-hour priority rule and the 28 separate data-quality warnings. |
| 22–26 | Review the offline suite and partial live report. Show a validation error or ambiguous/unsupported request. Explain quota handling, deadlines and limits. |
| 26–30 | Discuss limitations, next improvements and scaling trade-offs; take questions. |

Before the call, verify available free quota. If a live question is blocked, show the recorded evidence and deterministic anomaly/search endpoints, clearly identifying which checks are live and which are saved.

With more time, first complete the live regression suite and review failed interpretations. For a larger or changing dataset, move storage/querying to a database while retaining validated tool contracts; add authenticated access and ingestion controls before public deployment. True historical reporting requires status events and measured resolution timestamps that the supplied CSV does not contain.
