# Verification and evaluation

[README](../README.md) · [Architecture](architecture.md) · [Submission](submission.md)

There are two distinct claims to verify: deterministic application correctness and live LLM tool-selection accuracy. Mocked tests establish the first. Only a run against Groq can establish the second.

## Reproducible checks

```sh
uv sync --locked
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked pytest -m 'not live'
```

CI runs lint/format checks, application and launcher type checks, and the offline suite. It needs no model key and never calls Groq. Run commands from the project root. Dependencies are fixed by `uv.lock` on Python 3.12.

The suite covers:

- Dataset totals, source fingerprint, status counts, ratings and rating sample counts, tied estimated-resolution rankings, duration statistics, 28 timing inconsistencies, and malformed ingestion.
- Field/operator/metric validation, AND/OR filtering, literal search, nulls, no-match populations, zero-denominator rates, zero-baseline comparisons, percentile interpolation, ranking, and immutability of source data.
- Relative/explicit time windows, Monday week boundaries, timezone conversion, estimated timestamps, zero/null trend buckets, coverage disclosures, pending/unknown time targets, exact threshold equality, and insufficient anomaly baselines.
- All nine real LangChain tool wrappers with hidden runtime context, result IDs, compact model envelopes, total output budgets, rejected unknown tools, argument corrections, invalid IDs, execution limits, and concurrent request isolation.
- The actual ChatGroq adapter and HTTP payload shape using a mocked HTTP transport. Captured requests prove no full CSV, raw issue summaries, credentials-as-tool-arguments, or backend result store enters model messages. Provider retries, 429, authentication failures, deadlines, and parser errors are bounded and sanitized.
- FastAPI response shapes, schema validation, no-key operation, deterministic anomalies, result pagination without model calls, and safe error envelopes.
- Real launcher startup of FastAPI and Streamlit on available localhost ports, startup from another working directory, clean shutdown, and cleanup after an occupied-port failure. These tests open local sockets but make no model requests.
- Streamlit overview and anomaly interactions, question results, pagination without model calls, clarification/unsupported outcomes, missing credentials, and unavailable backend.

## Forty labelled questions

`tests/fixtures/query_cases.json` records the question, expected tool, canonical validated arguments, expected calculated evidence, required status, and whether it is an official assessment example. It fixes `as_of` to `2024-03-30T18:06:00Z`.

| Coverage | Case IDs |
|---|---|
| Official examples | `open`, `resolved_month`, `critical_target`, `technical_rating`, `weekly_anomalies` |
| Counts/distributions/ranking/statistics | `total`, `unresolved`, `critical_unresolved`, `resolved`, `escalated`, `categories`, `lowest_rating`, `distinct_agents`, `median_resolution`, `p90_resolution`, `sum_response`, `max_resolution` |
| Metadata | `overview`, `schema`, `definitions` |
| Data quality | `quality`, `timing_quality`, `missing_values` |
| Ticket retrieval/search | `refund`, `ticket_id`, `missing_id`, `billing_refunds` |
| Rates | `critical_rate`, `resolution_rate`, `escalation_rate` |
| Trends | `weekly_volume`, `monthly_response`, `open_week` |
| Comparisons | `category_compare`, `count_compare` |
| Targets/anomalies | `response_target`, `overdue`, `long_resolution` |
| Non-success outcomes | `ambiguous_best`, `unsupported_email` |

The offline parametrized test deliberately supplies the canonical tool call through a scripted model. It checks real tool execution and formatted API evidence, not language understanding. No offline percentage should be reported as live accuracy.

The live runner supplies **only the natural-language question** to the agent; it does not supply the labelled tool, canonical arguments, or expected values. It checks the resulting status, selected tool, and backend evidence. Floating values use a tolerance; expected dictionaries are subsets and expected arrays check row order and count. For clarification/unsupported cases, evidence must be empty and the message nonempty. The question set is a regression suite, not a large benchmark or a guarantee for arbitrary requests.

## Opt-in live run

Put a Groq key in `.env`, then run the official subset:

```sh
uv run --locked python -m support_ticket_ai.evaluation --live --official-only --output evaluation-official.local.json
```

Full acceptance run:

```sh
uv run --locked python -m support_ticket_ai.evaluation --live --output evaluation-full.local.json
```

Useful options: `--limit 5`, `--delay 90`, `--cases path/to/cases.json`, `--output path/to/report.json`. By default, questions run sequentially with 65 seconds between them. A full run takes about 43 minutes plus request time. The runner stops on a rate limit, saves the blocked attempt, and exits with code 3. No paid fallback is configured. Daily limits need enough replenishment for a complete multi-turn question; a short provider retry interval may allow only its first request.

Continue the saved full-suite run when quota is available:

```sh
uv run --locked python -m support_ticket_ai.evaluation --live --resume
```

Resume skips completed cases, including completed semantic failures. It retries only quota-blocked cases and questions not yet attempted. Previous quota attempts remain in each case's `previous_attempts`; `first_attempt_passed` preserves the first-attempt count. A fingerprint of application code, dependencies, and fixtures prevents mixing changed versions; model, timezone, dataset fingerprint, and reference time must also match. Do not combine resume with subset options. This is evaluation progress only: no conversation history, model checkpoints, or ticket rows are persisted.

To resume a fresh report stored under another filename, include that same `--output` path with `--resume`. Use separate output filenames for subset runs so they do not overwrite the submitted evidence.

The JSON report includes model ID, dataset fingerprint, reference time, total/passed/accuracy, full-suite status, acceptance status, and per-case pass/fail/tool names/safe error codes. Mismatches also retain calculated summaries to aid diagnosis, without ticket tables or keys. The report is saved after every completed case, so interruption preserves progress. `evaluation-results.json` is included as reviewed submission evidence. Scratch reports matching `evaluation-*.local.json` are ignored by Git.

The project's internal acceptance target is **all five official examples pass and at least 36 of 40 cases pass**, after all 40 cases have completed without remaining quota blocks. This is an internal regression target, not a pass mark specified in the assessment brief. A partial run can exit successfully if its selected cases pass but always reports `acceptance_met: false`. `accuracy` uses each case's latest recorded attempt and includes quota failures until they are resolved; all prior quota attempts remain visible. Exit 0 means the selected run passed its applicable checks, 1 means failed cases/acceptance, 2 means missing configuration or bad CLI arguments, and 3 means provider quota blocked progress.

## Delivery status and remaining verification

Local verification on **September 17, 2026**:

| Check | Result |
|---|---|
| Offline pytest suite | **104 passed**, including both real-service launcher scenarios |
| Ruff lint and format | Passed; 32 Python files formatted |
| mypy | Passed; no issues in 18 source files |
| Dependency lock | `uv sync --locked --offline --dry-run` resolved successfully; no dependency changes required |
| Documentation links | All local file targets resolved |
| Dataset integrity | Source and submitted CSV bytes match; SHA-256 matches the README/report |
| Saved live evidence | Report bytes unchanged; metadata matches current application, lockfile, fixtures, model and dataset |

The test run emitted one dependency deprecation warning from Starlette's test client about the AnyIO `BlockingPortal` alias; no checks failed. Launcher tests used unused localhost ports, leaving existing services alone. No GitHub Actions run or clean-machine install is claimed by these local checks.

The included live report predates this documentation/packaging pass. It records **12 passes**, including all five assessment example intents, **7 cases blocked by provider quota**, and **21 unattempted cases**. The fixtures paraphrase some of the brief's example questions; the saved report applies to the exact questions in those fixtures. No new live Groq requests were made during the submission update.

`evaluation-results.json` reports `state: quota_blocked`, `full_suite: false` and `acceptance_met: false`. Its `accuracy` value is 12/19 (about 63.2%), because the denominator includes quota-blocked cases; it is not a completed language-understanding benchmark. The 12 completed cases passed, but reporting that subset as 100% full-suite accuracy would be misleading.

The saved report contains seven blocked cases from the earlier run. The current runner stops on the first quota error; it will not reproduce seven blocked attempts in a single fresh run. The report's existing `verification_note` documents when resume metadata was added. Application source, dependencies and fixtures still match its evaluation fingerprint; adding the root launcher and updating documentation does not alter the evaluated agent. The original report and its outcomes have been preserved.

After free quota becomes available, resume the submitted run with the command above. If the application, fixtures, model or dependencies change, use a fresh output file; do not edit a saved fingerprint to force resume. The full 40-case live target remains unverified.

Provider model availability, free quotas and output behavior can change independently of the locked Python dependencies. Keep the model setting, fixture fingerprint and evaluation report together when comparing future runs.
