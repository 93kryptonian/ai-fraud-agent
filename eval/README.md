# Offline and live evaluation

Evaluation of the system's **behavior**: did the right path answer, did it refuse what
it should refuse, is the answer grounded, in the right language. Not model benchmarking.

The full description, results and limits are in [`docs/evaluation.md`](../docs/evaluation.md).

## Files

- `cases.yaml`: the cases and what the system should do for each
- `evaluator.py`: turns a `/query` response into an observation and checks it against the case
- `telemetry.py`: per-case latency, cost and routing layer, read from the system's own events
- `run_eval.py`: the live run

## Two kinds of check

**Offline gate** (runs in CI, no credentials): `pytest tests/test_eval_offline_routing.py`.
Calls the production routing and guardrail code with the LLM stubbed out. It proves the
deterministic rules route the labelled cases correctly. It does not prove the LLM, retrieval or
answers are right.

**Live evaluation** (by hand): real LLM and database, so it costs a few cents.

```
python -m eval.run_eval --out results.json
```

Needs the same environment as the API (an OpenAI key and Supabase settings, read from `.env`).
It prints one verdict per case and writes per-case measurements (no answer text) to `--out`.
Cases with `scored: false` are reported as INFO and not counted.

## Changing the cases

A case describes intended behavior. If a live case fails, that is a finding to report; edit the
case only when the expectation itself was wrong, and say why in the commit.
