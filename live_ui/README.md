# Live demo UI

A thin Streamlit client of the fraud-intelligence API. It sends each question to
`POST /query` and shows the answer with its evidence: the executed SQL and rows
for analytics answers, the cited passages for document answers.

- One setting: `API_URL` (defaults to the live API). No provider or database credentials.
- Imports nothing from the API's packages; `tests/test_live_ui.py` enforces this.
- Errors are explained on the page: rate limit (only when the 429 carries the API limiter's `Retry-After`), API not available (a 429 without it, or a 502/503/504: usually a free-tier service waking up), timeout, unreachable API, API error, guardrail rejection.

Run locally:

```
pip install -r live_ui/requirements.txt
API_URL=http://127.0.0.1:8000 streamlit run live_ui/app.py
```

Deploy: the `ai-fraud-agent-ui` service in `render.yaml` (root directory `live_ui`).

The API's per-client rate limit applies to the UI server as one client, so all
visitors share it. This is a live demo on a free tier, not production.
