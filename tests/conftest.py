import os

# The app's per-IP rate limiter is per-process and shared by every
# TestClient request in the run; keep it out of the way of the suite.
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")
