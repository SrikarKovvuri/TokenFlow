"""Stream one completion with the OpenAI SDK, allowing time for a cold GPU pool."""

import argparse
import os
import sys
import time
from email.utils import parsedate_to_datetime

from openai import APIStatusError, OpenAI


def retry_after_seconds(value: str | None) -> float:
    if value is None:
        return 0
    if value.isascii() and value.isdecimal():
        return float(value)
    try:
        return max(0, parsedate_to_datetime(value).timestamp() - time.time())
    except (TypeError, ValueError, OverflowError, OSError):
        return 0


def stream_when_ready(client: OpenAI, model: str, prompt: str, wait_seconds: int = 900):
    client = client.with_options(max_retries=0)
    deadline = time.monotonic() + wait_seconds
    backoff = 1
    while (remaining := deadline - time.monotonic()) > 0:
        try:
            return client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                stream=True, timeout=min(remaining, 120),
            )
        except APIStatusError as exc:
            headers = exc.response.headers
            if exc.status_code != 503 or headers.get("x-tokenflow-admission") != "waiting-for-worker":
                raise
            if backoff == 1:
                print(f"Waiting up to {wait_seconds}s for a worker serving {model}…", file=sys.stderr)
            delay = max(backoff, retry_after_seconds(headers.get("retry-after")))
            if delay >= deadline - time.monotonic():
                break
            time.sleep(delay)
            backoff = min(30, backoff * 2)
    raise TimeoutError(f"No worker became available for {model} within the {wait_seconds}s wait budget")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model alias supplied by the operator")
    parser.add_argument("prompt")
    parser.add_argument("--wait-seconds", type=int, default=900, help="Cold-start wait budget (default 900)")
    args = parser.parse_args()
    if args.wait_seconds <= 0:
        parser.error("--wait-seconds must be positive")
    if not os.environ.get("OPENAI_BASE_URL"):
        parser.error("Set OPENAI_BASE_URL to the TokenFlow URL supplied by your operator")
    with OpenAI() as client:
        aliases = [model.id for model in client.models.list().data]
        if args.model not in aliases:
            parser.error(f"Available models: {', '.join(aliases) or '(none)'}")
        with stream_when_ready(client, args.model, args.prompt, args.wait_seconds) as stream:
            for event in stream:
                if event.choices:
                    print(event.choices[0].delta.content or "", end="", flush=True)
        print()


if __name__ == "__main__":
    main()
