import argparse
import os
import sys
from pathlib import Path

from tokenflow.platform.config import PlatformConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="tokenflow")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="validate a managed deployment offline")
    check.add_argument("config", type=Path)
    serve = commands.add_parser("serve", help="start the gateway and its controller")
    serve.add_argument("--config", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        config = PlatformConfig.load(args.config)
    except (ValueError, OSError) as exc:
        print(f"Invalid deployment configuration: {exc}", file=sys.stderr)
        return 2
    if args.command == "validate":
        print(f"{config.namespace}: {len(config.pools)} pool(s), state at {config.state_path}")
        return 0

    # Set before Uvicorn imports the application and reads its settings.
    import uvicorn

    os.environ["TOKENFLOW_PLATFORM_CONFIG"] = str(args.config.resolve())
    uvicorn.run("tokenflow.main:app", host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
