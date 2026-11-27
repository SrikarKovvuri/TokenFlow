import argparse
import json
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
    account = commands.add_parser("aws-status", help="inspect the signed-in account, promotional credits and GPU quotas")
    account.add_argument("--expected-account", required=True, help="exact 12-digit account that owns the credits")
    account.add_argument("--profile", default="default")
    account.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    if args.command == "aws-status":
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError
        except ImportError:
            parser.exit(2, "Install the TokenFlow 'aws' extra to use aws-status.\n")
        from tokenflow.provisioner.aws_account import account_status

        try:
            session = boto3.Session(profile_name=args.profile, region_name=args.region)
            report = account_status(session, args.expected_account)
        except (BotoCoreError, ClientError, ValueError) as exc:
            parser.exit(1, f"AWS account check failed: {exc}\n")
        print(json.dumps(report, indent=2))
        return 0

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
