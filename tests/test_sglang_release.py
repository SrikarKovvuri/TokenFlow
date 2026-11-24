"""Optional contract check against the published Linux wheel, without installing CUDA."""

import argparse
import hashlib
import os
from pathlib import Path
import runpy
import zipfile

import pytest

from tokenflow.provisioner.worker import engine_command


def test_released_sglang_config_and_auth(tmp_path):
    wheel = os.environ.get("TOKENFLOW_SGLANG_TEST_WHEEL")
    if not wheel:
        pytest.skip("Set TOKENFLOW_SGLANG_TEST_WHEEL to the SGLang 0.5.19 CPython 3.12 x86_64 wheel")
    digest = hashlib.sha256(Path(wheel).read_bytes()).hexdigest()
    assert digest == "fdc65d1235f6861ad3a1116c36c2270e46a95f030200091f8de595c6c3426cd2"
    with zipfile.ZipFile(wheel) as archive:
        config_parser = archive.extract("sglang/srt/utils/server_args_config_parser.py", tmp_path)
        auth_module = archive.extract("sglang/srt/utils/auth.py", tmp_path)

    token = "example-worker-key"
    args = argparse.Namespace(backend="sglang", checkpoint="/checkpoint", gpus=2, capacity=4,
                              model="chat", dtype="bfloat16", max_model_len=2048)
    argv = engine_command(args, 8000, tmp_path / "key")[3:]
    (tmp_path / "sglang.yaml").write_text(f'api-key: "{token}"\n')
    merger = runpy.run_path(config_parser)["ConfigArgumentMerger"]()
    merged = merger.merge_config_with_args(argv)
    assert merged[merged.index("--api-key") + 1] == token
    assert token not in argv

    auth = runpy.run_path(auth_module)
    def allowed(path, header, method):
        return auth["decide_request_auth"](
            path=path, method=method, authorization_header=header, api_key=token,
            admin_api_key=None, auth_level=auth["AuthLevel"].NORMAL,
        ).allowed

    for path, method in [("/v1/models", "GET"), ("/v1/chat/completions", "POST")]:
        assert not allowed(path, None, method)
        assert allowed(path, f"Bearer {token}", method)
    assert allowed("/health", None, "GET")
    assert allowed("/metrics", None, "GET")
