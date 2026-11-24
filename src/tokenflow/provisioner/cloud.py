"""Manage a namespaced fleet of private, single-worker EC2 instances."""

import asyncio
import hashlib
from ipaddress import IPv4Address
import re
import shlex
from typing import Annotated

from pydantic import Field, StringConstraints

from tokenflow.provisioner.base import PoolSpec, Resource
from tokenflow.provisioner.launch import EngineOptions, check_mount_path, worker_args, worker_script


class CloudOptions(EngineOptions):
    region: Annotated[str, StringConstraints(pattern=r"^[a-z]{2}(?:-[a-z0-9]+)+-[0-9]+$")]
    instance_type: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]+\.[a-z0-9]+$")]
    ami_id: Annotated[str, StringConstraints(pattern=r"^ami-[a-f0-9]+$")]
    subnet_id: Annotated[str, StringConstraints(pattern=r"^subnet-[a-f0-9]+$")]
    security_group_ids: list[Annotated[str, StringConstraints(pattern=r"^sg-[a-f0-9]+$")]] = Field(min_length=1)
    iam_instance_profile: Annotated[str, StringConstraints(pattern=r"^[a-zA-Z0-9+=,.@_-]+$")] | None = None


def bootstrap(pool: PoolSpec, opts: CloudOptions) -> str:
    script = "/var/lib/tokenflow/worker.py"
    checkpoint = pool.checkpoint
    docker = ["docker", "run", "--detach", "--name", "tokenflow-worker", "--restart=unless-stopped",
              "--gpus", "all", "--shm-size=8g", "--log-opt", "max-size=25m", "--log-opt", "max-file=3",
              "--env", "CUDA_VISIBLE_DEVICES=" + ",".join(str(i) for i in range(pool.gpu_count)),
              "--publish", f"{pool.port}:{pool.port}", "--publish", f"{pool.port + 1}:{pool.port + 1}",
              "--mount", f"type=bind,src={script},dst=/worker.py,readonly"]
    if checkpoint.startswith("/"):
        docker.extend(["--mount", f"type=bind,src={checkpoint},dst=/checkpoint,readonly"])
        checkpoint = "/checkpoint"
    docker.extend(["--entrypoint", "python3", pool.image, "/worker.py"])
    docker.extend(worker_args(pool, opts, pool.port, checkpoint))
    data = f"""#!/bin/bash
set -euo pipefail
umask 022
install -d -m 0755 /var/lib/tokenflow
cat > {script} <<'TOKENFLOW_PY'
{worker_script()}
TOKENFLOW_PY
systemctl enable --now docker
{shlex.join(docker)}
"""
    if len(data.encode()) > 16 * 1024:
        raise ValueError("EC2 worker bootstrap exceeds the 16 KiB user-data limit")
    return data


class CloudProvisioner:
    def __init__(self, namespace: str):
        self.namespace = namespace
        self._clients = {}

    @staticmethod
    def validate_pool(pool: PoolSpec) -> None:
        opts = CloudOptions.model_validate(pool.options)
        if pool.image.startswith("-") or not re.fullmatch(r"[a-zA-Z0-9./_:@-]+", pool.image):
            raise ValueError("Expected an OCI image reference")
        if pool.checkpoint.startswith("/"):
            check_mount_path(pool.checkpoint)
        bootstrap(pool, opts)

    def _client(self, region: str):
        if region not in self._clients:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise RuntimeError("Install tokenflow[aws] to enable EC2 pools") from exc
            self._clients[region] = boto3.client(
                "ec2", region_name=region,
                config=Config(connect_timeout=5, read_timeout=30, retries={"total_max_attempts": 3}),
            )
        return self._clients[region]

    def _tags(self, pool: PoolSpec) -> dict[str, str]:
        return {"TokenFlowNamespace": self.namespace, "TokenFlowPool": pool.name}

    def _resource(self, pool: PoolSpec, instance: dict) -> Resource | None:
        tags = {tag["Key"]: tag["Value"] for tag in instance.get("Tags", [])}
        if any(tags.get(k) != v for k, v in self._tags(pool).items()):
            return None
        key = tags.get("TokenFlowKey", "")
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            return None
        state = instance["State"]["Name"]
        if state in {"terminated", "shutting-down"}:
            return Resource(key=key, external_id=instance["InstanceId"], phase="terminal")
        if state == "running" and instance.get("PrivateIpAddress"):
            ip = IPv4Address(instance["PrivateIpAddress"])
            return Resource(key=key, external_id=instance["InstanceId"], phase="running",
                            base_url=f"http://{ip}:{pool.port}")
        if state in {"running", "stopping", "stopped"}:
            # These instances still own resources and cannot serve without an
            # endpoint. Mark them allocated so the readiness deadline applies.
            return Resource(key=key, external_id=instance["InstanceId"], phase="running")
        return Resource(key=key, external_id=instance["InstanceId"], phase="pending")

    def _discover(self, pool: PoolSpec, key: str | None = None) -> list[Resource]:
        opts = CloudOptions.model_validate(pool.options)
        tags = self._tags(pool)
        if key:
            tags["TokenFlowKey"] = key
        filters = [{"Name": f"tag:{k}", "Values": [v]} for k, v in tags.items()]
        paginator = self._client(opts.region).get_paginator("describe_instances")
        resources = []
        for page in paginator.paginate(Filters=filters):
            for reservation in page["Reservations"]:
                for instance in reservation["Instances"]:
                    resource = self._resource(pool, instance)
                    if resource and (key is None or resource.key == key):
                        resources.append(resource)
        return resources

    async def discover(self, pool: PoolSpec) -> list[Resource]:
        return await asyncio.to_thread(self._discover, pool)

    def _launch(self, pool: PoolSpec, key: str) -> Resource:
        self.validate_pool(pool)
        if not re.fullmatch(r"[0-9a-f]{32}", key):
            raise ValueError("Invalid launch key")
        opts = CloudOptions.model_validate(pool.options)
        tags = self._tags(pool) | {"TokenFlowKey": key, "Name": f"tokenflow-{pool.name}"}
        params = {
            "MinCount": 1, "MaxCount": 1, "ImageId": opts.ami_id, "InstanceType": opts.instance_type,
            "NetworkInterfaces": [{"DeviceIndex": 0, "SubnetId": opts.subnet_id,
                                   "Groups": opts.security_group_ids, "AssociatePublicIpAddress": False,
                                   "DeleteOnTermination": True}],
            "MetadataOptions": {"HttpTokens": "required", "HttpPutResponseHopLimit": 1},
            "TagSpecifications": [{"ResourceType": "instance", "Tags": [
                {"Key": name, "Value": value} for name, value in tags.items()]}],
            "UserData": bootstrap(pool, opts),
            "ClientToken": hashlib.sha256(f"{self.namespace}/{pool.name}/{key}".encode()).hexdigest(),
        }
        if opts.iam_instance_profile:
            params["IamInstanceProfile"] = {"Name": opts.iam_instance_profile}
        result = self._client(opts.region).run_instances(**params)
        instance = result["Instances"][0]
        return Resource(key=key, external_id=instance["InstanceId"], phase="pending")

    async def launch(self, pool: PoolSpec, key: str) -> Resource:
        return await asyncio.to_thread(self._launch, pool, key)

    def _terminate(self, pool: PoolSpec, resource: Resource) -> None:
        opts = CloudOptions.model_validate(pool.options)
        for owned in self._discover(pool, resource.key):
            if owned.external_id == resource.external_id and owned.phase != "terminal":
                self._client(opts.region).terminate_instances(InstanceIds=[owned.external_id])
                return

    async def terminate(self, pool: PoolSpec, resource: Resource) -> None:
        await asyncio.to_thread(self._terminate, pool, resource)
