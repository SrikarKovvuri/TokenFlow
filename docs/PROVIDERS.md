# GPU worker providers

An operator configures a pool for each combination of checkpoint, backend, and hardware. A replica is one Slurm job or one EC2 instance. It runs one engine, optionally sharded across several GPUs with tensor parallelism. The gateway and its ingress have their own lifecycle. Students keep using the same base URL when an allocation ends.

## Insomnia with Slurm and Apptainer

For a same-host worker test, pass `--loopback-only` to `worker.py` so both
the engine and its adjacent GPU metrics listener bind to `127.0.0.1`.
Managed pool deployments use the default bind address for private network access.

Use a service account authorized for the requested account, partition, QoS and GPU type. Place the gateway on an approved service host with working Slurm clients and network routes to the compute nodes. Long-running services belong on a site-approved host. A workstation would need an equivalent private route and authenticated Slurm client. Students do not need SSH access to use an already deployed gateway. [Slurm submission](https://slurm.schedmd.com/sbatch.html)

1. Stage the checkpoint directory and a versioned SIF containing the chosen engine on shared, compute-readable storage. The image needs Python, PyTorch with CUDA device properties including UUID, and access to `nvidia-smi` through Apptainer. In the Insomnia example, substitute the actual paths to the existing Qwen checkpoint and vLLM image under `vllm-runtime`. Slurm mounts the checkpoint read-only at `/checkpoint`. [Apptainer bind mounts](https://apptainer.org/docs/user/latest/bind_paths_and_mounts.html)
2. Give the service account a dedicated shared project directory for `options.state_dir`. Create it with mode `0700` in a trusted parent. Job logs, submission receipts, derived worker keys, and tiny port files live here. Verify there is enough space for logs. Set the job's `TMPDIR` to node-local storage with room to unpack the SIF and compile the model kernels. The job puts its temporary runtime, image extraction, Hugging Face, vLLM, Triton, CUDA, and TorchInductor caches under that directory.
3. Set `gpu_type` to the GRES spelling the partition advertises, for example `A6000`, `H100`, or `l40s` where available. The device probe verifies that the CUDA-visible count and model names match the pool. Each job is limited to one node. Set `gpu_count` to the desired tensor parallelism, `capacity` to the engine's concurrent sequence limit, and `options.max_model_len` to the intended context length. Optional `options.dtype` accepts `auto`, `float16`, `bfloat16`, or `float32`. [vLLM serving arguments](https://docs.vllm.ai/en/v0.10.2/cli/serve.html), [SGLang server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
4. Set `TOKENFLOW_WORKER_SECRET` in the gateway's protected environment. The gateway derives a different bearer credential for each job. The batch script only contains the path to a private file. Slurm vLLM uses its API key and an ASGI authentication middleware that also covers its diagnostic routes. Slurm SGLang reads its inference API key from a private YAML file and runs at warning log level because its startup info log includes the resolved configuration. Native SGLang intentionally allows unauthenticated health and engine metrics requests. Restrict worker ingress to trusted hosts. The gateway uses HTTP on this internal link. [vLLM API key](https://docs.vllm.ai/en/v0.10.2/configuration/env_vars.html), [SGLang 0.5.19 authorization](https://github.com/sgl-project/sglang/blob/v0.5.19/python/sglang/srt/utils/auth.py)

The controller submits `sbatch` directly and tracks its numeric job ID. The Slurm installation must support `--export=NIL`. The job starts a login shell for the site's module initialization and inherits Slurm's resource bindings. Each compute job selects a free port pair on its node. After the engine answers an authenticated model-list request, the job atomically publishes its engine port. Discovery combines Slurm's batch hostname with this port. The telemetry listener uses the next port. Allow the service host to reach these compute-node ports on the private network. The pool's static `port` is used in cloud mode. [Slurm inventory fields](https://slurm.schedmd.com/squeue.html)

TokenFlow normally submits workers with `--no-requeue`. If site policy restarts a job, Slurm can reuse its ID on a different node. Discovery reads `RestartCnt` and waits for that attempt's own port file. An existing job on its first attempt keeps its original port file across a gateway upgrade. A restarted job running an older batch script needs a fresh allocation to publish the new receipt. [Slurm requeue policy and restart count](https://slurm.schedmd.com/sbatch.html)

The service account's Slurm client configuration selects the local cluster. Set `SLURM_CONF` when the site uses a nonstandard path. The provider clears inherited `SBATCH_*`, `SQUEUE_*`, `SCANCEL_*` and `SLURM_CLUSTERS` overrides so interactive array defaults and queue filters cannot change the pool's resources or hide its workers. [Slurm submission environment](https://slurm.schedmd.com/sbatch.html#SECTION_INPUT-ENVIRONMENT-VARIABLES)

A launch writes a private submission claim before invoking `sbatch`. Recognized Slurm rejections for invalid accounts, QoS or resource requests release the claim for a retry. If the response is lost, retries adopt the matching job or use the saved receipt. A claim without a receipt or matching job records an ambiguous submission and prevents duplicate allocation. To resolve that state, an operator checks `squeue` and accounting for the exact TokenFlow job name. If the job exists, let it reconcile. Once the operator independently establishes that the job never launched or has already ended, move the specific intent directory aside to allow a retry. Keep state directories until their jobs end. Archive or expire completed logs according to project storage policy.

## AWS EC2

Use an always-on control host in a VPC with routes to the worker subnet and a trusted HTTPS endpoint for students. Give the gateway a role that can describe instances, launch one GPU instance at a time, tag on creation, and terminate instances within its own namespace. Scope its permitted AMIs, region, instance types, subnets and security groups. Restrict termination by resource tag. When `iam_instance_profile` is configured, the control role also needs `iam:PassRole` for that specific role. Install the `aws` optional dependency on the gateway. [EC2 run instances](https://docs.aws.amazon.com/boto3/latest/reference/services/ec2/client/run_instances.html)

1. Supply a Linux AMI with cloud-init, systemd, Docker, a compatible NVIDIA driver and NVIDIA Container Toolkit. Verify that a Docker container can see the GPUs, and provision enough host RAM and disk for the model, the engine and its caches. Configure the AMI's root volume to delete on termination. Choose a regional GPU instance type matching `gpu_type` and `gpu_count`. An instance is dedicated to a single engine; a four-GPU instance costs the same when the pool uses only one of its devices.
2. Set `options.region`, `instance_type`, `ami_id`, `subnet_id`, and `security_group_ids` using the cloud example. Use a private subnet. Worker ingress should allow only the gateway security group on `port` and `port + 1`. The provider disables automatic public IPv4 assignment and requires IMDSv2. Configure network ACLs, routes, and IPv6 rules consistently with that private ingress. The gateway reaches the instance at its private IPv4 address.
3. Supply a trusted OCI engine image by digest or version. Use an absolute checkpoint path for weights baked into the AMI. For weights on a separate filesystem, configure its systemd mount unit or fstab entry in the AMI and set `options.checkpoint_mount` to its mount point, such as `/models` for checkpoint `/models/qwen`. At first boot TokenFlow starts that unit and verifies the mount before starting the container. A systemd dependency preserves the ordering on reboot. Docker bind-mounts the checkpoint read-only. A Hugging Face repository name is also supported when your image has the required credentials and network access. For isolated subnets, pre-stage the checkpoint and image. Keep credentials out of the catalog, image reference and EC2 user data. [systemd mount units](https://www.freedesktop.org/software/systemd/man/latest/systemd.mount.html)
4. The startup script launches Docker with `--restart=unless-stopped`, an explicit GPU mask and size-limited logs. EC2 retries with the same durable launch key use the same client token. Discovery paginates tagged instances and checks pool, deployment and launch key before termination. Use the pool limits endpoint to drain and release workers when finished. [AWS idempotent run requests](https://docs.aws.amazon.com/ec2/latest/devguide/ec2-api-idempotency.html), [Linux user data](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html)

Each worker is disposable. A first-boot script error, including a failed mount, image pull or Docker launch, requests a guest shutdown. TokenFlow configures EC2 to terminate that VM on guest shutdown. After Docker accepts the launch, engine readiness and recovery use the control plane's health checks and readiness deadline. Keep checkpoints and logs you need after termination outside the instance's disposable volumes. [EC2 guest shutdown behavior](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html)

## Verify one allocation

For SGLang, select `backend: sglang` and stage an engine image compatible with the host driver and GPU. The CUDA 12.9 multi-architecture image for the reference release is `lmsysorg/sglang:v0.5.19-cu129@sha256:59e11312666e1c5c155210ea335589b91daa0d70848521b390b93b1b1e8fb0ef`. EC2 accepts the OCI reference directly. Slurm requires the compute-readable path to the converted SIF. The release configuration checks run on CPU. Verify model generation and streaming on the target GPU before opening the pool to students. [SGLang release](https://github.com/sgl-project/sglang/releases/tag/v0.5.19), [publisher's image tag](https://hub.docker.com/r/lmsysorg/sglang/tags?name=v0.5.19-cu129)

To repeat the offline SGLang compatibility check, download the CPython 3.12 Linux x86-64 wheel with the command below. Set `TOKENFLOW_SGLANG_TEST_WHEEL` to its path and run `python -m pytest tests/test_sglang_release.py` in the gateway test environment with PyYAML installed. The test verifies the wheel's hash and exercises its config loader and bearer authorization decisions. [Published wheel](https://pypi.org/project/sglang/0.5.19/#files)

```console
python -m pip download --no-deps --only-binary=:all: --platform manylinux_2_34_x86_64 --python-version 312 --dest .pytest_cache/sglang-contract sglang==0.5.19
```

The gateway polls each managed worker's `/health` and checks that its OpenAI `/v1/models` list includes the pool's model alias. A reachable server serving a different checkpoint alias stays out of routing. If it later advertises the configured alias, the next successful probe makes it eligible. Manually registered endpoints continue to use `/health` for readiness.

Start with one warm replica and a maximum of one. The operator should see a provider allocation ID, a running resource, a healthy worker, and a response to a streaming chat request using the configured alias. Check that both the engine's queue/cache metrics and the per-GPU utilization signal arrive. Utilization is sampled using `nvidia-smi` over the CUDA-visible GPU UUIDs. The exporter uses the Prometheus gauge name `DCGM_FI_DEV_GPU_UTIL` for this percentage. A failed sample returns HTTP 503 and the scheduling layer can continue using its available signals.

For Slurm, inspect the service account's job log under `state_dir/tf-*/slurm-JOBID.out` and use `sacct` to distinguish queue time, allocation expiry, and process failures. For EC2, use your instance management channel to inspect `/var/log/cloud-init-output.log` and `docker logs tokenflow-worker`. Inspect the gateway's pool status before retrying a launch. Engine startup can take several minutes while CUDA compiles kernels.

## Local cold-start verification

On Linux or WSL2, use a gateway environment with `.[dev]` installed, a local
Python environment with CUDA-enabled vLLM, and a cached chat checkpoint that
fits the GPU in FP16. Make `nvidia-smi` available on PATH. On WSL2 it is
usually in `/usr/lib/wsl/lib`. From the repo root, substitute the engine
interpreter, checkpoint directory and a GPU label matching `nvidia-smi`.

```bash
TOKENFLOW_TEST_SLURM_ENGINE_PYTHON=/path/to/vllm-env/bin/python \
TOKENFLOW_TEST_SLURM_CHECKPOINT=/path/to/cached/chat-checkpoint \
TOKENFLOW_TEST_SLURM_GPU=4060 \
python -m pytest -q tests/test_slurm_worker_live.py
```

A cold chat request triggers the real Slurm provisioner and generated batch
script. The test emulates the Slurm client responses and translates Apptainer
bind paths so the local interpreter can run the shipped worker. It checks
the job's credentials, port receipt, JSON and SSE inference, and GPU and
engine telemetry. It drains and stops the process group and removes the
private job directory, credentials and caches. To retain measurements, set
`TOKENFLOW_TEST_SLURM_EVIDENCE` to a new JSON file in an existing directory.
