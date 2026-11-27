"""Run inside a CPU-only Slurm job. TokenFlow owns one separate GPU worker."""
import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from hpc_tp_load import benchmark
from tokenflow.provisioner.slurm import worker_token


def write(name, value):
    Path(name).write_text(json.dumps(value, indent=2))


def queue_ids():
    return subprocess.check_output(["squeue","--local","--me","-h","-o","%i"], text=True).split()


def ready(client):
    deadline = time.monotonic() + 5*3600
    job = None
    while time.monotonic() < deadline:
        receipts = list(Path("workers").glob("*/job-id"))
        if len(receipts) > 1:
            raise RuntimeError("Unexpected replacement worker")
        if receipts:
            if not job:
                job = receipts[0].read_text().strip()
                print("GPU job", job, flush=True)
            if job not in queue_ids():
                raise RuntimeError(f"GPU job {job} exited before readiness")
        workers = client.get("/v1/workers").raise_for_status().json()
        if workers and workers[0]["health_status"] == "healthy" and job:
            if (workers[0]["telemetry"] or {}).get("gpu_utilization") is not None:
                return job, workers[0]
        time.sleep(5)
    raise TimeoutError("queue or readiness deadline")


def sample_and_run(url, student, secret, job, worker, args):
    engine = worker["base_url"].rstrip("/")
    token = worker_token(secret, worker["metadata"]["resource_key"])
    with httpx.Client(headers={"Authorization":"Bearer "+token}, trust_env=False) as client:
        gpu = client.get(engine.rsplit(":", 1)[0]+":"+str(worker["metadata"]["gpu_metrics_port"])+"/metrics").raise_for_status().text
        uuids = re.findall(r'UUID="(GPU-[a-f0-9-]+)"', gpu, flags=re.I)
        assert len(set(uuids)) == len(uuids) == args.tp
        version = client.get(engine+"/version").raise_for_status().json()
    write("worker.json", dict(worker=worker, gpu_job=job, uuids=uuids, version=version, tp=args.tp, cap=args.cap))
    cmd=["srun","--overlap",f"--jobid={job}","-N1","-n1","-c1"]
    with open("topology.txt","x") as handle:
        subprocess.run(cmd+["nvidia-smi","topo","-m"],stdout=handle,check=True,timeout=60)
    fields="timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,temperature.gpu"
    with open("gpu.csv","x") as handle:
        proc=subprocess.Popen(cmd+["env","TZ=UTC","stdbuf","-oL","nvidia-smi","-i",",".join(uuids),
                                  "--query-gpu="+fields,"--format=csv,noheader,nounits","--loop-ms=500"],stdout=handle)
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    raise RuntimeError("GPU sampler exited")
                with open("gpu.csv") as f:
                    if set(uuids).issubset({r.split(",")[2].strip() for r in f if r.count(",")==10}):
                        break
                time.sleep(1)
            else:
                raise TimeoutError("GPU samples missing")
            asyncio.run(benchmark(url,engine,worker["metadata"]["model"],student,token,args.cap))
            if proc.poll() is not None:
                raise RuntimeError("GPU sampler stopped during benchmark")
        finally:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main(args):
    os.umask(0o077)
    root = Path.cwd()
    namespace="tp-"+os.environ["SLURM_JOB_ID"]+"-"+str(args.cap)
    pool_name="qwen-a6000"
    prefix="tf-"+hashlib.sha256(f"{namespace}/{pool_name}".encode()).hexdigest()[:24]+"-"
    Path("prefix").write_text(prefix)
    pool=dict(name=pool_name, provider="slurm", backend="vllm", model="qwen-14b",
              image=str(args.runtime/"containers/vllm.sif"),checkpoint=str(args.runtime/"models/qwen2.5-14b-instruct"),
              gpu_count=args.tp,gpu_type="A6000",capacity=args.cap,min_replicas=0,max_replicas=1,
              options=dict(account="edu",partition="burst",qos="burst",memory="96G",cpus=8,
                           time_limit="01:20:00",state_dir=str(root/"workers"),apptainer_module="apptainer/1.4.5",
                           dtype="bfloat16",max_model_len=2048))
    write("config.json",dict(namespace=namespace,state_path=str(root/"state.sqlite"),interval_seconds=4,
                             startup_timeout_seconds=4500,pools=[pool]))
    checkpoint=Path(pool["checkpoint"])
    write("checkpoint.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.glob("*.json")})
    admin,student,secret=[secrets.token_hex(32) for _ in range(3)]
    env=dict(os.environ,TOKENFLOW_PLATFORM_CONFIG=str(root/"config.json"),TOKENFLOW_ADMIN_KEY=admin,
             TOKENFLOW_API_KEYS=json.dumps([student]),TOKENFLOW_WORKER_SECRET=secret)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0)); sock.listen()
        url=f"http://127.0.0.1:{sock.getsockname()[1]}"
        gw=subprocess.Popen([sys.executable,"-m","uvicorn","tokenflow.main:app","--fd",str(sock.fileno()),
                             "--log-level","warning"],pass_fds=(sock.fileno(),),env=env)
    with httpx.Client(base_url=url,headers={"Authorization":"Bearer "+admin},trust_env=False,timeout=30) as client:
        def limits(n):
            client.put("/v1/pools/"+pool_name+"/limits",json={"minimum":n,"maximum":n}).raise_for_status()
        try:
            for _ in range(40):
                if gw.poll() is not None: raise RuntimeError("gateway exited")
                try:
                    if client.get("/healthz").is_success: break
                except httpx.ConnectError: pass
                time.sleep(1)
            limits(1)
            job,worker=ready(client)
            sample_and_run(url,student,secret,job,worker,args)
            limits(0)
            for _ in range(90):
                if job not in queue_ids() and not client.get("/v1/workers").raise_for_status().json():
                    write("release.json",dict(job=job,verified_epoch=time.time(),accounting=subprocess.check_output(
                        ["sacct","-X","-j",job,"-nP","-o","JobID,State,AllocTRES,End"],text=True)))
                    return
                time.sleep(3)
            raise TimeoutError("release")
        finally:
            try: limits(0)
            finally:
                gw.terminate()
                gw.wait(timeout=30)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp",choices=(1,2,4,8),type=int,required=True)
    parser.add_argument("--cap",choices=(16,64),type=int,required=True)
    parser.add_argument("--runtime",type=Path,default=Path.home()/"vllm-runtime")
    main(parser.parse_args())
