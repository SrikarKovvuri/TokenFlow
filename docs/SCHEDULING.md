# How TokenFlow chooses a worker

Each chat request names a model. TokenFlow considers enabled, healthy workers
whose `metadata.model` equals that alias. One gateway process owns the live
reservation count. It reserves a slot before contacting the worker and holds
it until the response closes, including through the final bytes of a stream.

## Capacity and estimated delay

`metadata.capacity` sets the maximum outstanding requests for a worker. It
defaults to 32. `metadata.service_time_seconds` is an operator's estimate of
the time a typical request takes on that worker. It defaults to one second.
Use comparable requests to calibrate this value across A6000, H100 and L40S
workers. GPU names alone do not determine serving speed.

For each candidate the gateway computes

```text
outstanding    = max(gateway reservations, engine running + engine waiting)
estimated wait = service time × outstanding / capacity
routing score  = service time + estimated wait
```

The maximum avoids counting gateway requests twice. A worker at capacity
receives no new reservations. Among the remaining workers the lowest score
wins. This estimates relative delay under similar workloads. Long prompts,
output lengths and continuous batching affect actual completion time. The
gateway breaks ties using the lowest sampled KV occupancy, then the lowest
physical GPU utilization, then registration order. An unavailable gauge has a
neutral tie-break value of 0.5. Equal calibration and absent metrics give the
existing least-active behavior.

## What the gateway measures

The health monitor calls `/health`. For workers declaring
`metadata.backend` as `vllm` or `sglang`, it also polls `/metrics`. Managed
workers expose a separate private GPU metrics port. A manual worker can set
`metadata.gpu_metrics_port` for an exporter on the same host. Its response
must describe only that worker's allocated GPUs.

- vLLM reports running and waiting requests and KV occupancy. The collector
  accepts both `vllm:kv_cache_usage_perc` and the older
  `vllm:gpu_cache_usage_perc`. [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/)
- SGLang reports `sglang:num_running_reqs`, `sglang:num_queue_reqs` and
  `sglang:token_usage` with metrics enabled. [SGLang production metrics](https://sgl-project.github.io/references/production_metrics.html)
- The GPU sampler reports `DCGM_FI_DEV_GPU_UTIL` in percent. TokenFlow exposes
  this as a fraction in `GET /v1/workers` alongside the sample's UTC time.

One worker endpoint represents one engine replica. For tensor parallelism the
gateway takes the maximum per gauge across ranks so repeated rank metrics
cannot count a single request twice. Independent replicas register separate
endpoints. Missing samples are null. A metrics outage leaves inference
available. Routing ignores samples older than 30 seconds.

## Overload, failures and draining

When every eligible worker is full the gateway returns HTTP 429 with
`Retry-After: 1`. When the model has no eligible healthy workers it returns
HTTP 503 with the same retry hint. Both responses arrive immediately. Configured
model pools record recent peak demand, including admission failures, so an
autoscaler can bring up capacity even when a model starts at zero workers.

For a cold model, the gateway remembers one arrival for up to five minutes
while waiting for an eligible pool's provider inventory or launch backoff.
This gives the controller a chance to submit one allocation even if a slow
poll outlasts its usual demand window. The gateway stores no prompt for
replay. Once Slurm accepts the first job, it can keep its queue position for
hours. The startup deadline begins when Slurm runs it. The first successful
health check starts its idle grace period. Send the chat request again once
the worker is healthy. A warm minimum of one avoids this scale-from-zero wait.
Pausing every pool for a model clears its recorded demand.

The autoscaler compares recent gateway demand with the capacity available to
that model. Healthy manual workers count toward supply. Fresh engine requests
above a worker's live gateway reservations occupy some of its slots, up to
its configured capacity. For example, if direct clients fill all four slots
on a manual worker and TokenFlow receives one more request, one additional
slot can satisfy that demand. The requests already queued at the engine
stay there. Direct traffic alone can keep an existing managed worker running
through idle scale-down. New replicas require a configured minimum or gateway
demand. Explicit replica ceilings take precedence, including a pool pause.

On a failed connection attempt the gateway marks that endpoint unhealthy,
releases the reservation and tries one different eligible worker. It preserves
the request ID. A successful health probe brings a recovered worker back into
service. A read timeout, worker HTTP response or broken stream may represent a
request the engine already processed. The gateway returns the error or closes
the interrupted stream, without replaying inference.

To drain a manual worker, PATCH `/v1/workers/{id}` with
`{"enabled": false}`. Existing reservations finish. When `active_requests`
reaches zero the operator can remove the worker. The managed pool controller
uses the same reservation count when scaling down.
