"""Offline analysis of saved requests and timestamped telemetry; no network I/O."""
import csv
import hashlib
import json
import math
import re
import statistics as st
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMALL = ROOT.parent / 'qwen-0.5B-single-gpu'

def loadlines(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]

def ts(s):
    return datetime.fromisoformat(s).timestamp()

def percentile(values, p=.95):
    a = sorted(values)
    index = (len(a)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return a[lo] + (a[hi]-a[lo])*(index-lo)

def stats(values):
    return dict(n=len(values), mean=st.mean(values), min=min(values), max=max(values)) if values else None

cells = loadlines(ROOT/'serving/cells.jsonl')
smallcells = loadlines(SMALL/'serving/cells.jsonl')
assert json.loads((ROOT/'baseline.json.txt').read_text()) == json.loads((SMALL/'baseline.json.txt').read_text())
hardware = loadlines(ROOT/'qwen14b-worker-13660605-fixed.jsonl')
scrapes = loadlines(ROOT/'vllm-metrics.jsonl')
phases = ['request_queue_time_seconds', 'request_prefill_time_seconds',
          'request_decode_time_seconds', 'e2e_request_latency_seconds',
          'inter_token_latency_seconds']
pattern = re.compile(r'^vllm:('+'|'.join(phases)+r')_(count|sum)(?:\{.*\})?\s+(\S+)')
snapshots = []
for s in scrapes:
    values = {}
    for line in s.get('raw_prometheus', '').splitlines():
        m = pattern.match(line)
        if m:
            key = m[1]+'_'+m[2]
            values[key] = values.get(key, 0) + float(m[3])
    if len(values) == 10:
        snapshots.append(dict(time=ts(s['timestamp']), values=values))
base = snapshots[0]['values']['e2e_request_latency_seconds_count']
resets = any(b['values'][p+'_count'] < a['values'][p+'_count']
             for a,b in zip(snapshots,snapshots[1:]) for p in phases)
assert not resets, 'Counter reset: cannot use cumulative completion boundaries'
complete = [c for c in cells if c['successes']==c['requests'] and not c['failures']]
results = []
for c in complete:
    start,end = ts(c['started_at']),ts(c['finished_at'])
    rows = loadlines(ROOT/f"serving/requests-{c['cell_id']:04d}.jsonl")
    assert len(rows)==64 and all(r.get('status_code')==200 and not r.get('error') for r in rows)
    assert all(r['usage']['completion_tokens']==128 for r in rows)
    h = [r for r in hardware if start <= ts(r['timestamp']) <= end]
    v = [r for r in scrapes if start <= ts(r['timestamp']) <= end]
    gaps = [ts(b['timestamp'])-ts(a['timestamp']) for a,b in zip(h,h[1:])]
    out = {k:c[k] for k in ['cell_id','concurrency','stream','trial','started_at','finished_at',
          'successful_requests_per_second','elapsed_seconds']}
    out.update(requests=len(rows), latency_mean=st.mean(r['latency_seconds'] for r in rows),
               latency_p95=percentile([r['latency_seconds'] for r in rows]),
               output_tokens=stats([r['usage']['completion_tokens'] for r in rows]),
               input_tokens=stats([r['usage']['prompt_tokens'] for r in rows]),
               first_content=stats([r['time_to_first_content_seconds'] for r in rows if r.get('time_to_first_content_seconds') is not None]),
               hardware_samples=len(h), hardware_max_gap_seconds=max(gaps) if gaps else None,
               gpu_errors=sum('gpu_error' in r for r in h))
    for key in ['utilization.gpu','utilization.memory','memory.used','memory.total','power.draw','temperature.gpu']:
        out['gpu_'+key] = stats([g[key] for r in h for g in r.get('gpus',[]) if isinstance(g.get(key),(int,float))])
    for key in ['cpu_percent','memory_percent','memory_used_bytes','disk_free_bytes']:
        out[key] = stats([r[key] for r in h if isinstance(r.get(key),(int,float))])
    for key in ['vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc']:
        out[key] = stats([m['value'] for r in v for m in r.get('metrics',[]) if m['name']==key])
    # Exact request phase deltas exclude warmups. Successful cells form the first 9 cells.
    prior = cells[:c['cell_id']]
    assert all(p['successes']==64 for p in prior)
    before = base + sum(p['requests']+len(json.loads((ROOT/f"serving/warmup-{p['cell_id']:04d}.json").read_text())) for p in prior)
    warmups = json.loads((ROOT/f"serving/warmup-{c['cell_id']:04d}.json").read_text())
    target = before + len(warmups)
    pre = [s for s in snapshots if s['time']<=end and all(s['values'][p+'_count']==target for p in phases[:4])]
    post = [s for s in snapshots if s['time']>=end and all(s['values'][p+'_count']==target+64 for p in phases[:4])]
    out['phase_means_seconds'] = None
    if pre and post:
        a,b = pre[-1],post[0]
        out['phase_means_seconds']={p:(b['values'][p+'_sum']-a['values'][p+'_sum'])/64 for p in phases[:4]}
        out['phase_boundary_counts']=[target,target+64]
        out['client_minus_engine_seconds']=out['latency_mean']-out['phase_means_seconds']['e2e_request_latency_seconds']
        # Direct ITL histogram includes warmups because observations accrue during decoding.
        fullpre=[s for s in snapshots if s['time']<=ts(warmups[0]['started_at']) and s['values']['e2e_request_latency_seconds_count']==before]
        if fullpre:
            z=fullpre[-1]['values']
            count=b['values']['inter_token_latency_seconds_count']-z['inter_token_latency_seconds_count']
            assert count==66*127, 'Unexpected inter-token observation count'
            out['itl_including_warmups']={'observations':count,'mean_seconds':(b['values']['inter_token_latency_seconds_sum']-z['inter_token_latency_seconds_sum'])/count}
    match=next(s for s in smallcells if all(s[k]==c[k] for k in ['concurrency','stream','trial']))
    oldrows=loadlines(SMALL/f"serving/requests-{match['cell_id']:04d}.jsonl")
    assert {r['index']:r['prompt_sha256'] for r in rows} == {r['index']:r['prompt_sha256'] for r in oldrows}, 'Matched prompts differ'
    out['matched_0_5b']={'cell_id':match['cell_id'],'rps':match['successful_requests_per_second'],
        'p95_seconds':percentile([r['latency_seconds'] for r in oldrows]),
        'throughput_ratio_14b_to_0_5b':c['successful_requests_per_second']/match['successful_requests_per_second'],
        'p95_ratio_14b_to_0_5b':out['latency_p95']/percentile([r['latency_seconds'] for r in oldrows])}
    results.append(out)

groups=[]
for stream in [False,True]:
    for conc in [1,4,16,32]:
        group=[r for r in results if r['stream']==stream and r['concurrency']==conc]
        if not group: continue
        requests=[r for c in group for r in loadlines(ROOT/f"serving/requests-{c['cell_id']:04d}.jsonl")]
        groups.append(dict(concurrency=conc,stream=stream,trials=len(group),requests=len(requests),
            rps=len(requests)/sum(c['elapsed_seconds'] for c in group),
            p95_seconds=percentile([r['latency_seconds'] for r in requests]),
            gpu_util_mean=sum(c['gpu_utilization.gpu']['mean']*c['hardware_samples'] for c in group)/sum(c['hardware_samples'] for c in group),
            gpu_vram_mib=max(c['gpu_memory.used']['max'] for c in group)))

report={'scope':'9 fully successful cells only; seven interrupted cells excluded from performance aggregates',
 'hardware':{'samples':len(hardware),'first':hardware[0]['timestamp'],'last':hardware[-1]['timestamp'],
             'gpu_errors':sum('gpu_error' in s for s in hardware)},
 'histogram_counter_resets':resets,'per_cell':results,'groups':groups,
 'method':'Hardware and gauges matched by UTC timestamps within measured cell windows. Phase means use exact count deltas of 64 completions, excluding warmups; direct ITL includes two warmups. Sample means are arithmetic, not time weighted. p95 uses linear interpolation of raw request durations. 0.5B comparisons match mode, concurrency and trial.',
 'limitations':['Only nine complete cells; uneven replication and no complete SSE concurrency-1 cell.',
 'No physical GPU telemetry for 0.5B.', 'CPU, RAM and disk are host-wide; inference process CPU attribution unavailable.',
 'GPU utilization is sampled busy time, not proof of compute or bandwidth saturation.',
 'Model ID is reconstructed from setup; no downloaded model revision was recorded.',
 'Hardware recording and worker connectivity stop together; Slurm accounting is absent, so cause unconfirmed.']}
(ROOT/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
flat=[]
for r in results:
    flat.append(dict(cell_id=r['cell_id'],concurrency=r['concurrency'],mode='SSE' if r['stream'] else 'JSON',trial=r['trial'],
        requests=r['requests'],rps=r['successful_requests_per_second'],p95_seconds=r['latency_p95'],
        hardware_samples=r['hardware_samples'],gpu_util_mean_pct=r['gpu_utilization.gpu']['mean'],
        gpu_util_max_pct=r['gpu_utilization.gpu']['max'],gpu_vram_mean_mib=r['gpu_memory.used']['mean'],
        gpu_power_mean_w=r['gpu_power.draw']['mean'],node_cpu_mean_pct=r['cpu_percent']['mean'],
        kv_cache_max_pct=100*r['vllm:kv_cache_usage_perc']['max'],waiting_max=r['vllm:num_requests_waiting']['max'],
        **(r['phase_means_seconds'] or {})))
with (ROOT/'completed-cells.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)))
    writer.writeheader();writer.writerows(flat)
print(json.dumps({'groups':groups,'per_cell':flat,'phase_coverage':sum(bool(r['phase_means_seconds']) for r in results)},indent=2))

if __name__=='__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for stream,marker,label in [(False,'o','JSON'),(True,'s','SSE')]:
        data=[g for g in groups if g['stream']==stream]
        for ax,key,title,ylabel in [(axes[0,0],'rps','Completed-request throughput','Requests / second'),
              (axes[0,1],'p95_seconds','Client p95 latency','Seconds'),
              (axes[1,0],'gpu_util_mean','Mean sampled GPU utilization','GPU busy (%)'),
              (axes[1,1],'gpu_vram_mib','Peak sampled GPU memory','MiB')]:
            ax.plot([g['concurrency'] for g in data],[g[key] for g in data],marker=marker,label=label)
            ax.set(title=title,xlabel='Client concurrency (requests)',ylabel=ylabel,xticks=[1,4,16,32]);ax.grid(alpha=.2);ax.legend()
            if key=='gpu_util_mean': ax.set_ylim(0,100)
    fig.suptitle('Qwen2.5 14B BF16 on one A6000 — 576 requests in 9 complete cells')
    fig.supxlabel('Source: saved Oct 25, 2026 run, 20:11–20:26 Eastern. Uneven trial counts; failed cells excluded.',fontsize=9)
    fig.savefig(ROOT/'performance-and-gpu.png',dpi=170)
    fig.savefig(ROOT/'performance-and-gpu.svg')
    comparison,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    json_groups=[g for g in groups if not g['stream']]
    for model in ['14B','0.5B matched trials']:
        throughputs=[];latencies=[]
        for g in json_groups:
            selected=[r for r in results if not r['stream'] and r['concurrency']==g['concurrency']]
            if model=='14B':
                throughputs.append(g['rps']);latencies.append(g['p95_seconds'])
            else:
                old=[next(c for c in smallcells if c['cell_id']==r['matched_0_5b']['cell_id']) for r in selected]
                oldrequests=[q for c in old for q in loadlines(SMALL/f"serving/requests-{c['cell_id']:04d}.jsonl")]
                throughputs.append(len(oldrequests)/sum(c['elapsed_seconds'] for c in old))
                latencies.append(percentile([q['latency_seconds'] for q in oldrequests]))
        for ax,values in zip(axs,[throughputs,latencies]):
            ax.plot([g['concurrency'] for g in json_groups],values,marker='o',label=model)
    for ax,title,unit in zip(axs,['JSON throughput','JSON client p95 latency'],['Requests / second','Seconds']):
        ax.set(title=title,xlabel='Client concurrency (requests)',ylabel=unit,xticks=[1,4,16,32]);ax.grid(alpha=.2);ax.legend()
    comparison.suptitle('Same prompts and matched trial IDs — Qwen 14B versus 0.5B')
    comparison.supxlabel('Source: Oct 23 and Oct 25, 2026 saved baselines. Only trials with complete 14B cells included.',fontsize=9)
    comparison.savefig(ROOT/'model-comparison.png',dpi=170)
    comparison.savefig(ROOT/'model-comparison.svg')
    # Record hashes of evidence, script and derived outputs for later verification.
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.csv']
    with (ROOT/'SHA256SUMS.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle);writer.writerow(['path','bytes','sha256'])
        for p in sorted(files):
            writer.writerow([p.relative_to(ROOT).as_posix(),p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest()])
