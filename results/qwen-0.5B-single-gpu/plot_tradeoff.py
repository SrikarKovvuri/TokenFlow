"""Plot throughput versus pooled p95 from the actual request records."""
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root = Path(__file__).resolve().parent
cells = [json.loads(line) for line in (root/'serving/cells.jsonl').read_text().splitlines()]
groups = defaultdict(list)
for c in cells:
    groups[c['stream'],c['concurrency']].append(c)
points = []
for (stream, concurrency), trials in sorted(groups.items()):
    rows = [json.loads(line) for c in trials for line in (root/f"serving/requests-{c['cell_id']:04d}.jsonl").read_text().splitlines()]
    good = [r for r in rows if r.get('status_code') == 200 and 'error' not in r]
    points.append(dict(stream=stream, concurrency=concurrency,
        rps=len(good)/sum(c['elapsed_seconds'] for c in trials),
        p95=float(np.quantile([r['latency_seconds'] for r in good],.95))))
changes = []
for stream in (False, True):
    series = [p for p in points if p['stream'] == stream]
    for a,b in zip(series,series[1:]):
        changes.append(dict(stream=stream, start=a['concurrency'], end=b['concurrency'],
            extra_rps=b['rps']-a['rps'], extra_p95_seconds=b['p95']-a['p95'],
            throughput_gain_percent=100*(b['rps']/a['rps']-1),
            p95_increase_percent=100*(b['p95']/a['p95']-1)))
report = dict(points=points,marginal_changes=changes)
(root/'tradeoff-analysis.json').write_text(json.dumps(report,indent=2))
fig,ax=plt.subplots(figsize=(9,5.8),constrained_layout=True)
for stream,label,color,marker in [(False,'JSON','#176B87','o'),(True,'SSE','#A35423','s')]:
    s=[p for p in points if p['stream']==stream]
    t=[c for c in cells if c['stream']==stream]
    ax.scatter([c['successful_requests_per_second'] for c in t],[c['latency_p95_seconds'] for c in t],color=color,alpha=.22,s=35,marker=marker)
    ax.plot([p['rps'] for p in s],[p['p95'] for p in s],marker=marker,color=color,linewidth=1.6,label=label)
    for p in s:
        ax.annotate(f"C={p['concurrency']}",(p['rps'],p['p95']),xytext=(5,9 if stream else -16),textcoords='offset points',fontsize=9,color=color)
ax.set(xlabel='Completed requests per second (requests/s)',ylabel='End-to-end p95 request latency (seconds)',title='Qwen 0.5B: throughput–latency tradeoff',xlim=(0,35),ylim=(0,1.6))
ax.legend(title='Response mode; C = client concurrency')
ax.grid(alpha=.18)
ax.spines[['top','right']].set_visible(False)
fig.get_layout_engine().set(rect=(0,.10,1,.88))
fig.text(.08,.025,'Source: saved Qwen run, 23 Oct 2026, 22:47–22:52 UTC. Outputs: 128 tokens.\nLines: pooled raw requests across two trials; faint points: individual trials. Gateway and SSH tunnel included.',fontsize=8,color='.35')
fig.savefig(root/'throughput-latency.png',dpi=180)
fig.savefig(root/'throughput-latency.svg')
print(json.dumps(report,indent=2))
