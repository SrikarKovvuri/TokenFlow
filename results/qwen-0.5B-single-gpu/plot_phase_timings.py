"""Visualize mean-latency accounting from saved counter deltas."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
report=json.loads((root/'phase-timing-analysis.json').read_text())
rows=report['pooled']
decode=np.array([r['phase_means_seconds']['request_decode_time_seconds']*1000 for r in rows])
other=np.array([(r['phase_means_seconds']['e2e_request_latency_seconds']-r['phase_means_seconds']['request_decode_time_seconds'])*1000 for r in rows])
residual=np.array([r['client_minus_engine_seconds']*1000 for r in rows])
x=np.arange(len(rows))
fig,ax=plt.subplots(figsize=(9,5.5),constrained_layout=True)
ax.bar(x,decode,.6,color='#176B87',label='vLLM decode interval')
ax.bar(x,other,.6,bottom=decode,color='#A6ADB4',label='Other within vLLM end-to-end interval')
ax.bar(x,residual,.6,bottom=decode+other,color='#A35423',label='Client minus vLLM end-to-end (unattributed)')
for i,r in enumerate(rows):
    ax.text(i,r['client_mean_seconds']*1000+12,f"{r['client_mean_seconds']*1000:.0f} ms",ha='center',fontsize=10)
ax.set_xticks(x,[f"{'SSE' if r['stream'] else 'JSON'}\nConcurrency {r['concurrency']}" for r in rows])
ax.set(ylabel='Mean elapsed time per measured request (milliseconds)',ylim=(0,1300),
       title='Where the extra mean latency appears: concurrency 16 versus 32')
ax.legend(loc='upper left',fontsize=9)
ax.spines[['right','top']].set_visible(False)
ax.grid(axis='y',alpha=.16)
ax.set_axisbelow(True)
fig.get_layout_engine().set(rect=(0,.12,1,.85))
fig.text(.06,.025,'Source: saved Qwen 0.5B run, 23 Oct 2026, 22:47–22:52 UTC. 128 measured requests per bar.\nPhase means use validated cumulative histogram deltas, excluding warmups. This is timing accounting, not GPU-time attribution.\nThe residual can include frontend, gateway, transport, and client work. It does not identify a specific bottleneck.',fontsize=8,color='.35')
fig.savefig(root/'phase-latency-breakdown.png',dpi=180)
fig.savefig(root/'phase-latency-breakdown.svg')
