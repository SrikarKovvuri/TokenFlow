"""Build a dependency-free interactive HTML report from the offline analysis."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
data = json.loads((root / 'analysis.json').read_text(encoding='utf-8'))
template = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen 14B serving analysis</title>
<style>
:root{color-scheme:light dark;font:16px/1.6 system-ui}body{max-width:1120px;margin:auto;padding:28px}h1{font-size:26px}h2{font-size:20px}small{opacity:.75}.lead{border-left:4px solid steelblue;padding:12px 18px;background:light-dark(#f2f5f7,#20272b)}select{padding:8px;font:inherit} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:24px}svg{width:100%;height:auto}svg text{fill:currentColor;font:12px system-ui}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:right;border-bottom:1px solid #8886}details{margin:24px 0}button{font:inherit;padding:8px} .scroll{overflow:auto}
</style>
<h1>Qwen2.5 14B: decode dominates; batching still helps</h1>
<p>One A6000 · BF16 · vLLM 0.10.2 · 576 successful measured requests · October 25, 2026</p>
<p class="lead">JSON concurrency 16 → 32: <strong>80% more throughput</strong> with <strong>10% higher p95 latency</strong>. Decode explains about 85% of the increase in mean client latency.</p>
<p>Partial run: nine successful cells, seven interrupted cells excluded. No new experiments. High GPU busy time does not identify compute versus bandwidth saturation.</p>
<label>Response mode <select id="mode"><option value="false">JSON</option><option value="true">SSE</option></select></label>
<div class="grid" id="charts"></div>
<small>Source: analysis.json, derived from saved request JSONL and timestamped hardware/engine telemetry. Successful measured windows: 20:11–20:26 Eastern. Uneven trial counts. Hover over points for values.</small>
<h2>Completed groups</h2><div class="scroll"><table><thead><tr><th>Concurrency</th><th>Trials</th><th>Requests</th><th>Requests/s</th><th>p95 (s)</th><th>GPU busy (%)</th><th>Peak VRAM (MiB)</th></tr></thead><tbody id="rows"></tbody></table></div>
<h2>Queue, prefill and decode by successful cell</h2>
<p>Means from exact 64-completion counter deltas; warmups excluded. These do not decompose p95. Select response mode above to filter.</p>
<div class="scroll"><table><thead><tr><th>Concurrency</th><th>Trial</th><th>Queue (ms)</th><th>Prefill (ms)</th><th>Decode (s)</th><th>Client mean (s)</th><th>Hardware samples</th></tr></thead><tbody id="phases"></tbody></table></div>
<h2>What the evidence supports</h2>
<ul><li>GPU busy time was already around 96% at JSON concurrency 1, yet batching improved aggregate throughput substantially.</li><li>GPU memory reservation stayed at 43,836 MiB (42.81 GiB). Peak allocated-pool KV occupancy was below 22%; sampled waiting requests stayed zero.</li><li>Most cells averaged approximately 295–298 W. Clocks and throttle reasons were not recorded, so power throttling is unproven.</li><li>At JSON concurrency 32, throughput was about 6.2× lower than the matched 0.5B trial; p95 was about 5.4× higher. Separate run dates and no physical GPU telemetry for 0.5B limit comparison.</li></ul>
<details><summary>Coverage and limitations</summary><ul id="limits"></ul><p>6,340 hardware samples; all nine successful cells covered; no reported GPU collection errors. Hardware and worker connectivity end together at approximately 20:26:04 Eastern. Slurm accounting is needed to confirm the interruption cause.</p></details>
<p><a href="README.md">Full written analysis</a> · <a href="analysis.json">Detailed data</a> · <a href="completed-cells.csv">Successful cells CSV</a> · <a href="model-comparison.png">0.5B comparison chart</a></p>
<script>
const report=__DATA__;
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs={},text){const node=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));if(text!==undefined)node.textContent=text;return node;}
function chart(rows,key,title,units){
 const section=document.createElement('section');const h=document.createElement('h2');h.textContent=title;section.append(h);
 const svg=el('svg',{viewBox:'0 0 480 290',role:'img','aria-label':title+' by client concurrency'});const max=key==='gpu_util_mean'?100:Math.max(...rows.map(r=>r[key]))*1.12;
 const x=v=>62+(v-1)/31*390,y=v=>230-v/max*180;
 for(let i=0;i<=4;i++){const v=max*i/4;svg.append(el('line',{x1:62,y1:y(v),x2:452,y2:y(v),stroke:'currentColor',opacity:.15}));svg.append(el('text',{x:54,y:y(v)+4,'text-anchor':'end'},v.toFixed(max>100?0:1)));}
 svg.append(el('text',{x:250,y:278,'text-anchor':'middle'},'Client concurrency (requests)'));
 svg.append(el('text',{x:16,y:140,transform:'rotate(-90 16 140)','text-anchor':'middle'},units));
 svg.append(el('polyline',{points:rows.map(r=>`${x(r.concurrency)},${y(r[key])}`).join(' '),fill:'none',stroke:'steelblue','stroke-width':3}));
 rows.forEach(r=>{svg.append(el('text',{x:x(r.concurrency),y:252,'text-anchor':'middle'},r.concurrency));const dot=el('circle',{cx:x(r.concurrency),cy:y(r[key]),r:6,fill:'steelblue',tabindex:0});dot.append(el('title',{},`${r.concurrency} concurrent: ${r[key].toFixed(3)} ${units}; ${r.trials} trial(s)`));svg.append(dot);});
 section.append(svg);return section;
}
function row(target,values){const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td);});target.append(tr);}
function render(){const stream=document.querySelector('#mode').value==='true';const groups=report.groups.filter(r=>r.stream===stream);const plots=document.querySelector('#charts');plots.replaceChildren();
 [['rps','Completed-request throughput','Requests / second'],['p95_seconds','Client p95 latency','Seconds'],['gpu_util_mean','Mean sampled GPU busy time','GPU busy (%)'],['gpu_vram_mib','Peak sampled GPU memory','MiB']].forEach(([k,t,u])=>plots.append(chart(groups,k,t,u)));
 const rows=document.querySelector('#rows');rows.replaceChildren();groups.forEach(r=>row(rows,[r.concurrency,r.trials,r.requests,r.rps.toFixed(3),r.p95_seconds.toFixed(3),r.gpu_util_mean.toFixed(1),r.gpu_vram_mib.toFixed(0)]));
 const phases=document.querySelector('#phases');phases.replaceChildren();report.per_cell.filter(r=>r.stream===stream).sort((a,b)=>a.concurrency-b.concurrency||a.trial-b.trial).forEach(r=>{const p=r.phase_means_seconds;if(!p)return;row(phases,[r.concurrency,r.trial,(1000*p.request_queue_time_seconds).toFixed(3),(1000*p.request_prefill_time_seconds).toFixed(1),p.request_decode_time_seconds.toFixed(3),r.latency_mean.toFixed(3),r.hardware_samples]);});
}
report.limitations.forEach(s=>{const li=document.createElement('li');li.textContent=s;document.querySelector('#limits').append(li);});
document.querySelector('#mode').addEventListener('change',render);render();
</script></html>'''
(root/'report.html').write_text(template.replace('__DATA__',json.dumps(data).replace('</','<\\/')),encoding='utf-8')
print(root/'report.html')
