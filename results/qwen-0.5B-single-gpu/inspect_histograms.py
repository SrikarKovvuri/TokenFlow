import json,re
from pathlib import Path
root=Path(__file__).resolve().parent
names=('request_queue_time_seconds','request_prefill_time_seconds','request_decode_time_seconds','inter_token_latency_seconds','e2e_request_latency_seconds')
regex=re.compile(r'^vllm:('+'|'.join(names)+r')_(count|sum)(\{.*\})?\s+(\S+)')
samples=[]
for line in (root/'vllm-metrics.jsonl').open():
    r=json.loads(line)
    d={}
    for text in r.get('raw_prometheus','').splitlines():
        m=regex.match(text)
        if m:d[m[1]+'_'+m[2]]=float(m[4])
    if len(d)==10:samples.append((r,d))
base=samples[0][1]['e2e_request_latency_seconds_count']
print('BASE/DELTA',base,{n:samples[-1][1][n+'_count']-samples[0][1][n+'_count'] for n in names})
cells=[json.loads(l) for l in (root/'serving/cells.jsonl').read_text().splitlines()]
for c in cells:
    if c['concurrency'] not in (16,32):continue
    i=c['cell_id']; before=base+i*66; measured=before+2; after=before+66
    available={n:sum(d['e2e_request_latency_seconds_count']==n for _,d in samples) for n in (before,measured,after)}
    print(i,c['concurrency'],c['stream'],c['trial'],available)
(root/'histogram-snapshots.json').write_text(json.dumps([{'timestamp':r['timestamp'],'scrape_started_at':r['scrape_started_at'],'scrape_finished_at':r['scrape_finished_at'],'values':d} for r,d in samples]))
