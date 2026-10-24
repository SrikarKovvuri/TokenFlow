"""Offline phase attribution using count-validated saved Prometheus snapshots."""
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
samples=json.loads((ROOT/'histogram-snapshots.json').read_text())
cells=[json.loads(line) for line in (ROOT/'serving/cells.jsonl').read_text().splitlines()]
PHASES=['request_queue_time_seconds','request_prefill_time_seconds','request_decode_time_seconds','e2e_request_latency_seconds']
base=samples[0]['values']['e2e_request_latency_seconds_count']
base_itl=samples[0]['values']['inter_token_latency_seconds_count']
expected_total=sum(c['requests']+len(json.loads((ROOT/f"serving/warmup-{c['cell_id']:04d}.json").read_text())) for c in cells)
assert samples[-1]['values']['e2e_request_latency_seconds_count']-base == expected_total
for prev,cur in zip(samples,samples[1:]):
    for key in prev['values']:
        assert cur['values'][key]>=prev['values'][key]-1e-8, ('Counter reset',key)

def snapshot(count, after=None, before=None, latest=False):
    candidates=[s for s in samples
        if all(s['values'][p+'_count']==count for p in PHASES)
        and (after is None or datetime.fromisoformat(s['timestamp'])>=after)
        and (before is None or datetime.fromisoformat(s['scrape_finished_at'])<=before)]
    assert candidates, ('Missing complete histogram boundary',count)
    return candidates[-1] if latest else candidates[0]

per_cell=[]
preceding=0
for c in cells:
    warmups=json.loads((ROOT/f"serving/warmup-{c['cell_id']:04d}.json").read_text())
    all_before=base+preceding
    measured_before=all_before+len(warmups)
    after=measured_before+c['requests']
    if c['concurrency'] in (16,32):
        pre=snapshot(measured_before)
        post=snapshot(after,after=datetime.fromisoformat(pre['timestamp']))
        fullpre=snapshot(all_before,before=datetime.fromisoformat(warmups[0]['started_at']),latest=True)
        values={}
        for p in PHASES:
            delta_count=post['values'][p+'_count']-pre['values'][p+'_count']
            delta_sum=post['values'][p+'_sum']-pre['values'][p+'_sum']
            assert delta_count==c['requests']
            values[p]={'count':delta_count,'sum':delta_sum,'mean_seconds':delta_sum/delta_count}
        rows=[json.loads(line) for line in (ROOT/f"serving/requests-{c['cell_id']:04d}.jsonl").read_text().splitlines()]
        assert all(r['status_code']==200 and r['usage']['completion_tokens']==128 for r in rows)
        itl_count=post['values']['inter_token_latency_seconds_count']-fullpre['values']['inter_token_latency_seconds_count']
        itl_sum=post['values']['inter_token_latency_seconds_sum']-fullpre['values']['inter_token_latency_seconds_sum']
        assert itl_count==(c['requests']+len(warmups))*127
        per_cell.append({'cell_id':c['cell_id'],'concurrency':c['concurrency'],'stream':c['stream'],'trial':c['trial'],
            'measured_requests':len(rows),'boundary_pre':pre['timestamp'],'boundary_post':post['timestamp'],
            'phases':values,'client_mean_seconds':statistics.mean(r['latency_seconds'] for r in rows),
            'client_p95_seconds':c['latency_p95_seconds'],
            'itl_including_warmups':{'count':itl_count,'sum':itl_sum,'mean_seconds':itl_sum/itl_count,
                                     'requests':c['requests']+len(warmups)},
            'decode_derived_token_gap_seconds':values['request_decode_time_seconds']['mean_seconds']/127})
    preceding+=c['requests']+len(warmups)

groups=defaultdict(list)
for c in per_cell:groups[c['stream'],c['concurrency']].append(c)
pooled=[]
for (stream,concurrency),rows in sorted(groups.items()):
    n=sum(r['measured_requests'] for r in rows)
    means={p:sum(r['phases'][p]['sum'] for r in rows)/n for p in PHASES}
    client=sum(r['client_mean_seconds']*r['measured_requests'] for r in rows)/n
    engine_other=means['e2e_request_latency_seconds']-sum(means[p] for p in PHASES[:3])
    pooled.append({'stream':stream,'concurrency':concurrency,'measured_requests':n,
        'phase_means_seconds':means,'client_mean_seconds':client,
        'other_engine_seconds':engine_other,
        'client_minus_engine_seconds':client-means['e2e_request_latency_seconds'],
        'itl_with_warmups_mean_seconds':sum(r['itl_including_warmups']['sum'] for r in rows)/sum(r['itl_including_warmups']['count'] for r in rows),
        'decode_derived_token_gap_seconds':means['request_decode_time_seconds']/127})
changes=[]
for stream in (False,True):
    a=next(p for p in pooled if p['stream']==stream and p['concurrency']==16)
    b=next(p for p in pooled if p['stream']==stream and p['concurrency']==32)
    delta=b['client_mean_seconds']-a['client_mean_seconds']
    components={p:b['phase_means_seconds'][p]-a['phase_means_seconds'][p] for p in PHASES[:3]}
    components['other_engine_seconds']=b['other_engine_seconds']-a['other_engine_seconds']
    components['client_minus_engine_seconds']=b['client_minus_engine_seconds']-a['client_minus_engine_seconds']
    assert abs(sum(components.values())-delta)<1e-8
    changes.append({'stream':stream,'client_mean_increase_seconds':delta,'components_seconds':components,
        'share_of_mean_increase':{k:v/delta for k,v in components.items()}})
report={'method':'Per-request phase means from exact 64-completion counter deltas; two trials pooled. ITL direct deltas include two warmups per cell. No p95 decomposition.',
        'total_completions_validated':expected_total,'counter_resets':False,'per_cell':per_cell,'pooled':pooled,'changes_16_to_32':changes}
(ROOT/'phase-timing-analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'pooled':pooled,'changes_16_to_32':changes},indent=2))
