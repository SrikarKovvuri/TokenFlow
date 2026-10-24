import json
from pathlib import Path
root=Path(__file__).resolve().parent
samples=json.loads((root/'histogram-snapshots.json').read_text())
base=samples[0]['values']['e2e_request_latency_seconds_count']; base_itl=samples[0]['values']['inter_token_latency_seconds_count']
cells=[json.loads(l) for l in (root/'serving/cells.jsonl').read_text().splitlines()]
for c in cells:
    if c['concurrency'] not in (16,32):continue
    expected=base+c['cell_id']*66+2
    pre=[s for s in samples if s['values']['e2e_request_latency_seconds_count']==expected]
    print(c['cell_id'],[(s['timestamp'],s['values']['inter_token_latency_seconds_count']-(base_itl+(expected-base)*127)) for s in pre])
