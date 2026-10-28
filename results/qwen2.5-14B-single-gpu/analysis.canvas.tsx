import { H1, H2, LineChart, Text, useHostTheme, useState } from "cursor/canvas";

const observations = [
  {concurrency:1,mode:"JSON",trials:1,rps:.1768972842,p95:5.72457366,gpu:95.6846473},
  {concurrency:4,mode:"JSON",trials:1,rps:.6805786579,p95:5.968671415,gpu:96.6436170},
  {concurrency:16,mode:"JSON",trials:2,rps:2.560678247,p95:6.31890229,gpu:96.38},
  {concurrency:32,mode:"JSON",trials:1,rps:4.620062818,p95:6.953888545,gpu:95.4444444},
  {concurrency:4,mode:"SSE",trials:2,rps:.6634477828,p95:6.375620475,gpu:96.4779221},
  {concurrency:16,mode:"SSE",trials:1,rps:2.562371485,p95:6.416872925,gpu:96.48},
  {concurrency:32,mode:"SSE",trials:1,rps:4.588102194,p95:6.99852873,gpu:93.3571429},
];

export default function Analysis() {
  const theme = useHostTheme();
  const [mode,setMode] = useState("JSON");
  const selected = observations.filter(r=>r.mode===mode);
  const categories=selected.map(r=>String(r.concurrency));
  const cellStyle={padding:"10px 12px",borderBottom:`1px solid ${theme.stroke.secondary}`,textAlign:"right" as const};
  return <main style={{padding:24,maxWidth:1150,margin:"auto",color:theme.text.primary,background:theme.bg.editor,fontFamily:"system-ui",lineHeight:1.6}}>
    <H1>Qwen2.5 14B: decode dominates; batching still helps</H1>
    <Text>One A6000 · BF16 · vLLM 0.10.2 · 128 output tokens · 576 successful requests in nine complete cells</Text>
    <p style={{padding:16,background:theme.fill.secondary}}>
      JSON concurrency 16 → 32: throughput increased <strong>80%</strong> while p95 latency increased <strong>10%</strong>.
      High GPU busy time alone does not establish that further batching is ineffective.
    </p>
    <p>Partial run: seven interrupted cells excluded. GPU busy time and cache occupancy support a decode-heavy workload;
      they do not identify compute versus memory-bandwidth saturation.</p>
    <label>Response mode <select value={mode} onChange={e=>setMode(e.target.value)} style={{marginLeft:10,padding:6,color:theme.text.primary,background:theme.bg.elevated}}>
      <option>JSON</option><option>SSE</option>
    </select></label>
    <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(300px,1fr))",gap:24,marginTop:20}}>
      <section><H2>Completed-request throughput</H2><Text>Y: requests/second · X: client concurrency (requests)</Text>
        <LineChart categories={categories} series={[{name:mode,data:selected.map(r=>r.rps)}]} height={240} showValues />
      </section>
      <section><H2>Client p95 latency</H2><Text>Y: seconds · X: client concurrency (requests)</Text>
        <LineChart categories={categories} series={[{name:mode,data:selected.map(r=>r.p95)}]} height={240} valueSuffix=" s" showValues />
      </section>
      <section><H2>Mean sampled GPU busy time</H2><Text>Y: percent · X: client concurrency (requests)</Text>
        <LineChart categories={categories} series={[{name:mode,data:selected.map(r=>r.gpu)}]} height={240} yMax={100} valueSuffix="%" />
      </section>
      <section><H2>Where mean JSON latency went</H2>
        <p>At concurrency 16, decode averaged <strong>5.997 s</strong>. At 32 it averaged <strong>6.532 s</strong>.
          Decode explains about <strong>85%</strong> of the 0.626 s increase in mean client latency.</p>
        <p>Queue means were about 0.010–0.012 ms; prefill about 53–57 ms. Mean phase deltas cannot decompose p95.</p>
        <p>Engine inter-token gaps: about 47.1 → 51.2 ms, including two warmup requests per cell.</p>
      </section>
    </div>
    <Text>Source: saved request JSONL, Prometheus counters and hardware monitor · October 25, 2026, 20:11–20:26 Eastern.
      Throughput pools elapsed trial times; p95 pools request durations. Hardware means pool samples. Uneven replication.</Text>
    <H2>Completed groups</H2>
    <div style={{overflowX:"auto"}}><table style={{borderCollapse:"collapse",width:"100%"}}>
      <thead><tr>{["Mode","Concurrency","Trials","Requests/s","p95 (s)","GPU busy (%)"].map(h=><th key={h} style={cellStyle}>{h}</th>)}</tr></thead>
      <tbody>{selected.map(r=><tr key={r.concurrency}>{[r.mode,r.concurrency,r.trials,r.rps.toFixed(3),r.p95.toFixed(3),r.gpu.toFixed(1)].map((v,i)=><td key={i} style={cellStyle}>{v}</td>)}</tr>)}</tbody>
    </table></div>
    <H2>Memory, power and comparison</H2>
    <p>GPU reservation remained 43,836 MiB (42.81 GiB). Peak allocated-pool KV occupancy reached 21.72%; sampled waiting requests stayed zero.
      Most cells averaged roughly 295–298 W. These observations do not prove cache exhaustion or power throttling.</p>
    <p>The matched JSON-32 0.5B trial was about 6.2× faster in throughput and 5.4× lower in p95 latency.
      Runs were recorded on different dates, and 0.5B has no physical GPU telemetry.</p>
    <H2>Evidence limits</H2>
    <p>6,340 hardware samples, no GPU collection errors; all nine completed cells covered. CPU/RAM/disk are node-wide.
      SSE concurrency 1 has no successful cell. At 20:26:04 Eastern, the monitor recording and worker connectivity stopped together;
      Slurm accounting is needed to confirm why. No new experiments were run.</p>
  </main>;
}
