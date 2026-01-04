import { useState, useMemo } from 'react';
import Papa from 'papaparse';
import KpiCard from './components/KpiCard.jsx';
import PlotCard from './components/PlotCard.jsx';

const BACKEND_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

function parseSummary(file) {
  return file.text().then((txt) => JSON.parse(txt));
}

function parseTimeseries(file) {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

async function parseTimeseriesFromUrl(url) {
  const txt = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`Failed to fetch timeseries: ${r.status}`);
    return r.text();
  });
  return new Promise((resolve, reject) => {
    Papa.parse(txt, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (res) => resolve(res.data),
      error: (err) => reject(err),
    });
  });
}

export default function App() {
  const [summary, setSummary] = useState(null);
  const [timeseries, setTimeseries] = useState([]);
  const [runLabel, setRunLabel] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Form for backend run
  const [traceContent, setTraceContent] = useState(`# id arrival_ms prompt_tokens gen_tokens streaming(0/1)
req1 0 200 400 0
req2 50 150 300 0
req3 100 180 320 0`);
  const [seed, setSeed] = useState('');
  const [configOpts, setConfigOpts] = useState({
    // GPU config (defaults for all GPUs)
    num_gpus: '1',
    vram_bytes: '4294967296',
    max_concurrent: '4',
    prefill_tps: '1200',
    decode_tps: '600',
    // Memory config
    kv_bytes_per_token: '2048',
    safe_reservation: '1',
    // Queue & scheduling
    max_queue: '64',
    max_retries: '2',
    scheduling: 'fifo',
    // Memory pressure
    memory_pressure_policy: 'reject',
    eviction_policy: 'lru',
    // Decode batching
    decode_sharing_cap: '8',
    decode_efficiency: '0.8',
    // Handoff (multi-GPU)
    handoff_bandwidth_gbps: '300',
    handoff_latency_us: '10',
    // Sampling
    timeseries_dt_ms: '20',
  });

  // Per-GPU overrides (only used when num_gpus > 1)
  const [perGpuConfig, setPerGpuConfig] = useState([]);

  // Update per-GPU config array when num_gpus changes
  const numGpus = parseInt(configOpts.num_gpus) || 1;
  const updatePerGpuConfig = (index, field, value) => {
    setPerGpuConfig((prev) => {
      const updated = [...prev];
      if (!updated[index]) updated[index] = {};
      updated[index] = { ...updated[index], [field]: value };
      return updated;
    });
  };

  const handleRunSim = async () => {
    setError('');
    setLoading(true);
    try {
      const body = {
        trace_content: traceContent,
        seed: seed ? Number(seed) : undefined,
        config_options: {},
      };
      // Include only non-empty config fields
      const perGpuFields = ['vram_bytes', 'prefill_tps', 'decode_tps', 'max_concurrent', 'decode_sharing_cap', 'decode_efficiency'];
      Object.entries(configOpts).forEach(([k, v]) => {
        if (v === '') return;
        // Skip per-GPU fields from global config when using multi-GPU
        if (numGpus > 1 && perGpuFields.includes(k)) return;
        body.config_options[k] = v;
      });
      // Include per-GPU overrides (format: "gpu 0 vram_bytes 600000")
      perGpuConfig.forEach((gpuConf, idx) => {
        if (!gpuConf) return;
        Object.entries(gpuConf).forEach(([field, value]) => {
          if (value !== '') {
            body.config_options[`gpu ${idx} ${field}`] = value;
          }
        });
      });
      const resp = await fetch(`${BACKEND_BASE}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const msg = await resp.text();
        throw new Error(msg || `Backend error ${resp.status}`);
      }
      const data = await resp.json();
      setSummary(data.summary);
      const ts = await parseTimeseriesFromUrl(`${BACKEND_BASE}${data.timeseries_url}`);
      setTimeseries(ts);
      setRunLabel(data.run_id);
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to run simulation.');
    } finally {
      setLoading(false);
    }
  };

  const derived = useMemo(() => {
    if (!timeseries.length) return { t: [], vram: [], queue: [], tps: [], globalQueue: [], vramPerGpu: [] };
    const t = timeseries.map((r) => r.time_ms);
    const vram = timeseries.map((r) => r.vram_used);
    const queue = timeseries.map((r) => r.queue_depth);
    const globalQueue = timeseries.map((r) => r.global_queue_depth ?? 0);
    const tps = timeseries.map((r, i) => {
      if (i === 0) return 0;
      const dt = (r.time_ms - timeseries[i - 1].time_ms) / 1000.0;
      return dt > 0 ? r.tokens_generated_delta / dt : 0;
    });
    // Extract per-GPU VRAM columns (vram_gpu0, vram_gpu1, etc.)
    const vramPerGpu = [];
    const firstRow = timeseries[0];
    for (let i = 0; ; i++) {
      const key = `vram_gpu${i}`;
      if (!(key in firstRow)) break;
      vramPerGpu.push({
        name: `GPU ${i}`,
        y: timeseries.map((r) => r[key] ?? 0),
      });
    }
    return { t, vram, queue, tps, globalQueue, vramPerGpu };
  }, [timeseries]);

  const kpis = summary
    ? [
        ['Throughput (tok/s)', summary.throughput_tokens_per_sec ?? '–'],
        ['p50 latency (ms)', summary.p50_latency_ms ?? '–'],
        ['p95 latency (ms)', summary.p95_latency_ms ?? '–'],
        ['p99 latency (ms)', summary.p99_latency_ms ?? '–'],
        ['p50 TTFT (ms)', summary.p50_ttft_ms ?? '–'],
        ['p95 TTFT (ms)', summary.p95_ttft_ms ?? '–'],
        ['Completion rate', summary.completion_rate ?? '–'],
        ['Reject rate', summary.reject_rate ?? '–'],
        ['Evictions', summary.evictions ?? '–'],
        ['Policy', summary.memory_pressure_policy ?? '–'],
        // Phase 8: Extended metrics
        ['Handoffs', summary.handoffs_total ?? '–'],
        ['Cross-GPU decodes', summary.cross_gpu_decodes ?? '–'],
        ['Retry success', summary.retry_attempts > 0 ? `${summary.retry_successes}/${summary.retry_attempts}` : '–'],
        ['Max global queue', summary.max_global_queue_depth ?? '–'],
      ]
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        <header className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">KV Sim Viewer</h1>
            <p className="text-sm text-slate-400">
              Enter a trace and config; backend runs C++ and we replot outputs
            </p>
          </div>
        </header>

        <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 shadow">
          <div className="text-sm font-semibold text-slate-200 mb-2">Run via backend</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
            <div>
              <label className="block text-slate-400 mb-1">Trace content</label>
              <textarea
                className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-2 h-32"
                value={traceContent}
                onChange={(e) => setTraceContent(e.target.value)}
                placeholder="# id arrival_ms prompt_tokens gen_tokens streaming(0/1)"
              />
            </div>
            <div>
              <label className="block text-slate-400 mb-1">Seed (optional)</label>
              <input
                className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1"
                value={seed}
                onChange={(e) => setSeed(e.target.value)}
              />
            </div>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm mt-3">
            {Object.entries(configOpts)
              .filter(([k]) => {
                // Hide per-GPU fields from global config when num_gpus > 1
                const perGpuFields = ['vram_bytes', 'prefill_tps', 'decode_tps', 'max_concurrent', 'decode_sharing_cap', 'decode_efficiency'];
                if (numGpus > 1 && perGpuFields.includes(k)) return false;
                return true;
              })
              .map(([k, v]) => (
              <div key={k}>
                <label className="block text-slate-400 mb-1">{k}</label>
                <input
                  className="w-full rounded bg-slate-800 border border-slate-700 px-2 py-1"
                  value={v}
                  onChange={(e) => setConfigOpts({ ...configOpts, [k]: e.target.value })}
                />
              </div>
            ))}
          </div>

          {/* Per-GPU configuration (shown when num_gpus > 1) */}
          {numGpus > 1 && (
            <div className="mt-4 border-t border-slate-700 pt-4">
              <div className="text-sm font-semibold text-slate-200 mb-2">
                Per-GPU Overrides <span className="text-slate-500 font-normal">(leave empty to use defaults above)</span>
              </div>
              <div className="space-y-3">
                {Array.from({ length: numGpus }, (_, idx) => (
                  <div key={idx} className="rounded bg-slate-800 p-3">
                    <div className="text-xs font-semibold text-emerald-400 mb-2">GPU {idx}</div>
                    <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-sm">
                      {['vram_bytes', 'prefill_tps', 'decode_tps', 'max_concurrent', 'decode_sharing_cap', 'decode_efficiency'].map((field) => (
                        <div key={field}>
                          <label className="block text-slate-500 text-xs mb-1">{field}</label>
                          <input
                            className="w-full rounded bg-slate-700 border border-slate-600 px-2 py-1 text-xs"
                            placeholder={configOpts[field] || '–'}
                            value={perGpuConfig[idx]?.[field] || ''}
                            onChange={(e) => updatePerGpuConfig(idx, field, e.target.value)}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="mt-3">
            <button
              onClick={handleRunSim}
              disabled={loading}
              className="px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-sm font-semibold disabled:opacity-60"
            >
              {loading ? 'Running...' : 'Run simulation'}
            </button>
          </div>
        </div>

        {error && <div className="text-sm text-red-400">{error}</div>}
        {summary && (
          <div className="text-sm text-slate-400">
            Loaded: <span className="text-slate-200 font-semibold">{runLabel}</span>
          </div>
        )}

        {summary && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
              {kpis.map(([label, value]) => (
                <KpiCard key={label} label={label} value={value} />
              ))}
            </div>

            {/* Per-GPU summary table */}
            {summary.per_gpu && summary.per_gpu.length > 1 && (
              <div className="rounded-lg border border-slate-800 bg-slate-900 p-4">
                <div className="text-sm font-semibold text-slate-200 mb-2">Per-GPU Breakdown</div>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-2">GPU</th>
                      <th className="text-right py-2">Requests</th>
                      <th className="text-right py-2">Tokens</th>
                      <th className="text-right py-2">Peak VRAM</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.per_gpu.map((gpu) => (
                      <tr key={gpu.gpu_index} className="border-b border-slate-800">
                        <td className="py-2">GPU {gpu.gpu_index}</td>
                        <td className="text-right py-2">{gpu.requests_finished}</td>
                        <td className="text-right py-2">{gpu.tokens_generated}</td>
                        <td className="text-right py-2">{(gpu.peak_vram_bytes / 1e6).toFixed(1)} MB</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <PlotCard
                title="VRAM over time (total)"
                x={derived.t}
                yTitle="bytes"
                traces={[{ name: 'Total VRAM', y: derived.vram }]}
              />
              {derived.vramPerGpu.length > 1 && (
                <PlotCard
                  title="VRAM per GPU"
                  x={derived.t}
                  yTitle="bytes"
                  traces={derived.vramPerGpu}
                />
              )}
              <PlotCard
                title="Queue depth over time"
                x={derived.t}
                yTitle="depth"
                traces={[
                  { name: 'Per-GPU queues', y: derived.queue },
                  { name: 'Global queue', y: derived.globalQueue },
                ]}
              />
              <PlotCard
                title="Tokens/sec over time"
                x={derived.t}
                yTitle="tokens/sec"
                traces={[{ name: 'Tokens/sec', y: derived.tps }]}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

