import Plot from 'react-plotly.js';

export default function PlotCard({ title, x, traces }) {
  const data = traces.map((t) => ({
    x,
    y: t.y,
    type: 'scatter',
    mode: 'lines',
    name: t.name,
    line: { width: 2 },
  }));

  return (
    <div className="rounded-lg bg-slate-900 border border-slate-800 p-4 shadow">
      <div className="text-sm text-slate-200 mb-2">{title}</div>
      <Plot
        data={data}
        layout={{
          autosize: true,
          height: 320,
          margin: { l: 50, r: 20, t: 20, b: 40 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e2e8f0' },
          xaxis: { title: 'time (ms)' },
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

