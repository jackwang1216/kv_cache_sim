import Plot from 'react-plotly.js';

const GRID_COLOR = '#334155'; // slate-700
const ZERO_COLOR = '#475569'; // slate-600
const COLORS = ['#60a5fa', '#f472b6', '#34d399', '#fbbf24'];

export default function PlotCard({ title, x, traces, yTitle }) {
  const data = traces.map((t, idx) => ({
    x,
    y: t.y,
    type: 'scatter',
    mode: 'lines+markers',
    name: t.name,
    line: { width: 2, color: COLORS[idx % COLORS.length] },
    marker: { size: 4, color: COLORS[idx % COLORS.length] },
    hovertemplate: `%{y:.2s}<extra>${t.name || ''}</extra>`,
  }));

  return (
    <div className="rounded-lg bg-slate-900 border border-slate-800 p-4 shadow">
      <div className="text-sm text-slate-200 mb-2">{title}</div>
      <Plot
        data={data}
        layout={{
          autosize: true,
          height: 360,
          margin: { l: 60, r: 20, t: 10, b: 50 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: { color: '#e2e8f0' },
          hovermode: 'x unified',
          xaxis: {
            title: 'time (ms)',
            gridcolor: GRID_COLOR,
            zerolinecolor: ZERO_COLOR,
            tickfont: { color: '#cbd5e1' },
            titlefont: { color: '#cbd5e1' },
          },
          yaxis: {
            title: yTitle || '',
            gridcolor: GRID_COLOR,
            zerolinecolor: ZERO_COLOR,
            tickfont: { color: '#cbd5e1' },
            titlefont: { color: '#cbd5e1' },
          },
          showlegend: traces.length > 1,
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%', height: '100%' }}
      />
    </div>
  );
}

