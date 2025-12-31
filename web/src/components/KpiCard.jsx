export default function KpiCard({ label, value }) {
  return (
    <div className="rounded-lg bg-slate-900 border border-slate-800 px-4 py-3 shadow">
      <div className="text-sm text-slate-400">{label}</div>
      <div className="text-lg font-semibold text-slate-50">{value}</div>
    </div>
  );
}

