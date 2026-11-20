import React, { useState, useEffect } from 'react';
import {
  History,
  Trash2,
  Search,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ShieldAlert,
  RefreshCw,
  Clock,
  Layers
} from 'lucide-react';
import { HistoryItem } from '../types';
import { fetchHistory, clearHistory } from '../services/api';

export const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const loadHistory = () => {
    setLoading(true);
    fetchHistory()
      .then((data) => setHistory(data))
      .catch((err) => console.log('History fetch error:', err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleClearHistory = async () => {
    if (confirm('Are you sure you want to clear all analysis history records?')) {
      await clearHistory();
      loadHistory();
    }
  };

  const filteredHistory = history.filter((item) =>
    item.filename.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.model_used.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/10 pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <History className="w-6 h-6 text-cyan-400" />
            Analysis Audit History
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Persistent SQLite database log of all MRI segmentation passes, uncertainty evaluations, and robustness experiments.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={loadHistory}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 border border-white/10 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Log
          </button>
          {history.length > 0 && (
            <button
              onClick={handleClearHistory}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-xs font-mono text-rose-400 border border-rose-500/30 transition-all"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Clear Records
            </button>
          )}
        </div>
      </div>

      {/* Search & Filter Bar */}
      <div className="med-card p-4 flex items-center gap-3">
        <Search className="w-4 h-4 text-slate-400 shrink-0" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by slice filename or model name..."
          className="bg-transparent text-sm text-slate-200 placeholder:text-slate-500 w-full focus:outline-none font-mono"
        />
      </div>

      {/* History Table */}
      <div className="med-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-white/10 text-slate-400 bg-slate-900/60">
                <th className="py-3 px-4">ID</th>
                <th className="py-3 px-4">Timestamp (UTC)</th>
                <th className="py-3 px-4">Slice Filename</th>
                <th className="py-3 px-4">Model Pipeline</th>
                <th className="py-3 px-4">Lesion Delineated</th>
                <th className="py-3 px-4">Tumor Area (%)</th>
                <th className="py-3 px-4">Pathology / Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-slate-300">
              {filteredHistory.length > 0 ? (
                filteredHistory.map((item) => (
                  <tr key={item.id} className="hover:bg-slate-900/40 transition-colors">
                    <td className="py-3 px-4 font-bold text-slate-500">#{item.id}</td>
                    <td className="py-3 px-4 text-slate-400">
                      {item.timestamp ? item.timestamp.replace('T', ' ').slice(0, 19) : 'Just now'}
                    </td>
                    <td className="py-3 px-4 font-semibold text-white truncate max-w-[180px]">
                      {item.filename}
                    </td>
                    <td className="py-3 px-4">
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] bg-slate-800 text-slate-300 border border-white/5">
                        {item.model_used}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {item.tumor_detected ? (
                        <span className="inline-flex items-center gap-1 text-rose-400">
                          <AlertCircle className="w-3.5 h-3.5" />
                          {item.tumor_pixel_count.toLocaleString()} px
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 text-emerald-400">
                          <CheckCircle2 className="w-3.5 h-3.5" /> None
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-cyan-400 font-bold">
                      {item.tumor_area_percentage}%
                    </td>
                    <td className="py-3 px-4">
                      {item.classification_label ? (
                        <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/30">
                          {item.classification_label}
                        </span>
                      ) : item.has_uncertainty ? (
                        <span className="text-amber-400">
                          Var: {item.mean_uncertainty.toFixed(4)}
                        </span>
                      ) : item.degradation_type !== 'none' ? (
                        <span className="text-rose-400 uppercase">
                          {item.degradation_type.replace('_', ' ')} (Sev {item.severity})
                        </span>
                      ) : (
                        <span className="text-slate-500">—</span>
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-500">
                    <History className="w-8 h-8 mx-auto mb-2 opacity-30" />
                    <p className="text-sm font-medium">No Analysis History Found</p>
                    <p className="text-xs mt-1">Run segmentations in the workspace to log experimental records</p>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
