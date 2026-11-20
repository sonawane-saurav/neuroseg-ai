import React, { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { MedicalDisclaimer } from './components/MedicalDisclaimer';
import { DashboardPage } from './pages/DashboardPage';
import { SegmentationWorkspacePage } from './pages/SegmentationWorkspacePage';
import { UncertaintyPage } from './pages/UncertaintyPage';
import { RobustnessLabPage } from './pages/RobustnessLabPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { HistoryPage } from './pages/HistoryPage';
import { AboutPage } from './pages/AboutPage';

import { HealthStatus, ModelInfo, MetricsData } from './types';
import { fetchHealth, fetchModelInfo, fetchMetrics } from './services/api';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);

  useEffect(() => {
    // Initial fetch
    fetchHealth()
      .then(setHealth)
      .catch((err) => console.log('Backend not connected yet:', err));

    fetchModelInfo()
      .then(setModelInfo)
      .catch((err) => console.log('Model info not loaded yet:', err));

    fetchMetrics()
      .then(setMetrics)
      .catch((err) => console.log('Metrics not loaded yet:', err));

    // Periodic heartbeat every 10s
    const timer = setInterval(() => {
      fetchHealth().then(setHealth).catch(() => setHealth(null));
    }, 10000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-[#090d16] flex flex-col">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} health={health} />
      <MedicalDisclaimer />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <DashboardPage
            health={health}
            modelInfo={modelInfo}
            metrics={metrics}
            onNavigate={setActiveTab}
          />
        )}
        {activeTab === 'segment' && <SegmentationWorkspacePage />}
        {activeTab === 'uncertainty' && <UncertaintyPage />}
        {activeTab === 'robustness' && <RobustnessLabPage />}
        {activeTab === 'performance' && <ModelPerformancePage metrics={metrics} />}
        {activeTab === 'history' && <HistoryPage />}
        {activeTab === 'about' && <AboutPage />}
      </main>

      <footer className="border-t border-white/5 py-6 bg-slate-950/60 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>NeuroSeg AI v1.0.0 — BRISC 2025 Brain MRI Research Platform</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
