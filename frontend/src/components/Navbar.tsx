import React from 'react';
import {
  Brain,
  Activity,
  Layers,
  HelpCircle,
  ShieldAlert,
  BarChart3,
  History,
  Cpu,
  CheckCircle2,
  XCircle
} from 'lucide-react';
import { HealthStatus } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  health: HealthStatus | null;
}

export const Navbar: React.FC<NavbarProps> = ({ activeTab, setActiveTab, health }) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: Activity },
    { id: 'segment', label: 'Segmentation', icon: Layers },
    { id: 'uncertainty', label: 'Uncertainty', icon: HelpCircle },
    { id: 'robustness', label: 'Robustness Lab', icon: ShieldAlert },
    { id: 'performance', label: 'Performance', icon: BarChart3 },
    { id: 'history', label: 'History', icon: History },
    { id: 'about', label: 'About', icon: Brain },
  ];

  const isConnected = health && health.status === 'healthy';

  return (
    <header className="sticky top-0 z-50 bg-[#090d16]/90 backdrop-blur-md border-b border-white/10">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('dashboard')}>
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <span className="font-semibold text-base tracking-tight text-white flex items-center gap-1.5">
                NeuroSeg <span className="text-xs px-1.5 py-0.5 rounded font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">AI</span>
              </span>
              <p className="text-[10px] text-slate-400 font-mono tracking-wider uppercase">BRISC 2025 Brain MRI</p>
            </div>
          </div>

          {/* Nav Links */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-slate-800 text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                      : 'text-slate-300 hover:text-white hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* System & Hardware Health Badge */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900 border border-white/10 text-xs font-mono">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-slate-300">{health?.device || 'CPU'}</span>
            </div>
            
            <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-900 border border-white/10 text-xs font-mono">
              {isConnected ? (
                <>
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-emerald-400">API Live</span>
                </>
              ) : (
                <>
                  <span className="w-2 h-2 rounded-full bg-rose-400" />
                  <span className="text-rose-400">Offline</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};
