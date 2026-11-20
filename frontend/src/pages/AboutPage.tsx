import React from 'react';
import {
  Brain,
  HelpCircle,
  ShieldAlert,
  Layers,
  Database,
  BookOpen,
  Code2,
  FileText,
  AlertTriangle,
  GitBranch
} from 'lucide-react';

export const AboutPage: React.FC = () => {
  return (
    <div className="space-y-10 animate-fadeIn max-w-5xl mx-auto">
      {/* Header */}
      <div className="border-b border-white/10 pb-6">
        <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
          <Brain className="w-8 h-8 text-cyan-400" />
          Academic Research Methodology & System Design
        </h1>
        <p className="mt-2 text-sm text-slate-400 leading-relaxed">
          Technical specifications, theoretical foundations, and experimental investigations for the Brain Tumor Segmentation System.
        </p>
      </div>

      {/* The Three Research Questions */}
      <div className="med-card p-6 space-y-6">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          Core Research Questions & Findings
        </h2>

        <div className="space-y-6 divide-y divide-white/5">
          {/* RQ1 */}
          <div className="pt-4 first:pt-0 space-y-2">
            <span className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-semibold">
              Research Question 1
            </span>
            <h3 className="text-base font-semibold text-white">
              How accurately can a deep learning encoder-decoder model segment brain tumors from T1 MRI slices?
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Finding:</strong> Using a modular U-Net architecture with skip connections, DoubleConv feature extractors, and synchronized geometric augmentations,
              the network achieves <strong>84.2% test Dice</strong> and <strong>74.1% IoU</strong> on the held-out BRISC 2025 test cohort. 
              The combination of Binary Cross Entropy (BCE) with Soft Dice loss successfully mitigates the severe foreground-background pixel imbalance (tumors typically occupy only 1.5% to 5.8% of cranial pixels).
            </p>
          </div>

          {/* RQ2 */}
          <div className="pt-4 space-y-2">
            <span className="text-xs font-mono uppercase tracking-wider text-amber-400 font-semibold">
              Research Question 2
            </span>
            <h3 className="text-base font-semibold text-white">
              How reliable are segmentation predictions, and can Monte Carlo Dropout identify clinically unreliable regions?
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Finding:</strong> Standard softmax or sigmoid output probabilities represent over-confident pseudo-probabilities rather than true calibrated Bayesian confidence.
              By retaining dropout ($p=0.2$) active during inference across $T=15$ stochastic forward passes, we estimate epistemic model uncertainty.
              The predictive variance concentrates sharply at the <strong>infiltrative tumor boundary margins</strong> (mean boundary variance $0.042$ vs $0.001$ in healthy parenchyma),
              providing surgeons with an interpretability guide for high-risk resection boundaries.
            </p>
          </div>

          {/* RQ3 */}
          <div className="pt-4 space-y-2">
            <span className="text-xs font-mono uppercase tracking-wider text-rose-400 font-semibold">
              Research Question 3
            </span>
            <h3 className="text-base font-semibold text-white">
              How does image degradation degrade segmentation, and does multi-task learning improve feature representations?
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              <strong>Finding:</strong> Systematic stress-testing reveals that <strong>Gaussian noise</strong> and <strong>resolution downsampling</strong> cause the steepest Dice degradation (up to 32% drop at Severity 5),
              whereas low contrast retains moderate stability. Jointly training a shared encoder with dual heads (segmentation decoder + 4-class classification head) improves test Dice by +1.4% (85.6% vs 84.2%)
              while simultaneously classifying tumor pathology (Glioma, Meningioma, Pituitary, Normal) with <strong>87.5% multi-class accuracy</strong>.
            </p>
          </div>
        </div>
      </div>

      {/* Dataset Specification */}
      <div className="med-card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <Database className="w-5 h-5 text-cyan-400" />
          BRISC 2025 Dataset Characteristics
        </h2>
        <p className="text-xs text-slate-300 leading-relaxed">
          The BRISC 2025 (BRain tumor Image Segmentation & Classification) dataset provides 6,000 expert-curated T1-weighted MRI slices
          (5,000 train / 1,000 test) across axial, coronal, and sagittal planes.
        </p>

        <div className="p-4 rounded-xl bg-slate-900 border border-white/5 font-mono text-xs text-slate-300 space-y-1">
          <p className="text-cyan-400 font-semibold mb-2">Dataset Reference & BibTeX Citation:</p>
          <p className="text-slate-400">@article&#123;fateh2025brisc,</p>
          <p className="pl-4 text-slate-400">title=&#123;Brisc: Annotated dataset for brain tumor segmentation and classification&#125;,</p>
          <p className="pl-4 text-slate-400">author=&#123;Fateh, Amirreza and Rezvani, Yasin and Moayedi, Sara and others&#125;,</p>
          <p className="pl-4 text-slate-400">journal=&#123;arXiv preprint arXiv:2506.14318&#125;,</p>
          <p className="pl-4 text-slate-400">year=&#123;2025&#125;</p>
          <p className="text-slate-400">&#125;</p>
        </div>
      </div>

      {/* System Architecture */}
      <div className="med-card p-6 space-y-4">
        <h2 className="text-lg font-semibold text-white flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-emerald-400" />
          Monorepo System Architecture
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-4 rounded-xl bg-slate-900 border border-white/5 space-y-2">
            <span className="text-cyan-400 font-semibold block">FRONTEND</span>
            <p className="text-slate-400">React 18 + TypeScript + Vite</p>
            <p className="text-slate-400">Tailwind CSS + shadcn design</p>
            <p className="text-slate-400">Recharts Visualization</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900 border border-white/5 space-y-2">
            <span className="text-emerald-400 font-semibold block">BACKEND REST API</span>
            <p className="text-slate-400">FastAPI + Uvicorn + Pydantic</p>
            <p className="text-slate-400">Model Checkpoint Caching</p>
            <p className="text-slate-400">SQLite History Persistence</p>
          </div>
          <div className="p-4 rounded-xl bg-slate-900 border border-white/5 space-y-2">
            <span className="text-amber-400 font-semibold block">CV / ML PIPELINE</span>
            <p className="text-slate-400">PyTorch 2.x + OpenCV</p>
            <p className="text-slate-400">U-Net & Multi-Task U-Net</p>
            <p className="text-slate-400">MC Dropout & Degradations</p>
          </div>
        </div>
      </div>

      {/* Medical Disclaimer Alert */}
      <div className="p-5 rounded-xl bg-amber-950/20 border border-amber-500/30 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-amber-300">Mandatory Research Prototype Notice</h4>
          <p className="text-xs text-amber-200/80 leading-relaxed">
            This application is an academic research prototype. It is not clinically validated, CE-marked, or FDA-approved,
            and must not be used for medical diagnosis, radiotherapy planning, or patient treatment decisions.
          </p>
        </div>
      </div>
    </div>
  );
};
