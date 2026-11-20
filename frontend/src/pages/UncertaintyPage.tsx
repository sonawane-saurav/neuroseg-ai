import React, { useState, useEffect, useRef } from 'react';
import {
  HelpCircle,
  Play,
  Download,
  Sliders,
  Info,
  RefreshCw,
  AlertCircle,
  FileImage,
  Layers,
  Sparkles
} from 'lucide-react';
import { UncertaintyResult, TestSample } from '../types';
import { runUncertainty, fetchSamples, base64ToBlob } from '../services/api';

export const UncertaintyPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<TestSample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);

  const [numPasses, setNumPasses] = useState<number>(15);
  const [threshold, setThreshold] = useState<number>(0.5);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UncertaintyResult | null>(null);

  const [displayMode, setDisplayMode] = useState<'overlay' | 'heatmap' | 'mask' | 'original'>('overlay');
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchSamples()
      .then((data) => {
        setSamples(data);
        if (data.length > 0) {
          handleSelectSample(data[0]);
        }
      })
      .catch((err) => console.log('Samples not yet prepared:', err));
  }, []);

  const handleSelectSample = (sample: TestSample) => {
    setSelectedSampleId(sample.id);
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setPreviewUrl(sample.image_base64);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setSelectedSampleId(null);
      setResult(null);
      setError(null);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleRunEvaluation = async () => {
    if (!selectedFile && !previewUrl) {
      setError('Please choose an MRI slice to evaluate.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      let fileToSend: Blob;
      let filename = 'mri_scan.png';

      if (selectedFile) {
        fileToSend = selectedFile;
        filename = selectedFile.name;
      } else if (previewUrl) {
        fileToSend = base64ToBlob(previewUrl);
        filename = `brisc_sample_${selectedSampleId || 1}.png`;
      } else {
        throw new Error('No image available.');
      }

      const res = await runUncertainty(fileToSend, filename, numPasses, threshold);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Uncertainty evaluation failed.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/10 pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <HelpCircle className="w-6 h-6 text-amber-400" />
            Uncertainty-Aware Tumor Segmentation
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Monte Carlo Dropout epistemic uncertainty estimation to quantify model confidence and detect borderline lesion ambiguities.
          </p>
        </div>

        {samples.length > 0 && (
          <div className="flex items-center gap-2 overflow-x-auto pb-1 max-w-full">
            <span className="text-xs font-mono text-slate-400 shrink-0">Test Slices:</span>
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => handleSelectSample(s)}
                className={`px-2.5 py-1 rounded text-xs font-mono shrink-0 transition-all ${
                  selectedSampleId === s.id
                    ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                    : 'bg-slate-850 hover:bg-slate-800 text-slate-400 border border-white/5'
                }`}
              >
                Sample #{s.id}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Controls Column (4 Cols) */}
        <div className="lg:col-span-4 space-y-6">
          <div className="med-card p-6">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <FileImage className="w-4 h-4 text-amber-400" />
              Target MRI Slice
            </h3>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".jpg,.jpeg,.png,.tif"
              className="hidden"
            />

            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-white/15 hover:border-amber-500/50 rounded-xl p-5 text-center cursor-pointer transition-all bg-slate-900/40 hover:bg-slate-900/80 group"
            >
              <FileImage className="w-8 h-8 text-amber-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-xs font-medium text-slate-200">Upload custom MRI or select test slice</p>
            </div>
          </div>

          <div className="med-card p-6 space-y-5">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-amber-400" />
              Monte Carlo Parameters
            </h3>

            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-slate-400">Stochastic Forward Passes (T):</span>
                <span className="text-amber-400 font-bold">{numPasses} passes</span>
              </div>
              <input
                type="range"
                min="5"
                max="30"
                step="5"
                value={numPasses}
                onChange={(e) => setNumPasses(parseInt(e.target.value))}
                className="w-full accent-amber-400 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                <span>T=5 (Fast)</span>
                <span>T=15 (Standard)</span>
                <span>T=30 (Exhaustive)</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-slate-400">Mean Cutoff Threshold:</span>
                <span className="text-slate-300 font-bold">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-slate-400 cursor-pointer"
              />
            </div>

            <button
              onClick={handleRunEvaluation}
              disabled={loading || (!selectedFile && !previewUrl)}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-400 hover:to-amber-500 disabled:opacity-50 text-slate-950 font-semibold text-sm transition-all shadow-lg shadow-amber-500/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Sampling T={numPasses} Passes...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  Run MC Dropout Analysis
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2">
                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Visual & Analytics Display (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          <div className="med-card p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-white/5">
                <button
                  onClick={() => setDisplayMode('overlay')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    displayMode === 'overlay' ? 'bg-amber-500/20 text-amber-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Uncertainty Overlay
                </button>
                <button
                  onClick={() => setDisplayMode('heatmap')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    displayMode === 'heatmap' ? 'bg-amber-500/20 text-amber-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Variance Heatmap (Plasma)
                </button>
                <button
                  onClick={() => setDisplayMode('mask')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    displayMode === 'mask' ? 'bg-amber-500/20 text-amber-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Mean Binary Mask
                </button>
                <button
                  onClick={() => setDisplayMode('original')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    displayMode === 'original' ? 'bg-amber-500/20 text-amber-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Original T1 MRI
                </button>
              </div>
            </div>

            <div className="relative aspect-square max-w-[500px] mx-auto rounded-xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center">
              {previewUrl ? (
                <img
                  src={
                    displayMode === 'overlay' && result
                      ? result.uncertainty_overlay_base64
                      : displayMode === 'heatmap' && result
                      ? result.heatmap_base64
                      : displayMode === 'mask' && result
                      ? result.mask_base64
                      : displayMode === 'original' && result
                      ? result.original_base64
                      : previewUrl
                  }
                  alt="Uncertainty Map"
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="text-center p-8 text-slate-500">
                  <HelpCircle className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm font-medium">No MRI Scan Loaded</p>
                </div>
              )}
            </div>
          </div>

          {/* Uncertainty Metrics Cards */}
          {result && (
            <div className="med-card p-6 space-y-6">
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-amber-400" />
                Epistemic Variance Diagnostics
              </h3>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">SAMPLING PASSES</span>
                  <p className="text-lg font-bold font-mono text-white mt-1">T = {result.num_passes}</p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">MEAN VARIANCE</span>
                  <p className="text-lg font-bold font-mono text-amber-400 mt-1">
                    {result.mean_uncertainty.toFixed(5)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">BOUNDARY UNCERTAINTY</span>
                  <p className="text-lg font-bold font-mono text-rose-400 mt-1">
                    {result.mean_boundary_uncertainty.toFixed(5)}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">HIGH-UNCERTAIN PIXELS</span>
                  <p className="text-lg font-bold font-mono text-white mt-1">
                    {result.high_uncertainty_pixels.toLocaleString()}
                  </p>
                </div>
              </div>

              {/* Research Methodology Note */}
              <div className="p-4 rounded-xl bg-slate-900/80 border border-white/5 text-xs text-slate-300 space-y-2">
                <div className="flex items-center gap-2 text-amber-400 font-semibold">
                  <Info className="w-4 h-4 shrink-0" />
                  <span>Clinical Interpretation & Uncertainty vs. Accuracy</span>
                </div>
                <p className="leading-relaxed text-slate-400">
                  Epistemic uncertainty reflects the model's posterior distribution variance over weights $q(W)$.
                  Notice that uncertainty concentrates sharply along <strong>tumor transition perimeters</strong> (where partial volume effects and diffuse infiltration occur),
                  while remaining near zero in clear background parenchyma. Uncertainty is not a diagnosis and must be interpreted as a surgical boundary margin guide.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
