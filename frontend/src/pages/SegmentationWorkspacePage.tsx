import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  Layers,
  Sparkles,
  Download,
  Sliders,
  CheckCircle2,
  AlertCircle,
  FileImage,
  RefreshCw,
  Compass,
  Maximize2
} from 'lucide-react';
import { SegmentationResult, TestSample } from '../types';
import { runSegmentation, fetchSamples, base64ToBlob } from '../services/api';

export const SegmentationWorkspacePage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<TestSample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);

  const [threshold, setThreshold] = useState<number>(0.5);
  const [useMultitask, setUseMultitask] = useState<boolean>(true);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentationResult | null>(null);

  const [activeView, setActiveView] = useState<'overlay' | 'mask' | 'original' | 'split'>('overlay');
  const [sliderPos, setSliderPos] = useState<number>(50);

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

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setSelectedSampleId(null);
      setResult(null);
      setError(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleSelectSample = (sample: TestSample) => {
    setSelectedSampleId(sample.id);
    setSelectedFile(null);
    setResult(null);
    setError(null);
    setPreviewUrl(sample.image_base64);
  };

  const handleRunSegmentation = async () => {
    if (!selectedFile && !previewUrl) {
      setError('Please select an MRI image or choose a preloaded sample.');
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

      const res = await runSegmentation(fileToSend, filename, threshold, useMultitask);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Segmentation failed. Check backend status.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadMask = () => {
    if (!result) return;
    const link = document.createElement('a');
    link.href = result.mask_base64;
    link.download = `tumor_mask_${Date.now()}.png`;
    link.click();
  };

  return (
    <div className="space-y-8 animate-fadeIn">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-white/10 pb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2.5">
            <Layers className="w-6 h-6 text-cyan-400" />
            Tumor Segmentation Workspace
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Upload patient T1-weighted MRI slices or select validated BRISC 2025 cohort samples for pixel-level lesion delineation.
          </p>
        </div>

        {/* Preloaded Samples Bar */}
        {samples.length > 0 && (
          <div className="flex items-center gap-2 overflow-x-auto pb-1 max-w-full">
            <span className="text-xs font-mono text-slate-400 shrink-0">Test Slices:</span>
            {samples.map((s) => (
              <button
                key={s.id}
                onClick={() => handleSelectSample(s)}
                className={`px-2.5 py-1 rounded text-xs font-mono shrink-0 transition-all ${
                  selectedSampleId === s.id
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                    : 'bg-slate-850 hover:bg-slate-800 text-slate-400 border border-white/5'
                }`}
              >
                Sample #{s.id}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Input, Upload & Parameters (4 Cols) */}
        <div className="lg:col-span-4 space-y-6">
          {/* Upload Area */}
          <div className="med-card p-6">
            <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
              <Upload className="w-4 h-4 text-cyan-400" />
              Upload MRI Slice
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
              className="border-2 border-dashed border-white/15 hover:border-cyan-500/50 rounded-xl p-6 text-center cursor-pointer transition-all bg-slate-900/40 hover:bg-slate-900/80 group"
            >
              <div className="w-12 h-12 rounded-full bg-cyan-500/10 text-cyan-400 flex items-center justify-center mx-auto mb-3 group-hover:scale-110 transition-transform">
                <FileImage className="w-6 h-6" />
              </div>
              <p className="text-sm font-medium text-slate-200">
                Click to browse or drag MRI slice
              </p>
              <p className="text-xs text-slate-500 mt-1 font-mono">
                PNG, JPEG, TIF up to 25MB
              </p>
            </div>

            {selectedFile && (
              <div className="mt-3 flex items-center justify-between text-xs font-mono bg-slate-900 px-3 py-2 rounded border border-white/5 text-slate-300">
                <span className="truncate max-w-[200px]">{selectedFile.name}</span>
                <span className="text-slate-500">{(selectedFile.size / 1024).toFixed(0)} KB</span>
              </div>
            )}
          </div>

          {/* Model Controls & Hyperparameters */}
          <div className="med-card p-6 space-y-5">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              Inference Controls
            </h3>

            {/* Threshold Slider */}
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-slate-400">Probability Threshold:</span>
                <span className="text-cyan-400 font-bold">{threshold.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="w-full accent-cyan-400 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                <span>0.1 (High Recall)</span>
                <span>0.5 (Balanced)</span>
                <span>0.9 (High Precision)</span>
              </div>
            </div>

            {/* Multi-Task Learning Toggle */}
            <div className="pt-2 border-t border-white/5">
              <label className="flex items-center justify-between cursor-pointer">
                <div>
                  <span className="text-xs font-medium text-slate-200 block">Multi-Task Pathology Head</span>
                  <span className="text-[11px] text-slate-500 block">Classify tumor type simultaneously</span>
                </div>
                <input
                  type="checkbox"
                  checked={useMultitask}
                  onChange={(e) => setUseMultitask(e.target.checked)}
                  className="w-4 h-4 rounded text-cyan-500 focus:ring-0 cursor-pointer accent-cyan-500"
                />
              </label>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunSegmentation}
              disabled={loading || (!selectedFile && !previewUrl)}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium text-sm transition-all shadow-lg shadow-cyan-500/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Segmenting Lesion...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Run Segmentation
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

        {/* Right Column: Visualization & Quantitative Morphometrics (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          {/* Main Visual Display Card */}
          <div className="med-card p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
              <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-white/5">
                <button
                  onClick={() => setActiveView('overlay')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    activeView === 'overlay' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Lesion Overlay
                </button>
                <button
                  onClick={() => setActiveView('split')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    activeView === 'split' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Interactive Split
                </button>
                <button
                  onClick={() => setActiveView('mask')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    activeView === 'mask' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Binary Mask
                </button>
                <button
                  onClick={() => setActiveView('original')}
                  className={`px-3 py-1 rounded text-xs font-medium transition-all ${
                    activeView === 'original' ? 'bg-cyan-500/20 text-cyan-300 font-semibold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  Original T1 MRI
                </button>
              </div>

              {result && (
                <button
                  onClick={handleDownloadMask}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 border border-white/10 transition-all"
                >
                  <Download className="w-3.5 h-3.5 text-cyan-400" />
                  Export Mask
                </button>
              )}
            </div>

            {/* Visual Canvas Area */}
            <div className="relative aspect-square max-w-[500px] mx-auto rounded-xl overflow-hidden bg-black/60 border border-white/10 flex items-center justify-center">
              {previewUrl ? (
                activeView === 'split' && result ? (
                  <div className="relative w-full h-full select-none">
                    <img
                      src={result.original_base64}
                      alt="Original"
                      className="absolute inset-0 w-full h-full object-contain"
                    />
                    <div
                      className="absolute inset-0 overflow-hidden"
                      style={{ clipPath: `inset(0 0 0 ${sliderPos}%)` }}
                    >
                      <img
                        src={result.overlay_base64}
                        alt="Overlay"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    {/* Split line */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-cyan-400 shadow-lg shadow-cyan-400/50"
                      style={{ left: `${sliderPos}%` }}
                    >
                      <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-6 h-6 rounded-full bg-cyan-400 text-slate-950 flex items-center justify-center shadow-md">
                        <Sliders className="w-3.5 h-3.5" />
                      </div>
                    </div>
                    {/* Range input controller */}
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={sliderPos}
                      onChange={(e) => setSliderPos(parseFloat(e.target.value))}
                      className="absolute inset-0 opacity-0 cursor-ew-resize w-full h-full"
                    />
                  </div>
                ) : (
                  <img
                    src={
                      activeView === 'mask' && result
                        ? result.mask_base64
                        : activeView === 'overlay' && result
                        ? result.overlay_base64
                        : activeView === 'original' && result
                        ? result.original_base64
                        : previewUrl
                    }
                    alt="MRI Visualization"
                    className="w-full h-full object-contain"
                  />
                )
              ) : (
                <div className="text-center p-8 text-slate-500">
                  <Layers className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p className="text-sm font-medium">No MRI Image Selected</p>
                  <p className="text-xs mt-1">Upload a file or choose a test slice from the top</p>
                </div>
              )}
            </div>
          </div>

          {/* Morphometrics & Quantitative Analysis Results */}
          {result && (
            <div className="med-card p-6 space-y-6">
              <div className="flex items-center justify-between border-b border-white/5 pb-4">
                <h3 className="text-base font-semibold text-white flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400" />
                  Quantitative Tumor Morphometrics
                </h3>
                {result.tumor_detected ? (
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-rose-500/10 text-rose-400 border border-rose-500/30">
                    <AlertCircle className="w-3.5 h-3.5" /> Tumor Lesion Delineated
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                    <CheckCircle2 className="w-3.5 h-3.5" /> No Lesion Detected
                  </span>
                )}
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">TUMOR PIXELS</span>
                  <p className="text-lg font-bold font-mono text-white mt-1">
                    {result.tumor_pixel_count.toLocaleString()}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">CRANIAL COVERAGE</span>
                  <p className="text-lg font-bold font-mono text-cyan-400 mt-1">
                    {result.tumor_area_percentage}%
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">BOUNDING BOX</span>
                  <p className="text-xs font-mono text-slate-300 mt-1.5">
                    {result.bounding_box
                      ? `${result.bounding_box.width} × ${result.bounding_box.height} px`
                      : 'N/A'}
                  </p>
                </div>
                <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                  <span className="text-xs text-slate-500 font-mono">CENTROID (X, Y)</span>
                  <p className="text-xs font-mono text-slate-300 mt-1.5">
                    {result.centroid
                      ? `(${result.centroid.x}, ${result.centroid.y})`
                      : 'N/A'}
                  </p>
                </div>
              </div>

              {/* Multi-Task Classification Panel */}
              {result.classification && (
                <div className="p-4 rounded-xl bg-gradient-to-r from-slate-900 via-slate-900 to-indigo-950/30 border border-indigo-500/20">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-mono uppercase text-indigo-400 tracking-wider">
                      Multi-Task Pathology Prediction
                    </span>
                    <span className="text-xs font-mono text-slate-300">
                      Confidence: {(result.classification.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex items-baseline gap-2 mb-3">
                    <span className="text-2xl font-bold text-white">
                      {result.classification.predicted_class}
                    </span>
                  </div>

                  {/* Class Probabilities Bars */}
                  <div className="space-y-2 pt-2 border-t border-white/5">
                    {Object.entries(result.classification.all_probabilities).map(([cls, prob]) => (
                      <div key={cls} className="space-y-1">
                        <div className="flex justify-between text-[11px] font-mono text-slate-400">
                          <span>{cls}</span>
                          <span>{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              cls === result.classification?.predicted_class
                                ? 'bg-cyan-400'
                                : 'bg-slate-600'
                            }`}
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
