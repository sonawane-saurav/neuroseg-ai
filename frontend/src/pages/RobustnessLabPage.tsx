import React, { useState, useEffect, useRef } from 'react';
import {
  ShieldAlert,
  Play,
  Sliders,
  AlertTriangle,
  RefreshCw,
  TrendingDown,
  Layers,
  FileImage,
  ArrowRight
} from 'lucide-react';
import { RobustnessResult, TestSample } from '../types';
import { runRobustness, fetchSamples, base64ToBlob } from '../services/api';

export const RobustnessLabPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<TestSample[]>([]);
  const [selectedSampleId, setSelectedSampleId] = useState<number | null>(null);

  const [degradationType, setDegradationType] = useState<string>('gaussian_noise');
  const [severity, setSeverity] = useState<number>(3);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RobustnessResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const degradationOptions = [
    { id: 'gaussian_noise', label: 'Gaussian Noise', desc: 'Electronic / thermal sensor noise' },
    { id: 'low_contrast', label: 'Low Contrast', desc: 'Sub-optimal contrast uptake' },
    { id: 'motion_blur', label: 'Motion Blur', desc: 'Patient head movement artifacts' },
    { id: 'resolution_degradation', label: 'Resolution Drop', desc: 'Low-field scanner downsampling' },
  ];

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

  const handleRunRobustness = async () => {
    if (!selectedFile && !previewUrl) {
      setError('Please select an MRI slice to benchmark.');
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

      const res = await runRobustness(fileToSend, filename, degradationType, severity);
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Robustness test failed.');
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
            <ShieldAlert className="w-6 h-6 text-rose-400" />
            Robustness Laboratory & Stress Testing
          </h1>
          <p className="mt-1 text-sm text-slate-400">
            Investigate how clinical MRI image quality degradation degrades segmentation precision and lesion area estimation.
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
                    ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
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
              <FileImage className="w-4 h-4 text-rose-400" />
              Source MRI Slice
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
              className="border-2 border-dashed border-white/15 hover:border-rose-500/50 rounded-xl p-5 text-center cursor-pointer transition-all bg-slate-900/40 hover:bg-slate-900/80 group"
            >
              <FileImage className="w-8 h-8 text-rose-400 mx-auto mb-2 group-hover:scale-110 transition-transform" />
              <p className="text-xs font-medium text-slate-200">Upload slice or pick from test cohort</p>
            </div>
          </div>

          <div className="med-card p-6 space-y-5">
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-rose-400" />
              Degradation Hazard & Severity
            </h3>

            {/* Hazard Selector */}
            <div className="space-y-2">
              <label className="text-xs font-mono text-slate-400">Select Clinical Hazard:</label>
              <div className="grid grid-cols-1 gap-2">
                {degradationOptions.map((opt) => (
                  <button
                    key={opt.id}
                    onClick={() => setDegradationType(opt.id)}
                    className={`p-3 rounded-lg text-left border transition-all ${
                      degradationType === opt.id
                        ? 'bg-rose-500/10 border-rose-500/40 text-white'
                        : 'bg-slate-900/60 border-white/5 text-slate-400 hover:text-white'
                    }`}
                  >
                    <span className="text-xs font-semibold block">{opt.label}</span>
                    <span className="text-[11px] text-slate-500 block">{opt.desc}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Severity Slider */}
            <div>
              <div className="flex justify-between text-xs font-mono mb-2">
                <span className="text-slate-400">Severity Level:</span>
                <span className="text-rose-400 font-bold">Level {severity} / 5</span>
              </div>
              <input
                type="range"
                min="1"
                max="5"
                step="1"
                value={severity}
                onChange={(e) => setSeverity(parseInt(e.target.value))}
                className="w-full accent-rose-400 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-500 font-mono mt-1">
                <span>1 (Mild)</span>
                <span>3 (Moderate)</span>
                <span>5 (Extreme)</span>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={handleRunRobustness}
              disabled={loading || (!selectedFile && !previewUrl)}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-rose-500 to-rose-600 hover:from-rose-400 hover:to-rose-500 disabled:opacity-50 text-white font-medium text-sm transition-all shadow-lg shadow-rose-500/20 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Simulating Degradation...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  Run Robustness Test
                </>
              )}
            </button>

            {error && (
              <div className="p-3 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {/* Visual Comparison Grid (8 Cols) */}
        <div className="lg:col-span-8 space-y-6">
          {result ? (
            <div className="space-y-6">
              {/* Dual Visual Comparison: Clean vs Degraded */}
              <div className="med-card p-6">
                <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-rose-400" />
                  Side-by-Side Hazard Impact Comparison
                </h3>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  {/* Clean Column */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-slate-400">CLEAN MRI INPUT</span>
                      <span className="text-emerald-400">Baseline</span>
                    </div>
                    <div className="aspect-square rounded-xl overflow-hidden bg-black/60 border border-white/10">
                      <img
                        src={result.clean_overlay_base64}
                        alt="Clean Overlay"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <div className="p-3 rounded-lg bg-slate-900 border border-white/5 text-xs font-mono flex justify-between">
                      <span className="text-slate-400">Segmented Pixels:</span>
                      <span className="text-white font-bold">{result.clean_tumor_pixels.toLocaleString()}</span>
                    </div>
                  </div>

                  {/* Degraded Column */}
                  <div className="space-y-3">
                    <div className="flex items-center justify-between text-xs font-mono">
                      <span className="text-slate-400">DEGRADED (SEV {result.severity})</span>
                      <span className="text-rose-400 uppercase">{result.degradation_type.replace('_', ' ')}</span>
                    </div>
                    <div className="aspect-square rounded-xl overflow-hidden bg-black/60 border border-rose-500/30 shadow-lg shadow-rose-500/5">
                      <img
                        src={result.degraded_overlay_base64}
                        alt="Degraded Overlay"
                        className="w-full h-full object-contain"
                      />
                    </div>
                    <div className="p-3 rounded-lg bg-slate-900 border border-white/5 text-xs font-mono flex justify-between">
                      <span className="text-slate-400">Segmented Pixels:</span>
                      <span className="text-rose-400 font-bold">{result.degraded_tumor_pixels.toLocaleString()}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Stress Test Diagnostics Cards */}
              <div className="med-card p-6 space-y-4">
                <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                  <TrendingDown className="w-4 h-4 text-rose-400" />
                  Degradation Quantifications
                </h3>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                    <span className="text-xs text-slate-500 font-mono">PIXEL DEVIATION</span>
                    <p className={`text-lg font-bold font-mono mt-1 ${result.pixel_difference === 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                      {result.pixel_difference > 0 ? `+${result.pixel_difference}` : result.pixel_difference} px
                    </p>
                    <p className="text-[11px] text-slate-500 mt-0.5">Deviation from clean segmentation</p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                    <span className="text-xs text-slate-500 font-mono">HAZARD TYPE</span>
                    <p className="text-sm font-bold font-mono text-white mt-1 uppercase">
                      {result.degradation_type.replace('_', ' ')}
                    </p>
                    <p className="text-[11px] text-slate-500 mt-0.5">Severity Level {result.severity} of 5</p>
                  </div>

                  <div className="p-3 rounded-lg bg-slate-900 border border-white/5">
                    <span className="text-xs text-slate-500 font-mono">RELATIVE STABILITY</span>
                    <p className="text-lg font-bold font-mono text-cyan-400 mt-1">
                      {Math.max(0, 100 - Math.abs((result.pixel_difference / (result.clean_tumor_pixels + 1)) * 100)).toFixed(1)}%
                    </p>
                    <p className="text-[11px] text-slate-500 mt-0.5">Retained area agreement</p>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-slate-900/80 border border-white/5 text-xs text-slate-400 leading-relaxed">
                  <span className="text-rose-400 font-semibold block mb-1">Academic Finding:</span>
                  Notice that high-frequency noise and low-contrast compressions cause peripheral erosion of predicted tumor masks.
                  Models trained with synchronized photometric augmentations maintain significantly higher Dice retention than unaugmented networks.
                </div>
              </div>
            </div>
          ) : (
            <div className="med-card p-12 text-center text-slate-500">
              <ShieldAlert className="w-12 h-12 mx-auto mb-3 opacity-30" />
              <p className="text-base font-medium text-slate-300">Robustness Laboratory Ready</p>
              <p className="text-xs mt-1 max-w-md mx-auto">
                Select an MRI slice, configure the degradation hazard (noise, contrast, blur, resolution), and click "Run Robustness Test" to evaluate tolerance.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
