import {
  HealthStatus,
  ModelInfo,
  SegmentationResult,
  UncertaintyResult,
  RobustnessResult,
  HistoryItem,
  MetricsData,
  TestSample
} from '../types';

const API_BASE = `${import.meta.env.VITE_API_BASE_URL || ''}/api`;

export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Failed to fetch system health status.');
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/model/info`);
  if (!res.ok) throw new Error('Failed to fetch model info.');
  return res.json();
}

export async function fetchMetrics(): Promise<MetricsData> {
  const res = await fetch(`${API_BASE}/metrics`);
  if (!res.ok) throw new Error('Failed to fetch research metrics.');
  return res.json();
}

export async function fetchSamples(): Promise<TestSample[]> {
  const res = await fetch(`${API_BASE}/samples`);
  if (!res.ok) throw new Error('Failed to fetch test samples.');
  const data = await res.json();
  return data.samples || [];
}

export async function fetchHistory(): Promise<HistoryItem[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error('Failed to fetch analysis history.');
  return res.json();
}

export async function clearHistory(): Promise<void> {
  const res = await fetch(`${API_BASE}/history`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear history.');
}

export async function fileToBase64(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export async function runSegmentation(
  file: File | Blob,
  filename: string = 'mri_scan.png',
  threshold: number = 0.5,
  useMultitask: boolean = false
): Promise<SegmentationResult> {
  const base64Str = await fileToBase64(file);
  const res = await fetch(`${API_BASE}/segment`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: base64Str,
      threshold,
      use_multitask: useMultitask,
      filename
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Segmentation request failed' }));
    throw new Error(err.detail || 'Segmentation failed.');
  }
  return res.json();
}

export async function runUncertainty(
  file: File | Blob,
  filename: string = 'mri_scan.png',
  numPasses: number = 15,
  threshold: number = 0.5
): Promise<UncertaintyResult> {
  const base64Str = await fileToBase64(file);
  const res = await fetch(`${API_BASE}/uncertainty/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: base64Str,
      num_passes: numPasses,
      threshold,
      filename
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Uncertainty evaluation failed' }));
    throw new Error(err.detail || 'Uncertainty evaluation failed.');
  }
  return res.json();
}

export async function runRobustness(
  file: File | Blob,
  filename: string = 'mri_scan.png',
  degradationType: string = 'gaussian_noise',
  severity: number = 3
): Promise<RobustnessResult> {
  const base64Str = await fileToBase64(file);
  const res = await fetch(`${API_BASE}/robustness/evaluate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_base64: base64Str,
      degradation_type: degradationType,
      severity,
      filename
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Robustness evaluation failed' }));
    throw new Error(err.detail || 'Robustness evaluation failed.');
  }
  return res.json();
}

export function base64ToBlob(base64: string): Blob {
  const parts = base64.split(';base64,');
  const contentType = parts[0].replace('data:', '');
  const raw = window.atob(parts[1]);
  const rawLength = raw.length;
  const uInt8Array = new Uint8Array(rawLength);
  for (let i = 0; i < rawLength; ++i) {
    uInt8Array[i] = raw.charCodeAt(i);
  }
  return new Blob([uInt8Array], { type: contentType });
}
