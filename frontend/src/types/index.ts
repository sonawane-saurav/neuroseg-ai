export interface HealthStatus {
  status: string;
  project_name: string;
  device: string;
  torch_version: string;
  cuda_available: boolean;
  unet_available: boolean;
  multitask_available: boolean;
}

export interface ModelInfo {
  model_name: string;
  input_resolution: [number, number];
  input_channels: number;
  output_classes: number;
  total_parameters: number;
  trainable_parameters: number;
  checkpoint_available: boolean;
  checkpoint_path?: string;
  features: number;
  architecture: string;
}

export interface BoundingBox {
  ymin: number;
  ymax: number;
  xmin: number;
  xmax: number;
  width: number;
  height: number;
}

export interface Centroid {
  x: number;
  y: number;
}

export interface ClassificationResult {
  predicted_class: string;
  confidence: number;
  all_probabilities: Record<string, number>;
}

export interface SegmentationResult {
  tumor_detected: boolean;
  tumor_pixel_count: number;
  tumor_area_percentage: number;
  bounding_box: BoundingBox | null;
  centroid: Centroid | null;
  classification?: ClassificationResult | null;
  mask_base64: string;
  overlay_base64: string;
  original_base64: string;
  resolution: [number, number];
  analysis_id?: number;
}

export interface UncertaintyResult {
  num_passes: number;
  mean_uncertainty: number;
  max_uncertainty: number;
  mean_tumor_uncertainty: number;
  mean_boundary_uncertainty: number;
  high_uncertainty_pixels: number;
  tumor_pixels: number;
  mask_base64: string;
  heatmap_base64: string;
  uncertainty_overlay_base64: string;
  original_base64: string;
  analysis_id?: number;
}

export interface RobustnessMetricsDiff {
  clean_dice: number;
  degraded_dice: number;
  absolute_dice_drop: number;
  relative_dice_drop_pct: number;
  clean_iou: number;
  degraded_iou: number;
}

export interface RobustnessResult {
  degradation_type: string;
  severity: number;
  original_base64: string;
  degraded_image_base64: string;
  clean_mask_base64: string;
  degraded_mask_base64: string;
  clean_overlay_base64: string;
  degraded_overlay_base64: string;
  clean_tumor_pixels: number;
  degraded_tumor_pixels: number;
  pixel_difference: number;
  metrics_comparison?: RobustnessMetricsDiff;
  analysis_id?: number;
}

export interface TestSample {
  id: number;
  name: string;
  image_base64: string;
  gt_mask_base64?: string;
}

export interface HistoryItem {
  id: number;
  filename: string;
  timestamp: string;
  model_used: string;
  tumor_detected: boolean;
  tumor_pixel_count: number;
  tumor_area_percentage: number;
  classification_label?: string;
  confidence?: number;
  has_uncertainty: boolean;
  mean_uncertainty: number;
  degradation_type: string;
  severity: number;
  details?: Record<string, any>;
}

export interface MetricsData {
  dataset_inspection?: any;
  unet_training?: {
    summary?: any;
    history?: {
      train_loss: number[];
      val_loss: number[];
      train_dice: number[];
      val_dice: number[];
      train_iou: number[];
      val_iou: number[];
    };
  };
  multitask_training?: {
    summary?: any;
    history?: {
      train_loss: number[];
      val_loss: number[];
      train_dice: number[];
      val_dice: number[];
      train_cls_acc: number[];
      val_cls_acc: number[];
    };
  };
  model_comparison?: any;
  robustness_benchmark?: any;
}
