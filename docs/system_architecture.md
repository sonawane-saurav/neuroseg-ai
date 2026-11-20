# System Architecture & Technical Specifications

**Architecture Style:** Decoupled Monorepo with Modular CV Engine, RESTful Backend, and React Frontend  
**Environment:** Cross-Platform (Windows, Linux, macOS), CPU and CUDA acceleration supported  

---

## 1. High-Level Architecture Overview

```
+---------------------------------------------------------------------------------+
|                                 REACT FRONTEND                                  |
|  - Vite + React 18 + TypeScript + Tailwind CSS                                  |
|  - Pages: Dashboard, Segmentation, Uncertainty, Robustness, Performance, History|
|  - Visualizations: Recharts + Interactive Split Slider + Plasma Heatmaps        |
+---------------------------------------+-----------------------------------------+
                                        | HTTP / JSON & Base64 Media
                                        v
+---------------------------------------------------------------------------------+
|                               FASTAPI BACKEND                                   |
|  - Uvicorn ASGI Server                                                          |
|  - Endpoints: /api/segment, /api/uncertainty, /api/robustness, /api/metrics     |
|  - Singleton Checkpoint In-Memory Caching (Zero Reload Overhead)                |
|  - SQLite Database for Audit & Inference History (WAL Mode)                     |
+---------------------------------------+-----------------------------------------+
                                        | Python In-Process Pipeline
                                        v
+---------------------------------------------------------------------------------+
|                        COMPUTER VISION & DEEP LEARNING                          |
|  - PyTorch 2.x U-Net (DoubleConv, Skip Connections, MC Dropout p=0.2)           |
|  - Multi-Task U-Net (Shared Encoder + Segmentation Decoder + Classification)    |
|  - Monte Carlo Dropout Uncertainty Engine (T Stochastic Forward Passes)         |
|  - Synthetic Degradation Engine (Gaussian Noise, Contrast, Blur, Downsampling)  |
|  - Synchronized Geometric & Intensity Preprocessing Pipelines                  |
+---------------------------------------------------------------------------------+
```

---

## 2. Directory Layout

```
brain-tumor-det-segmentation/
├── frontend/
│   ├── src/
│   │   ├── components/      # Navbar, MedicalDisclaimer, SplitSlider, etc.
│   │   ├── pages/           # 7 complete interactive application pages
│   │   ├── services/api.ts  # Typed API client communicating with backend
│   │   ├── types/index.ts   # Shared TypeScript interface definitions
│   │   └── App.tsx          # Root container and routing
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── backend/
│   ├── app/
│   │   ├── api/endpoints/   # Modular REST route handlers
│   │   ├── core/config.py   # Global configuration and constants
│   │   ├── database/db.py   # SQLite persistence engine
│   │   ├── schemas/         # Pydantic validation schemas
│   │   └── main.py          # FastAPI application entry point
│   └── requirements.txt
│
├── cv/
│   ├── data/                # Dataset loaders, splitters, inspection tools
│   ├── preprocessing/       # MRIPreprocessor and SynchronizedAugmentor
│   ├── models/              # U-Net, MultiTaskUNet, AttentionUNet
│   ├── training/            # Training loops, loss functions, metrics
│   ├── evaluation/          # Comparative benchmark and visualizers
│   ├── uncertainty/         # Monte Carlo Dropout engine
│   ├── robustness/          # 4 Degradation pipelines & stress benchmark
│   ├── inference/           # Unified predictor engine
│   ├── notebooks/           # GPU-accelerated Google Colab research notebook
│   ├── scripts/             # Orchestration scripts
│   └── tests/               # 13 Automated unit tests
│
├── artifacts/
│   ├── checkpoints/         # Trained best_unet.pth and best_multitask.pth
│   ├── metrics/             # JSON reports, history logs, benchmark stats
│   ├── predictions/         # Pre-exported sample slices for instant UI testing
│   └── visualizations/      # High-resolution research figures
│
├── docs/                    # Research methodology, inspection, and architecture docs
├── README.md
└── LICENSE
```

---

## 3. Data Flow & Security Principles

1. **Decoupled Lifecycle:** Training is executed completely independently of web server startup. Model weights are saved to `artifacts/checkpoints/` and loaded once into RAM when FastAPI launches.
2. **Deterministic Preprocessing:** Images submitted via the web API undergo the exact intensity normalization (`minmax` / `zscore`) and spatial resizing (256 × 256) used during model training.
3. **Nearest-Neighbor Masks:** All segmentation masks are strictly interpolated via nearest-neighbor algorithms to prevent label bleeding or non-binary artifacts.
4. **Epistemic Traceability:** Uncertainty estimates isolate model ambiguity, which is saved alongside segmentation metrics in SQLite.
