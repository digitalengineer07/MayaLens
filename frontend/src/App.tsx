import React, { useState, useEffect } from 'react';
import { 
  ShieldAlert, 
  ShieldCheck, 
  Upload, 
  Activity, 
  Layers, 
  Radio, 
  Eye, 
  Download, 
  RefreshCw, 
  ExternalLink,
  Database
} from 'lucide-react';

interface StreamResult {
  id: string;
  name: string;
  confidence: number;
  weight: string;
  focus: string;
  status: string;
}

interface Keyframe {
  frame_idx: number;
  timestamp: string;
  fake_score: number;
  anomaly_detected: string;
  xai_focus: string;
  crop_type: string;
  features: {
    fft_peaks: string;
    vit_attention: string;
    facial_landmarks: string;
  };
}

interface TemporalPoint {
  frame: number;
  time: string;
  score: number;
  is_anomaly: boolean;
}

interface AnalysisData {
  filename: string;
  media_type: string;
  decision: string;
  is_fake: boolean;
  confidence: number;
  risk_level: string;
  inference_time_ms: number;
  summary: string;
  streams: StreamResult[];
  keyframes: Keyframe[];
  temporal_series: TemporalPoint[];
  report_url: string;
}

interface BenchmarkData {
  evaluation: Array<Record<string, string>>;
  training: {
    total_epochs?: number;
    final_accuracy?: number;
    final_auc?: number;
    val_accuracy?: number;
    val_auc?: number;
    loss?: number;
  };
  model_architecture: {
    spatial_stream: string;
    context_stream: string;
    frequency_stream: string;
    fusion: string;
    xai_methods: string[];
  };
}

export default function App() {
  const [analyzing, setAnalyzing] = useState(false);
  const [data, setData] = useState<AnalysisData | null>(null);
  const [selectedKeyframe, setSelectedKeyframe] = useState<number>(0);
  const [activeXAIMethod, setActiveXAIMethod] = useState<'gradcam' | 'gradcam_plus' | 'lime' | 'saliency'>('gradcam');
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null);
  const [showBenchmarks, setShowBenchmarks] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  // Load sample on mount
  useEffect(() => {
    loadSample();
    fetchBenchmarks();
  }, []);

  const loadSample = async () => {
    setAnalyzing(true);
    try {
      const res = await fetch('/api/sample');
      if (!res.ok) throw new Error('API sample error');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);
    } catch (err) {
      console.error(err);
    } finally {
      setAnalyzing(false);
    }
  };

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch('/api/metrics');
      if (res.ok) {
        const json = await res.json();
        setBenchmarks(json);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadedFile(file);
    setAnalyzing(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Analysis failed');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);
    } catch (err) {
      console.error(err);
      alert('Error analyzing file. Backend may be offline.');
    } finally {
      setAnalyzing(false);
    }
  };

  const currentFrame = data?.keyframes[selectedKeyframe];

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Top Navbar */}
      <header className="border-b border-slate-800/80 bg-[#090e1a]/90 backdrop-blur sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-cyan-600 via-indigo-600 to-pink-500 p-0.5 shadow-lg shadow-cyan-500/20">
            <div className="h-full w-full bg-[#090e1a] rounded-[10px] flex items-center justify-center">
              <Radio className="h-5 w-5 text-cyan-400 animate-pulse" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 font-mono">
                MAYA<span className="text-slate-100">LENS</span>
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                XAI Forensic v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono">Multi-Stream Deepfake Detection & Explainability</p>
          </div>
        </div>

        <div className="flex items-center gap-3 text-sm">
          <button 
            onClick={() => setShowBenchmarks(!showBenchmarks)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-slate-700 bg-slate-800/50 hover:bg-slate-700/50 text-slate-300 transition"
          >
            <Database className="w-4 h-4 text-indigo-400" />
            <span>Model Benchmark</span>
          </button>

          <a 
            href="/api/report/demo"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-medium shadow-md shadow-cyan-600/20 transition"
          >
            <Download className="w-4 h-4" />
            <span>Forensic PDF Report</span>
          </a>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        
        {/* Top Control Bar: Upload & Fast Demo Actions */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Upload card */}
          <div className="md:col-span-2 relative group overflow-hidden border border-dashed border-slate-700 hover:border-cyan-500/60 rounded-2xl bg-[#0d1424]/60 p-6 flex flex-col justify-center items-center text-center transition cursor-pointer">
            <input 
              type="file" 
              accept="video/*,image/*" 
              onChange={handleFileUpload}
              className="absolute inset-0 opacity-0 cursor-pointer z-10" 
            />
            <div className="h-12 w-12 rounded-full bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center mb-3 group-hover:scale-110 transition">
              <Upload className="h-6 w-6 text-cyan-400" />
            </div>
            <h3 className="font-semibold text-slate-200">
              {uploadedFile ? uploadedFile.name : "Drop Video or Face Image to Inspect"}
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Supports MP4, AVI, MOV, JPG, PNG • Automatically aligned via MTCNN & 68 landmarks
            </p>
          </div>

          {/* Quick Demo & State card */}
          <div className="border border-slate-800 rounded-2xl bg-[#0d1424]/60 p-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between text-xs text-slate-400 mb-2 font-mono">
                <span>PIPELINE ENGINE</span>
                <span className="flex items-center gap-1.5 text-emerald-400">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
                  ACTIVE
                </span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                Processes dual crops (1.15× tight + 1.70× context) across spatial, ViT, and spectral FFT/DCT streams with multi-head BCE loss.
              </p>
            </div>
            <button 
              onClick={loadSample}
              disabled={analyzing}
              className="w-full mt-4 flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700 text-sm font-medium text-slate-200 transition"
            >
              <RefreshCw className={`w-4 h-4 text-cyan-400 ${analyzing ? 'animate-spin' : ''}`} />
              <span>{analyzing ? 'Processing Pipeline...' : 'Reload Demo Sample Video'}</span>
            </button>
          </div>
        </section>

        {/* Forensic Verdict Banner */}
        {data && (
          <section className={`rounded-2xl border p-6 relative overflow-hidden transition-all ${
            data.is_fake 
              ? 'bg-gradient-to-r from-red-950/40 via-red-900/20 to-slate-900/60 border-red-500/40 shadow-xl shadow-red-950/20' 
              : 'bg-gradient-to-r from-emerald-950/40 via-emerald-900/20 to-slate-900/60 border-emerald-500/40 shadow-xl shadow-emerald-950/20'
          }`}>
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
              <div className="flex items-start gap-4">
                <div className={`p-3.5 rounded-2xl ${
                  data.is_fake 
                    ? 'bg-red-500/20 border border-red-500/50 text-red-400' 
                    : 'bg-emerald-500/20 border border-emerald-500/50 text-emerald-400'
                }`}>
                  {data.is_fake ? <ShieldAlert className="w-8 h-8" /> : <ShieldCheck className="w-8 h-8" />}
                </div>

                <div>
                  <div className="flex items-center gap-3">
                    <h2 className={`text-2xl font-black tracking-wide font-mono ${
                      data.is_fake ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {data.decision}
                    </h2>
                    <span className={`text-xs px-2.5 py-0.5 rounded-md font-bold uppercase tracking-wider ${
                      data.is_fake ? 'bg-red-500/20 text-red-300 border border-red-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                    }`}>
                      {data.risk_level} RISK
                    </span>
                  </div>
                  <p className="text-sm text-slate-300 mt-1.5 max-w-2xl leading-relaxed">
                    {data.summary}
                  </p>
                </div>
              </div>

              {/* Confidence Score Gauge */}
              <div className="flex items-center gap-4 bg-slate-900/80 px-6 py-4 rounded-xl border border-slate-800">
                <div className="text-right">
                  <div className="text-3xl font-extrabold font-mono text-cyan-400">{data.confidence}%</div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider">Prediction Confidence</div>
                </div>
                <div className="h-10 w-px bg-slate-800" />
                <div>
                  <div className="text-xs font-mono text-slate-400">LATENCY</div>
                  <div className="text-base font-bold text-slate-200 font-mono">{data.inference_time_ms} ms</div>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Multi-Stream Tri-Core Architecture Cards */}
        {data && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold tracking-wider text-slate-400 uppercase font-mono flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                Multi-Stream Core Decomposition (Report §5)
              </h3>
              <span className="text-xs text-slate-500 font-mono">Weighted Feature Fusion Layer</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {data.streams.map((stream) => (
                <div 
                  key={stream.id}
                  className="rounded-xl border border-slate-800 bg-[#0c1220]/80 p-5 hover:border-slate-700 transition"
                >
                  <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-2">
                    <span className="font-semibold text-slate-200">{stream.name}</span>
                    <span className="bg-slate-800 px-2 py-0.5 rounded text-cyan-400 border border-slate-700">{stream.weight}</span>
                  </div>

                  <div className="flex items-end justify-between my-3">
                    <div className="text-2xl font-bold font-mono text-slate-100">
                      {stream.confidence}%
                    </div>
                    <span className={`text-xs px-2 py-0.5 rounded-full font-semibold ${
                      stream.confidence > 50 ? 'bg-red-500/10 text-red-400 border border-red-500/30' : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    }`}>
                      {stream.status}
                    </span>
                  </div>

                  <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mb-3">
                    <div 
                      className={`h-full rounded-full transition-all duration-700 ${
                        stream.confidence > 50 ? 'bg-gradient-to-r from-amber-500 to-red-500' : 'bg-gradient-to-r from-cyan-500 to-emerald-500'
                      }`}
                      style={{ width: `${stream.confidence}%` }}
                    />
                  </div>

                  <p className="text-xs text-slate-400 leading-normal">
                    {stream.focus}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Explainable AI (XAI) & Key-Frame Visualizer */}
        {data && currentFrame && (
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            
            {/* Visualizer Frame & Heatmaps (7 cols) */}
            <div className="lg:col-span-7 rounded-2xl border border-slate-800 bg-[#0c1220]/90 p-5 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 pb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4 text-cyan-400" />
                  <span className="font-semibold text-sm text-slate-200">XAI Visualizer (Key Frame #{currentFrame.frame_idx})</span>
                </div>

                {/* XAI Method Selector */}
                <div className="flex items-center bg-slate-900 rounded-lg p-1 border border-slate-800 text-xs">
                  <button 
                    onClick={() => setActiveXAIMethod('gradcam')}
                    className={`px-2.5 py-1 rounded-md transition font-mono ${
                      activeXAIMethod === 'gradcam' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Grad-CAM
                  </button>
                  <button 
                    onClick={() => setActiveXAIMethod('gradcam_plus')}
                    className={`px-2.5 py-1 rounded-md transition font-mono ${
                      activeXAIMethod === 'gradcam_plus' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Grad-CAM++
                  </button>
                  <button 
                    onClick={() => setActiveXAIMethod('lime')}
                    className={`px-2.5 py-1 rounded-md transition font-mono ${
                      activeXAIMethod === 'lime' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    LIME Superpixels
                  </button>
                  <button 
                    onClick={() => setActiveXAIMethod('saliency')}
                    className={`px-2.5 py-1 rounded-md transition font-mono ${
                      activeXAIMethod === 'saliency' ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/30' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Saliency
                  </button>
                </div>
              </div>

              {/* Simulated Face Crop with Interactive Heatmap Overlays */}
              <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center group">
                {/* SVG Forensic Canvas Representing the Aligned Face + Heatmap */}
                <svg className="w-full h-full" viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
                  {/* Background Grid */}
                  <defs>
                    <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                      <path d="M 20 0 L 0 0 0 20" fill="none" stroke="#1e293b" strokeWidth="0.5" />
                    </pattern>
                    <radialGradient id="heatGrad" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity="0.85" />
                      <stop offset="40%" stopColor="#f59e0b" stopOpacity="0.65" />
                      <stop offset="70%" stopColor="#06b6d4" stopOpacity="0.35" />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                    </radialGradient>
                    <radialGradient id="limeGrad" cx="50%" cy="50%" r="50%">
                      <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                      <stop offset="60%" stopColor="#059669" stopOpacity="0.4" />
                      <stop offset="100%" stopColor="#065f46" stopOpacity="0" />
                    </radialGradient>
                  </defs>
                  
                  <rect width="640" height="360" fill="#090d16" />
                  <rect width="640" height="360" fill="url(#grid)" />

                  {/* MTCNN Face Box (1.15x / 1.70x crop bounds) */}
                  <rect x="220" y="60" width="200" height="240" rx="16" fill="#131c31" stroke="#38bdf8" strokeWidth="1.5" strokeDasharray="4 2" />
                  <text x="230" y="80" fill="#38bdf8" fontSize="10" fontFamily="monospace">MTCNN BOUNDS (CROP {currentFrame.crop_type})</text>

                  {/* Stylized Face Silhouette */}
                  <ellipse cx="320" cy="180" rx="75" ry="100" fill="#1e293b" />
                  {/* Eyes */}
                  <ellipse cx="290" cy="155" rx="12" ry="7" fill="#475569" />
                  <ellipse cx="350" cy="155" rx="12" ry="7" fill="#475569" />
                  {/* Nose */}
                  <polygon points="320,165 315,190 325,190" fill="#334155" />
                  {/* Mouth */}
                  <ellipse cx="320" cy="225" rx="24" ry="10" fill="#475569" />

                  {/* Dynamic Heatmap Layer according to selected XAI */}
                  {activeXAIMethod === 'gradcam' && (
                    <g className="animate-pulse">
                      <circle cx="320" cy="235" r="45" fill="url(#heatGrad)" />
                      <circle cx="280" cy="220" r="30" fill="url(#heatGrad)" />
                      <text x="370" y="240" fill="#f87171" fontSize="11" fontFamily="monospace">GRAD-CAM [Mouth/Jaw Seam]</text>
                      <line x1="335" y1="235" x2="365" y2="235" stroke="#f87171" strokeWidth="1" />
                    </g>
                  )}

                  {activeXAIMethod === 'gradcam_plus' && (
                    <g>
                      <circle cx="290" cy="155" r="32" fill="url(#heatGrad)" />
                      <circle cx="350" cy="155" r="32" fill="url(#heatGrad)" />
                      <text x="390" y="155" fill="#fbbf24" fontSize="11" fontFamily="monospace">GRAD-CAM++ [Orbital Discontinuity]</text>
                      <line x1="365" y1="155" x2="385" y2="155" stroke="#fbbf24" strokeWidth="1" />
                    </g>
                  )}

                  {activeXAIMethod === 'lime' && (
                    <g>
                      {/* Superpixel polygons */}
                      <polygon points="260,200 300,195 310,240 270,245" fill="url(#limeGrad)" stroke="#34d399" strokeWidth="1.5" />
                      <polygon points="330,195 370,200 360,245 320,240" fill="url(#limeGrad)" stroke="#34d399" strokeWidth="1.5" />
                      <text x="375" y="215" fill="#34d399" fontSize="11" fontFamily="monospace">LIME [SLIC Superpixel Boundaries]</text>
                      <line x1="355" y1="215" x2="370" y2="215" stroke="#34d399" strokeWidth="1" />
                    </g>
                  )}

                  {activeXAIMethod === 'saliency' && (
                    <g>
                      <circle cx="320" cy="180" r="85" fill="none" stroke="#a855f7" strokeWidth="2" strokeDasharray="3 3" />
                      <path d="M 280 155 Q 320 180 360 155" fill="none" stroke="#e879f9" strokeWidth="2" />
                      <path d="M 290 225 Q 320 245 350 225" fill="none" stroke="#e879f9" strokeWidth="2" />
                      <text x="410" y="180" fill="#e879f9" fontSize="11" fontFamily="monospace">SALIENCY [Input Gradients]</text>
                      <line x1="385" y1="180" x2="405" y2="180" stroke="#e879f9" strokeWidth="1" />
                    </g>
                  )}

                  {/* 68 Landmark overlay points */}
                  <g fill="#38bdf8" opacity="0.6">
                    <circle cx="280" cy="150" r="2" /><circle cx="290" cy="148" r="2" /><circle cx="300" cy="150" r="2" />
                    <circle cx="340" cy="150" r="2" /><circle cx="350" cy="148" r="2" /><circle cx="360" cy="150" r="2" />
                    <circle cx="320" cy="170" r="2" /><circle cx="320" cy="185" r="2" />
                    <circle cx="305" cy="225" r="2" /><circle cx="320" cy="223" r="2" /><circle cx="335" cy="225" r="2" />
                  </g>
                </svg>

                <div className="absolute bottom-3 left-3 bg-slate-900/90 backdrop-blur border border-slate-800 text-xs px-3 py-1.5 rounded-lg font-mono text-slate-300">
                  {currentFrame.xai_focus}
                </div>
              </div>

              {/* Keyframe selector chips */}
              <div className="flex items-center gap-2 pt-1">
                <span className="text-xs text-slate-400 font-mono">SELECT KEYFRAME:</span>
                {data.keyframes.map((kf, idx) => (
                  <button
                    key={kf.frame_idx}
                    onClick={() => setSelectedKeyframe(idx)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition flex items-center gap-2 ${
                      selectedKeyframe === idx 
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold' 
                        : 'bg-slate-800/60 text-slate-400 hover:bg-slate-800 border border-slate-800'
                    }`}
                  >
                    <span>Frame #{kf.frame_idx}</span>
                    <span className="text-[10px] opacity-75">{kf.timestamp}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Keyframe Forensic Breakdown (5 cols) */}
            <div className="lg:col-span-5 rounded-2xl border border-slate-800 bg-[#0c1220]/90 p-5 flex flex-col justify-between space-y-4">
              <div>
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3 mb-4">
                  <span className="text-xs font-mono uppercase text-slate-400">Forensic Evidence Profile</span>
                  <span className="text-xs font-mono font-bold text-red-400">
                    Anomaly Score: {(currentFrame.fake_score * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="space-y-4">
                  <div className="bg-slate-900/60 p-3.5 rounded-xl border border-slate-800/80">
                    <span className="text-xs text-cyan-400 font-mono font-semibold block mb-1">
                      PRIMARY DEFECT IDENTIFIED
                    </span>
                    <p className="text-sm text-slate-200">
                      {currentFrame.anomaly_detected}
                    </p>
                  </div>

                  <div className="space-y-2.5 text-xs font-mono">
                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
                      <span className="text-slate-400 block mb-0.5 font-bold">FFT / DCT FREQUENCY:</span>
                      <span className="text-slate-200">{currentFrame.features.fft_peaks}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
                      <span className="text-slate-400 block mb-0.5 font-bold">VISION TRANSFORMER ATTENTION:</span>
                      <span className="text-slate-200">{currentFrame.features.vit_attention}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/40 border border-slate-800">
                      <span className="text-slate-400 block mb-0.5 font-bold">LANDMARK ALIGNMENT:</span>
                      <span className="text-slate-200">{currentFrame.features.facial_landmarks}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-slate-800/80 flex items-center justify-between">
                <span className="text-xs text-slate-400 font-mono">Export format: ReportLab PDF</span>
                <a 
                  href="/api/report/demo"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 text-xs text-cyan-400 hover:text-cyan-300 font-mono font-semibold transition"
                >
                  <span>Download Report Section</span>
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              </div>
            </div>

          </section>
        )}

        {/* Temporal Fluctuation Chart (Video Frame Analysis) */}
        {data && data.temporal_series && data.temporal_series.length > 1 && (
          <section className="rounded-2xl border border-slate-800 bg-[#0c1220]/80 p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                <h3 className="text-sm font-semibold text-slate-200">Temporal Manipulation Probability Curve</h3>
              </div>
              <span className="text-xs text-slate-500 font-mono">Threshold ε = 0.50</span>
            </div>

            {/* Custom Bar chart timeline */}
            <div className="pt-4 pb-2">
              <div className="h-28 flex items-end gap-1.5 sm:gap-2">
                {data.temporal_series.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1 group relative">
                    {/* Tooltip */}
                    <div className="absolute -top-10 opacity-0 group-hover:opacity-100 transition pointer-events-none bg-slate-900 border border-slate-700 text-[10px] font-mono px-2 py-1 rounded text-slate-200 whitespace-nowrap z-20 shadow-lg">
                      Frame {pt.frame} ({pt.time}): {pt.score}%
                    </div>
                    {/* Bar */}
                    <div 
                      className={`w-full rounded-t transition-all duration-500 ${
                        pt.score > 50 
                          ? 'bg-gradient-to-t from-red-600/60 to-red-500 group-hover:brightness-125' 
                          : 'bg-gradient-to-t from-cyan-600/60 to-cyan-500 group-hover:brightness-125'
                      }`}
                      style={{ height: `${Math.max(8, pt.score)}%` }}
                    />
                    <span className="text-[9px] font-mono text-slate-500 truncate w-full text-center">
                      {pt.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Benchmark & Training Metrics Drawer / Modal */}
        {showBenchmarks && benchmarks && (
          <section className="rounded-2xl border border-indigo-500/40 bg-[#0c1022] p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Database className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-slate-100">Official Model Benchmark & Evaluation (From Repo)</h3>
              </div>
              <button 
                onClick={() => setShowBenchmarks(false)}
                className="text-xs text-slate-400 hover:text-slate-200 px-2 py-1 bg-slate-800 rounded"
              >
                Close
              </button>
            </div>

            {/* Metrics summary cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">TRAINING ACCURACY</div>
                <div className="text-2xl font-bold font-mono text-emerald-400">{benchmarks.training.final_accuracy}%</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">TRAINING AUC-ROC</div>
                <div className="text-2xl font-bold font-mono text-cyan-400">{benchmarks.training.final_auc}%</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">VALIDATION AUC</div>
                <div className="text-2xl font-bold font-mono text-indigo-400">{benchmarks.training.val_auc}%</div>
              </div>
              <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800">
                <div className="text-xs text-slate-400 font-mono">TOTAL LOGGED EPOCHS</div>
                <div className="text-2xl font-bold font-mono text-slate-200">{benchmarks.training.total_epochs}</div>
              </div>
            </div>

            {/* Benchmark Table */}
            <div className="overflow-x-auto rounded-xl border border-slate-800">
              <table className="w-full text-left text-xs font-mono">
                <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="p-2.5">Dataset & Split</th>
                    <th className="p-2.5">Level</th>
                    <th className="p-2.5">Classification Head</th>
                    <th className="p-2.5">AUC-ROC</th>
                    <th className="p-2.5">LogLoss</th>
                    <th className="p-2.5">EER</th>
                    <th className="p-2.5">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {benchmarks.evaluation.slice(0, 8).map((row, idx) => (
                    <tr key={idx} className="hover:bg-slate-900/40">
                      <td className="p-2.5 font-semibold text-slate-200">{row.set}</td>
                      <td className="p-2.5">{row.level}</td>
                      <td className="p-2.5 text-cyan-400">{row.head}</td>
                      <td className="p-2.5 font-bold text-emerald-400">{row.AUC}</td>
                      <td className="p-2.5">{row.LogLoss}</td>
                      <td className="p-2.5 text-amber-400">{row.EER}</td>
                      <td className="p-2.5">{row.ACC}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 py-4 px-6 text-center text-xs text-slate-500 font-mono">
        MayaLens Deepfake Detection • Developed with React + Vite + Tailwind CSS + FastAPI
      </footer>
    </div>
  );
}
