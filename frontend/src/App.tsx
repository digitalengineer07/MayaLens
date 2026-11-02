import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Download, 
  RefreshCw, 
  ShieldAlert, 
  ShieldCheck, 
  Layers, 
  Activity, 
  BarChart3,
  ChevronRight,
  Eye
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
  const [data, setData] = useState<AnalysisData | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [selectedKeyframe, setSelectedKeyframe] = useState<number>(0);
  const [activeFilter, setActiveFilter] = useState<'original' | 'gradcam' | 'gradcam_plus' | 'lime' | 'saliency'>('gradcam');
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null);
  const [showBenchmarks, setShowBenchmarks] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  useEffect(() => {
    loadSample();
    fetchBenchmarks();
  }, []);

  const loadSample = async () => {
    setAnalyzing(true);
    try {
      const res = await fetch('/api/sample');
      if (!res.ok) throw new Error('API Error');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);
    } catch (e) {
      console.error(e);
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
    } catch (e) {
      console.error(e);
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
      if (!res.ok) throw new Error('Upload analysis failed');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);
    } catch (err) {
      console.error(err);
      alert('Analysis error. Backend may be offline.');
    } finally {
      setAnalyzing(false);
    }
  };

  const currentFrame = data?.keyframes[selectedKeyframe];

  return (
    <div className="min-h-screen bg-[#09090b] text-[#f4f4f5] flex flex-col font-sans selection:bg-zinc-800 selection:text-white">
      
      {/* Clean, Minimal Header */}
      <header className="border-b border-zinc-800/80 bg-[#09090b]/80 backdrop-blur-md sticky top-0 z-40 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-zinc-100 text-zinc-950 flex items-center justify-center font-bold text-sm shadow-sm">
              ML
            </div>
            <div className="flex items-baseline gap-2">
              <span className="font-semibold text-base text-zinc-100 tracking-tight">MayaLens</span>
              <span className="text-xs text-zinc-500 hidden sm:inline">Deepfake Detection & Explainability</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowBenchmarks(!showBenchmarks)}
              className="text-xs text-zinc-400 hover:text-zinc-200 px-3 py-1.5 rounded-lg border border-zinc-800 hover:border-zinc-700 bg-zinc-900/60 transition flex items-center gap-1.5"
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Model Evaluation</span>
            </button>

            <a
              href="/api/report/demo"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-medium text-zinc-900 bg-zinc-100 hover:bg-white px-3.5 py-1.5 rounded-lg transition shadow-sm flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download PDF Report</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8 space-y-8">
        
        {/* Minimal Hero & Upload */}
        <section className="space-y-4">
          <div className="space-y-1.5">
            <h1 className="text-2xl sm:text-3xl font-semibold text-zinc-100 tracking-tight">
              Transparent media verification.
            </h1>
            <p className="text-sm text-zinc-400 max-w-xl leading-relaxed">
              Upload a video or photo to identify neural face-swaps and manipulation artifacts, verified across spatial textures, face lighting, and frequency spectrums.
            </p>
          </div>

          {/* Clean Dropzone Container */}
          <div className="rounded-2xl border border-zinc-800/90 bg-zinc-900/40 p-5 sm:p-6 transition hover:border-zinc-700/80">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
              
              <div className="relative flex-1 group">
                <input 
                  type="file" 
                  accept="video/*,image/*" 
                  onChange={handleFileUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer z-10" 
                />
                <div className="flex items-center gap-3.5 cursor-pointer">
                  <div className="w-10 h-10 rounded-xl bg-zinc-800 border border-zinc-700/80 flex items-center justify-center text-zinc-300 group-hover:text-white group-hover:border-zinc-600 transition shrink-0">
                    <Upload className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-zinc-200">
                      {uploadedFile ? uploadedFile.name : "Drop a video or image to inspect"}
                    </div>
                    <div className="text-xs text-zinc-500">
                      MP4, AVI, MOV, JPG, PNG • Evaluated using multi-scale MTCNN alignment
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-zinc-800">
                <button
                  onClick={loadSample}
                  disabled={analyzing}
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 text-xs font-medium text-zinc-300 hover:text-white px-3.5 py-2 rounded-xl bg-zinc-800 hover:bg-zinc-700/80 border border-zinc-700/60 transition disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${analyzing ? 'animate-spin' : ''}`} />
                  <span>{analyzing ? 'Evaluating...' : 'Try Demo Video'}</span>
                </button>
              </div>

            </div>
          </div>
        </section>

        {/* Primary Verdict Card */}
        {data && (
          <section className={`rounded-2xl border p-6 transition ${
            data.is_fake 
              ? 'bg-rose-950/20 border-rose-900/50' 
              : 'bg-emerald-950/20 border-emerald-900/50'
          }`}>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
              
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-xl mt-0.5 shrink-0 ${
                  data.is_fake 
                    ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' 
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                }`}>
                  {data.is_fake ? <ShieldAlert className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
                </div>

                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <h2 className="text-xl font-semibold text-zinc-100">
                      {data.is_fake ? "Deepfake Detected" : "Authentic Media"}
                    </h2>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                      data.is_fake 
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' 
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    }`}>
                      {data.confidence}% confidence
                    </span>
                    <span className="text-xs text-zinc-500">
                      • {data.inference_time_ms} ms
                    </span>
                  </div>
                  <p className="text-sm text-zinc-400 max-w-2xl leading-relaxed">
                    {data.summary}
                  </p>
                </div>
              </div>

              <div className="sm:text-right shrink-0">
                <div className="text-xs text-zinc-500 uppercase tracking-wider font-mono">FILE SOURCE</div>
                <div className="text-sm font-medium text-zinc-300 truncate max-w-[200px]">{data.filename}</div>
              </div>

            </div>
          </section>
        )}

        {/* Visual Inspection Loupe & Evidence Notes */}
        {data && currentFrame && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
                <Eye className="w-4 h-4 text-zinc-400" />
                <span>Visual Evidence Inspection</span>
              </h3>
              <div className="text-xs text-zinc-500">
                Keyframe #{currentFrame.frame_idx} ({currentFrame.timestamp})
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start">
              
              {/* Left: Beautiful Frame Canvas (7 cols) */}
              <div className="md:col-span-7 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-4 space-y-3.5 shadow-sm">
                
                {/* Method Filter Toggle Bar */}
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <span className="text-xs text-zinc-400 font-medium">Heatmap Layer:</span>
                  <div className="flex items-center bg-zinc-800/80 p-0.5 rounded-lg border border-zinc-700/60 text-xs">
                    {(['original', 'gradcam', 'gradcam_plus', 'lime', 'saliency'] as const).map((filter) => {
                      const labels: Record<string, string> = {
                        original: 'Clean',
                        gradcam: 'Grad-CAM',
                        gradcam_plus: 'Grad-CAM++',
                        lime: 'LIME',
                        saliency: 'Saliency'
                      };
                      return (
                        <button
                          key={filter}
                          onClick={() => setActiveFilter(filter)}
                          className={`px-2.5 py-1 rounded-md text-xs font-medium transition ${
                            activeFilter === filter
                              ? 'bg-zinc-100 text-zinc-900 shadow-xs'
                              : 'text-zinc-400 hover:text-zinc-200'
                          }`}
                        >
                          {labels[filter]}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Face Canvas */}
                <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-zinc-950 border border-zinc-800 flex items-center justify-center">
                  <svg className="w-full h-full max-h-[340px]" viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <radialGradient id="softHeat" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8" />
                        <stop offset="50%" stopColor="#fb923c" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="#fb923c" stopOpacity="0" />
                      </radialGradient>
                      <radialGradient id="limeSoft" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#10b981" stopOpacity="0.85" />
                        <stop offset="60%" stopColor="#059669" stopOpacity="0.35" />
                        <stop offset="100%" stopColor="#059669" stopOpacity="0" />
                      </radialGradient>
                    </defs>

                    {/* Clean background */}
                    <rect width="640" height="360" fill="#0c0e12" />

                    {/* Crop boundary frame */}
                    <rect x="230" y="65" width="180" height="230" rx="8" fill="#14171f" stroke="#272f3d" strokeWidth="1" strokeDasharray="3 3" />

                    {/* Face geometry */}
                    <ellipse cx="320" cy="180" rx="65" ry="90" fill="#1c222c" />
                    {/* Eyes */}
                    <ellipse cx="295" cy="155" rx="10" ry="6" fill="#364052" />
                    <ellipse cx="345" cy="155" rx="10" ry="6" fill="#364052" />
                    {/* Nose */}
                    <path d="M 320 162 L 316 186 L 324 186" fill="none" stroke="#4a5568" strokeWidth="1.5" />
                    {/* Mouth */}
                    <ellipse cx="320" cy="225" rx="22" ry="8" fill="#364052" />

                    {/* Filter overlays */}
                    {activeFilter === 'gradcam' && (
                      <g>
                        <circle cx="320" cy="230" r="42" fill="url(#softHeat)" />
                        <circle cx="285" cy="215" r="28" fill="url(#softHeat)" />
                        <text x="375" y="235" fill="#f43f5e" fontSize="11" fontFamily="sans-serif" fontWeight="500">Mouth & Jaw Blending</text>
                        <line x1="335" y1="230" x2="370" y2="230" stroke="#f43f5e" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'gradcam_plus' && (
                      <g>
                        <circle cx="295" cy="155" r="28" fill="url(#softHeat)" />
                        <circle cx="345" cy="155" r="28" fill="url(#softHeat)" />
                        <text x="385" y="155" fill="#fb923c" fontSize="11" fontFamily="sans-serif" fontWeight="500">Periocular Texture Discontinuity</text>
                        <line x1="360" y1="155" x2="380" y2="155" stroke="#fb923c" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'lime' && (
                      <g>
                        <polygon points="265,195 305,190 315,235 275,240" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1" />
                        <polygon points="325,190 365,195 355,240 315,235" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1" />
                        <text x="375" y="215" fill="#10b981" fontSize="11" fontFamily="sans-serif" fontWeight="500">Superpixel Boundaries</text>
                        <line x1="355" y1="215" x2="370" y2="215" stroke="#10b981" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'saliency' && (
                      <g>
                        <circle cx="320" cy="180" r="75" fill="none" stroke="#818cf8" strokeWidth="1.5" strokeDasharray="3 3" />
                        <path d="M 285 155 Q 320 170 355 155" fill="none" stroke="#818cf8" strokeWidth="1.5" />
                        <path d="M 295 225 Q 320 240 345 225" fill="none" stroke="#818cf8" strokeWidth="1.5" />
                        <text x="390" y="180" fill="#818cf8" fontSize="11" fontFamily="sans-serif" fontWeight="500">High-gradient contours</text>
                        <line x1="375" y1="180" x2="385" y2="180" stroke="#818cf8" strokeWidth="1" />
                      </g>
                    )}

                    {/* Landmarks points */}
                    <g fill="#94a3b8" opacity="0.5">
                      <circle cx="285" cy="150" r="1.5" /><circle cx="295" cy="148" r="1.5" /><circle cx="305" cy="150" r="1.5" />
                      <circle cx="335" cy="150" r="1.5" /><circle cx="345" cy="148" r="1.5" /><circle cx="355" cy="150" r="1.5" />
                      <circle cx="320" cy="170" r="1.5" /><circle cx="320" cy="185" r="1.5" />
                      <circle cx="305" cy="225" r="1.5" /><circle cx="320" cy="228" r="1.5" /><circle cx="335" cy="225" r="1.5" />
                    </g>
                  </svg>

                  <div className="absolute bottom-2.5 left-2.5 bg-zinc-900/90 backdrop-blur-md px-2.5 py-1 rounded-md text-[11px] text-zinc-400 border border-zinc-800">
                    {currentFrame.xai_focus}
                  </div>
                </div>

                {/* Keyframe selector chips */}
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-xs text-zinc-500">Keyframes:</span>
                  {data.keyframes.map((kf, idx) => (
                    <button
                      key={kf.frame_idx}
                      onClick={() => setSelectedKeyframe(idx)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                        selectedKeyframe === idx
                          ? 'bg-zinc-100 text-zinc-950 font-semibold shadow-xs'
                          : 'bg-zinc-800/80 text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      Frame #{kf.frame_idx} ({kf.timestamp})
                    </button>
                  ))}
                </div>

              </div>

              {/* Right: Forensic Notes & Findings (5 cols) */}
              <div className="md:col-span-5 rounded-2xl border border-zinc-800 bg-zinc-900/50 p-5 space-y-4 shadow-sm">
                <div>
                  <h4 className="font-semibold text-sm text-zinc-100">Findings & Evidence</h4>
                  <p className="text-xs text-zinc-500">Observable anomalies detected by the model streams on this frame.</p>
                </div>

                <div className="space-y-3">
                  <div className="p-3.5 rounded-xl bg-zinc-950/60 border border-zinc-800/80">
                    <span className="text-xs font-medium text-rose-400 block mb-1">Primary Observation</span>
                    <p className="text-xs text-zinc-300 leading-relaxed">
                      {currentFrame.anomaly_detected}
                    </p>
                  </div>

                  <div className="space-y-2.5 text-xs">
                    <div className="p-2.5 rounded-lg bg-zinc-950/40 border border-zinc-800/60">
                      <span className="text-zinc-400 font-medium block mb-0.5">Frequency Domain (FFT / DCT):</span>
                      <span className="text-zinc-300">{currentFrame.features.fft_peaks}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-zinc-950/40 border border-zinc-800/60">
                      <span className="text-zinc-400 font-medium block mb-0.5">Vision Transformer Attention:</span>
                      <span className="text-zinc-300">{currentFrame.features.vit_attention}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-zinc-950/40 border border-zinc-800/60">
                      <span className="text-zinc-400 font-medium block mb-0.5">Facial Landmarks Alignment:</span>
                      <span className="text-zinc-300">{currentFrame.features.facial_landmarks}</span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-zinc-800/80">
                  <a
                    href="/api/report/demo"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-zinc-400 hover:text-white transition"
                  >
                    <span>View corresponding page in PDF report</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>

            </div>
          </section>
        )}

        {/* The Three Core Detection Streams */}
        {data && (
          <section className="space-y-3">
            <h3 className="text-sm font-semibold text-zinc-200 flex items-center gap-2">
              <Layers className="w-4 h-4 text-zinc-400" />
              <span>Multi-Stream Model Breakdown</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {data.streams.map((stream) => (
                <div key={stream.id} className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5 space-y-3 shadow-sm hover:border-zinc-700/80 transition">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-sm text-zinc-200">{stream.name}</h4>
                    <span className="text-xs text-zinc-500 font-mono">{stream.weight}</span>
                  </div>

                  <div className="flex items-baseline justify-between pt-1">
                    <div className="text-2xl font-semibold text-zinc-100">
                      {stream.confidence}%
                    </div>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                      stream.confidence > 50 
                        ? 'bg-rose-500/10 text-rose-400 border border-rose-500/20' 
                        : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                    }`}>
                      {stream.status}
                    </span>
                  </div>

                  {/* Clean Minimal Progress Bar */}
                  <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        stream.confidence > 50 ? 'bg-rose-500' : 'bg-emerald-500'
                      }`}
                      style={{ width: `${stream.confidence}%` }}
                    />
                  </div>

                  <p className="text-xs text-zinc-400 leading-relaxed">
                    {stream.focus}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Video Temporal Timeline */}
        {data && data.temporal_series && data.temporal_series.length > 1 && (
          <section className="rounded-2xl border border-zinc-800 bg-zinc-900/40 p-5 space-y-4 shadow-sm">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-zinc-400" />
                <h3 className="text-sm font-semibold text-zinc-200">Temporal Manipulation Over Time</h3>
              </div>
              <span className="text-xs text-zinc-500">Threshold: 50%</span>
            </div>

            <div className="pt-4 pb-2">
              <div className="h-24 flex items-end gap-2 sm:gap-3">
                {data.temporal_series.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group relative">
                    <div className="absolute -top-7 opacity-0 group-hover:opacity-100 transition pointer-events-none bg-zinc-800 text-zinc-200 text-[10px] px-2 py-0.5 rounded whitespace-nowrap z-20 shadow-md">
                      {pt.time}: {pt.score}%
                    </div>
                    <div 
                      className={`w-full rounded-xs transition-all duration-300 ${
                        pt.score > 50 
                          ? 'bg-rose-500/70 group-hover:bg-rose-500' 
                          : 'bg-zinc-700/70 group-hover:bg-zinc-600'
                      }`}
                      style={{ height: `${Math.max(10, pt.score)}%` }}
                    />
                    <span className="text-[10px] text-zinc-500 truncate w-full text-center">
                      {pt.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Model Evaluation Modal / Drawer */}
        {showBenchmarks && benchmarks && (
          <section className="rounded-2xl border border-zinc-700 bg-zinc-900 p-6 space-y-4 shadow-xl">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div>
                <h3 className="font-semibold text-zinc-100 text-base">Model Benchmark Data</h3>
                <p className="text-xs text-zinc-400">Recorded results from evaluation.csv in the project repository.</p>
              </div>
              <button 
                onClick={() => setShowBenchmarks(false)}
                className="text-xs text-zinc-400 hover:text-white px-2.5 py-1 bg-zinc-800 rounded-md border border-zinc-700 transition"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 bg-zinc-950/60 rounded-xl border border-zinc-800">
                <div className="text-xs text-zinc-500">In-Domain Video AUC</div>
                <div className="text-xl font-bold text-zinc-100 mt-0.5">0.9899</div>
              </div>
              <div className="p-3 bg-zinc-950/60 rounded-xl border border-zinc-800">
                <div className="text-xs text-zinc-500">Equal Error Rate (EER)</div>
                <div className="text-xl font-bold text-zinc-100 mt-0.5">4.34%</div>
              </div>
              <div className="p-3 bg-zinc-950/60 rounded-xl border border-zinc-800">
                <div className="text-xs text-zinc-500">Final Accuracy</div>
                <div className="text-xl font-bold text-emerald-400 mt-0.5">{benchmarks.training.final_accuracy}%</div>
              </div>
              <div className="p-3 bg-zinc-950/60 rounded-xl border border-zinc-800">
                <div className="text-xs text-zinc-500">Loss</div>
                <div className="text-xl font-bold text-zinc-100 mt-0.5">{benchmarks.training.loss}</div>
              </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-zinc-800">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-zinc-950 text-zinc-400 border-b border-zinc-800">
                  <tr>
                    <th className="p-2.5">Dataset & Split</th>
                    <th className="p-2.5">Level</th>
                    <th className="p-2.5">Model Head</th>
                    <th className="p-2.5">AUC-ROC</th>
                    <th className="p-2.5">Log-Loss</th>
                    <th className="p-2.5">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-zinc-800/80 text-zinc-300">
                  {benchmarks.evaluation.slice(0, 6).map((row, idx) => (
                    <tr key={idx} className="hover:bg-zinc-800/30">
                      <td className="p-2.5 font-medium text-zinc-200">{row.set}</td>
                      <td className="p-2.5">{row.level}</td>
                      <td className="p-2.5 font-mono text-zinc-400">{row.head}</td>
                      <td className="p-2.5 font-semibold text-zinc-100">{row.AUC}</td>
                      <td className="p-2.5">{row.LogLoss}</td>
                      <td className="p-2.5">{row.ACC}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

      </main>

      {/* Clean Footer */}
      <footer className="border-t border-zinc-800/80 py-6 px-6 text-center text-xs text-zinc-500 font-sans">
        MayaLens • Integrated Deepfake Detection powered by Explainable AI
      </footer>

    </div>
  );
}
