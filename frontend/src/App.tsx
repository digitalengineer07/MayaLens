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
  Eye,
  ExternalLink,
  Sparkles
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
    <div className="min-h-screen bg-[#0b0c16] text-[#e5e7f2] flex flex-col font-sans periwinkle-glow selection:bg-[#B8C0FF]/30 selection:text-[#E7D8FF]">
      
      {/* Floating Rounded Header with Dreamy Periwinkle Accent */}
      <header className="sticky top-4 z-50 px-4 sm:px-6">
        <div className="max-w-5xl mx-auto rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/80 backdrop-blur-xl px-4 sm:px-6 py-3 shadow-xl shadow-black/40 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-[#B8C0FF] to-[#E7D8FF] text-[#0b0c16] flex items-center justify-center font-bold text-sm shadow-md shadow-[#B8C0FF]/20 ring-1 ring-white/20">
              ML
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-base text-white tracking-tight">MayaLens</span>
                <span className="inline-flex items-center gap-1 text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded-full bg-[#B8C0FF]/15 text-[#E7D8FF] border border-[#B8C0FF]/30">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#B8C0FF] animate-pulse" />
                  Online
                </span>
              </div>
              <p className="text-[11px] text-[#B8C0FF]/70 font-normal hidden sm:block">Explainable Deepfake Forensics</p>
            </div>
          </div>

          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setShowBenchmarks(!showBenchmarks)}
              className="text-xs text-[#E7D8FF] hover:text-white px-3 py-1.5 rounded-xl border border-[#B8C0FF]/25 hover:border-[#B8C0FF]/40 bg-[#171a2e]/80 hover:bg-[#1e233d] transition flex items-center gap-1.5 shadow-xs"
            >
              <BarChart3 className="w-3.5 h-3.5 text-[#B8C0FF]" />
              <span className="hidden sm:inline">Model</span> Evaluation
            </button>

            <a
              href="/api/report/demo"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-semibold text-[#0b0c16] bg-gradient-to-r from-[#B8C0FF] via-[#cfd7ff] to-[#E7D8FF] hover:opacity-95 px-3.5 py-1.5 rounded-xl transition shadow-md shadow-[#B8C0FF]/25 flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5 text-[#0b0c16]" />
              <span>PDF Report</span>
            </a>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-8 space-y-8">
        
        {/* Minimal Hero with Dreamy Periwinkle Text Gradient */}
        <section className="space-y-4">
          <div className="space-y-2">
            <div className="inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1 rounded-full bg-[#B8C0FF]/10 text-[#E7D8FF] border border-[#B8C0FF]/20">
              <Sparkles className="w-3.5 h-3.5 text-[#B8C0FF]" />
              <span>Multi-Stream Explainable AI Framework</span>
            </div>
            <h1 className="text-2xl sm:text-4xl font-semibold text-white tracking-tight">
              Transparent media verification, <br className="hidden sm:inline" />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-[#B8C0FF] to-[#E7D8FF]">
                grounded in visual evidence.
              </span>
            </h1>
            <p className="text-sm text-[#B8C0FF]/80 max-w-xl leading-relaxed">
              Upload a video or photo to identify neural face-swaps and manipulation artifacts, verified across spatial textures, face lighting, and frequency spectrums.
            </p>
          </div>

          {/* Clean Dropzone Container with Periwinkle Highlights */}
          <div className="rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/60 backdrop-blur-md p-5 sm:p-6 transition hover:border-[#B8C0FF]/40 shadow-lg shadow-black/20">
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
              
              <div className="relative flex-1 group">
                <input 
                  type="file" 
                  accept="video/*,image/*" 
                  onChange={handleFileUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer z-10" 
                />
                <div className="flex items-center gap-3.5 cursor-pointer">
                  <div className="w-10 h-10 rounded-xl bg-[#181b30] border border-[#B8C0FF]/30 flex items-center justify-center text-[#B8C0FF] group-hover:text-[#E7D8FF] group-hover:border-[#B8C0FF]/60 transition shrink-0 shadow-inner">
                    <Upload className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-white">
                      {uploadedFile ? uploadedFile.name : "Drop a video or image to inspect"}
                    </div>
                    <div className="text-xs text-[#B8C0FF]/60">
                      MP4, AVI, MOV, JPG, PNG • Evaluated using multi-scale MTCNN alignment
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0 border-t sm:border-t-0 pt-3 sm:pt-0 border-[#B8C0FF]/15">
                <button
                  onClick={loadSample}
                  disabled={analyzing}
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 text-xs font-medium text-[#E7D8FF] hover:text-white px-4 py-2 rounded-xl bg-[#171a2e] hover:bg-[#1f233e] border border-[#B8C0FF]/25 transition disabled:opacity-50 shadow-xs"
                >
                  <RefreshCw className={`w-3.5 h-3.5 text-[#B8C0FF] ${analyzing ? 'animate-spin' : ''}`} />
                  <span>{analyzing ? 'Evaluating...' : 'Try Demo Video'}</span>
                </button>
              </div>

            </div>
          </div>
        </section>

        {/* Primary Verdict Card */}
        {data && (
          <section className={`rounded-2xl border p-6 transition shadow-xl ${
            data.is_fake 
              ? 'bg-rose-950/20 border-rose-500/30' 
              : 'bg-emerald-950/20 border-emerald-500/30'
          }`}>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
              
              <div className="flex items-start gap-4">
                <div className={`p-3 rounded-xl mt-0.5 shrink-0 ${
                  data.is_fake 
                    ? 'bg-rose-500/10 text-rose-300 border border-rose-500/30' 
                    : 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/30'
                }`}>
                  {data.is_fake ? <ShieldAlert className="w-6 h-6" /> : <ShieldCheck className="w-6 h-6" />}
                </div>

                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <h2 className="text-xl font-semibold text-white">
                      {data.is_fake ? "Deepfake Detected" : "Authentic Media"}
                    </h2>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                      data.is_fake 
                        ? 'bg-rose-500/20 text-rose-200 border border-rose-500/40' 
                        : 'bg-emerald-500/20 text-emerald-200 border border-emerald-500/40'
                    }`}>
                      {data.confidence}% confidence
                    </span>
                    <span className="text-xs text-[#B8C0FF]/60 font-mono">
                      • {data.inference_time_ms} ms
                    </span>
                  </div>
                  <p className="text-sm text-zinc-300 max-w-2xl leading-relaxed">
                    {data.summary}
                  </p>
                </div>
              </div>

              <div className="sm:text-right shrink-0">
                <div className="text-xs text-[#B8C0FF]/60 uppercase tracking-wider font-mono">SOURCE FILE</div>
                <div className="text-sm font-medium text-[#E7D8FF] truncate max-w-[200px]">{data.filename}</div>
              </div>

            </div>
          </section>
        )}

        {/* Visual Inspection Loupe & Evidence Notes */}
        {data && currentFrame && (
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-[#B8C0FF]" />
                <span>Visual Evidence Inspection</span>
              </h3>
              <div className="text-xs text-[#B8C0FF]/70">
                Keyframe #{currentFrame.frame_idx} ({currentFrame.timestamp})
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-12 gap-5 items-start">
              
              {/* Left: Frame Canvas (7 cols) */}
              <div className="md:col-span-7 rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/70 backdrop-blur-md p-4 space-y-3.5 shadow-lg shadow-black/20">
                
                {/* Method Filter Toggle Bar */}
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <span className="text-xs text-[#B8C0FF]/80 font-medium">Heatmap Layer:</span>
                  <div className="flex items-center bg-[#171a2e] p-0.5 rounded-xl border border-[#B8C0FF]/20 text-xs">
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
                          className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                            activeFilter === filter
                              ? 'bg-gradient-to-r from-[#B8C0FF] to-[#E7D8FF] text-[#0b0c16] font-semibold shadow-xs'
                              : 'text-[#B8C0FF]/70 hover:text-white'
                          }`}
                        >
                          {labels[filter]}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Face Canvas */}
                <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-[#0a0b14] border border-[#B8C0FF]/20 flex items-center justify-center">
                  <svg className="w-full h-full max-h-[340px]" viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                      <radialGradient id="periwinkleHeat" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.8" />
                        <stop offset="50%" stopColor="#B8C0FF" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="#B8C0FF" stopOpacity="0" />
                      </radialGradient>
                      <radialGradient id="limeSoft" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#10b981" stopOpacity="0.85" />
                        <stop offset="60%" stopColor="#059669" stopOpacity="0.35" />
                        <stop offset="100%" stopColor="#059669" stopOpacity="0" />
                      </radialGradient>
                    </defs>

                    {/* Dark Canvas Background */}
                    <rect width="640" height="360" fill="#090a12" />

                    {/* Crop boundary frame */}
                    <rect x="230" y="65" width="180" height="230" rx="10" fill="#111320" stroke="#B8C0FF" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="3 3" />

                    {/* Face geometry */}
                    <ellipse cx="320" cy="180" rx="65" ry="90" fill="#181a2e" stroke="#B8C0FF" strokeOpacity="0.2" strokeWidth="1" />
                    {/* Eyes */}
                    <ellipse cx="295" cy="155" rx="10" ry="6" fill="#313654" />
                    <ellipse cx="345" cy="155" rx="10" ry="6" fill="#313654" />
                    {/* Nose */}
                    <path d="M 320 162 L 316 186 L 324 186" fill="none" stroke="#4a527d" strokeWidth="1.5" />
                    {/* Mouth */}
                    <ellipse cx="320" cy="225" rx="22" ry="8" fill="#313654" />

                    {/* Filter overlays */}
                    {activeFilter === 'gradcam' && (
                      <g>
                        <circle cx="320" cy="230" r="42" fill="url(#periwinkleHeat)" />
                        <circle cx="285" cy="215" r="28" fill="url(#periwinkleHeat)" />
                        <text x="375" y="235" fill="#E7D8FF" fontSize="11" fontFamily="sans-serif" fontWeight="600">Mouth & Jaw Blending</text>
                        <line x1="335" y1="230" x2="370" y2="230" stroke="#E7D8FF" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'gradcam_plus' && (
                      <g>
                        <circle cx="295" cy="155" r="28" fill="url(#periwinkleHeat)" />
                        <circle cx="345" cy="155" r="28" fill="url(#periwinkleHeat)" />
                        <text x="385" y="155" fill="#E7D8FF" fontSize="11" fontFamily="sans-serif" fontWeight="600">Periocular Texture Discontinuity</text>
                        <line x1="360" y1="155" x2="380" y2="155" stroke="#E7D8FF" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'lime' && (
                      <g>
                        <polygon points="265,195 305,190 315,235 275,240" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1" />
                        <polygon points="325,190 365,195 355,240 315,235" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1" />
                        <text x="375" y="215" fill="#10b981" fontSize="11" fontFamily="sans-serif" fontWeight="600">Superpixel Boundaries</text>
                        <line x1="355" y1="215" x2="370" y2="215" stroke="#10b981" strokeWidth="1" />
                      </g>
                    )}

                    {activeFilter === 'saliency' && (
                      <g>
                        <circle cx="320" cy="180" r="75" fill="none" stroke="#B8C0FF" strokeWidth="1.5" strokeDasharray="3 3" />
                        <path d="M 285 155 Q 320 170 355 155" fill="none" stroke="#E7D8FF" strokeWidth="1.5" />
                        <path d="M 295 225 Q 320 240 345 225" fill="none" stroke="#E7D8FF" strokeWidth="1.5" />
                        <text x="390" y="180" fill="#E7D8FF" fontSize="11" fontFamily="sans-serif" fontWeight="600">High-gradient contours</text>
                        <line x1="375" y1="180" x2="385" y2="180" stroke="#B8C0FF" strokeWidth="1" />
                      </g>
                    )}

                    {/* Landmarks points */}
                    <g fill="#B8C0FF" opacity="0.6">
                      <circle cx="285" cy="150" r="1.5" /><circle cx="295" cy="148" r="1.5" /><circle cx="305" cy="150" r="1.5" />
                      <circle cx="335" cy="150" r="1.5" /><circle cx="345" cy="148" r="1.5" /><circle cx="355" cy="150" r="1.5" />
                      <circle cx="320" cy="170" r="1.5" /><circle cx="320" cy="185" r="1.5" />
                      <circle cx="305" cy="225" r="1.5" /><circle cx="320" cy="228" r="1.5" /><circle cx="335" cy="225" r="1.5" />
                    </g>
                  </svg>

                  <div className="absolute bottom-2.5 left-2.5 bg-[#121424]/90 backdrop-blur-md px-2.5 py-1 rounded-md text-[11px] text-[#E7D8FF] border border-[#B8C0FF]/25">
                    {currentFrame.xai_focus}
                  </div>
                </div>

                {/* Keyframe selector chips */}
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-xs text-[#B8C0FF]/70">Keyframes:</span>
                  {data.keyframes.map((kf, idx) => (
                    <button
                      key={kf.frame_idx}
                      onClick={() => setSelectedKeyframe(idx)}
                      className={`px-3 py-1.5 rounded-xl text-xs font-medium transition ${
                        selectedKeyframe === idx
                          ? 'bg-gradient-to-r from-[#B8C0FF] to-[#E7D8FF] text-[#0b0c16] font-semibold shadow-sm'
                          : 'bg-[#171a2e] text-[#B8C0FF]/80 hover:text-white border border-[#B8C0FF]/20'
                      }`}
                    >
                      Frame #{kf.frame_idx} ({kf.timestamp})
                    </button>
                  ))}
                </div>

              </div>

              {/* Right: Forensic Notes & Findings (5 cols) */}
              <div className="md:col-span-5 rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/70 backdrop-blur-md p-5 space-y-4 shadow-lg shadow-black/20">
                <div>
                  <h4 className="font-semibold text-sm text-white">Findings & Evidence</h4>
                  <p className="text-xs text-[#B8C0FF]/70">Observable anomalies detected by the model streams on this frame.</p>
                </div>

                <div className="space-y-3">
                  <div className="p-3.5 rounded-xl bg-[#171a2e]/90 border border-[#B8C0FF]/20">
                    <span className="text-xs font-semibold text-[#E7D8FF] block mb-1">Primary Observation</span>
                    <p className="text-xs text-zinc-300 leading-relaxed">
                      {currentFrame.anomaly_detected}
                    </p>
                  </div>

                  <div className="space-y-2.5 text-xs">
                    <div className="p-2.5 rounded-xl bg-[#171a2e]/50 border border-[#B8C0FF]/15">
                      <span className="text-[#B8C0FF] font-medium block mb-0.5">Frequency Domain (FFT / DCT):</span>
                      <span className="text-zinc-300">{currentFrame.features.fft_peaks}</span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#171a2e]/50 border border-[#B8C0FF]/15">
                      <span className="text-[#B8C0FF] font-medium block mb-0.5">Vision Transformer Attention:</span>
                      <span className="text-zinc-300">{currentFrame.features.vit_attention}</span>
                    </div>

                    <div className="p-2.5 rounded-xl bg-[#171a2e]/50 border border-[#B8C0FF]/15">
                      <span className="text-[#B8C0FF] font-medium block mb-0.5">Facial Landmarks Alignment:</span>
                      <span className="text-zinc-300">{currentFrame.features.facial_landmarks}</span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-[#B8C0FF]/15">
                  <a
                    href="/api/report/demo"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-[#B8C0FF] hover:text-[#E7D8FF] transition"
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
            <h3 className="text-sm font-semibold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-[#B8C0FF]" />
              <span>Multi-Stream Model Breakdown</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {data.streams.map((stream) => (
                <div key={stream.id} className="rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/60 backdrop-blur-md p-5 space-y-3 shadow-lg shadow-black/20 hover:border-[#B8C0FF]/40 transition">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-sm text-white">{stream.name}</h4>
                    <span className="text-xs text-[#B8C0FF]/70 font-mono">{stream.weight}</span>
                  </div>

                  <div className="flex items-baseline justify-between pt-1">
                    <div className="text-2xl font-semibold text-[#E7D8FF]">
                      {stream.confidence}%
                    </div>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full font-medium ${
                      stream.confidence > 50 
                        ? 'bg-rose-500/15 text-rose-300 border border-rose-500/30' 
                        : 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30'
                    }`}>
                      {stream.status}
                    </span>
                  </div>

                  {/* Clean Minimal Progress Bar with Dreamy Periwinkle Gradient */}
                  <div className="w-full bg-[#181b30] h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-500 bg-gradient-to-r from-[#B8C0FF] to-[#E7D8FF]"
                      style={{ width: `${stream.confidence}%` }}
                    />
                  </div>

                  <p className="text-xs text-[#B8C0FF]/80 leading-relaxed">
                    {stream.focus}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Video Temporal Timeline */}
        {data && data.temporal_series && data.temporal_series.length > 1 && (
          <section className="rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/60 backdrop-blur-md p-5 space-y-4 shadow-lg shadow-black/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-4 h-4 text-[#B8C0FF]" />
                <h3 className="text-sm font-semibold text-white">Temporal Manipulation Over Time</h3>
              </div>
              <span className="text-xs text-[#B8C0FF]/60 font-mono">Threshold: 50%</span>
            </div>

            <div className="pt-4 pb-2">
              <div className="h-24 flex items-end gap-2 sm:gap-3">
                {data.temporal_series.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group relative">
                    <div className="absolute -top-7 opacity-0 group-hover:opacity-100 transition pointer-events-none bg-[#171a2e] text-[#E7D8FF] border border-[#B8C0FF]/30 text-[10px] px-2 py-0.5 rounded whitespace-nowrap z-20 shadow-md">
                      {pt.time}: {pt.score}%
                    </div>
                    <div 
                      className={`w-full rounded-xs transition-all duration-300 ${
                        pt.score > 50 
                          ? 'bg-gradient-to-t from-rose-500/70 to-rose-400 group-hover:brightness-125' 
                          : 'bg-gradient-to-t from-[#B8C0FF]/40 to-[#E7D8FF]/70 group-hover:brightness-125'
                      }`}
                      style={{ height: `${Math.max(10, pt.score)}%` }}
                    />
                    <span className="text-[10px] text-[#B8C0FF]/60 truncate w-full text-center">
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
          <section className="rounded-2xl border border-[#B8C0FF]/30 bg-[#121424] p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#B8C0FF]/20 pb-3">
              <div>
                <h3 className="font-semibold text-white text-base">Model Benchmark Data</h3>
                <p className="text-xs text-[#B8C0FF]/70">Recorded results from evaluation.csv in the project repository.</p>
              </div>
              <button 
                onClick={() => setShowBenchmarks(false)}
                className="text-xs text-[#E7D8FF] hover:text-white px-2.5 py-1 bg-[#171a2e] rounded-xl border border-[#B8C0FF]/30 transition"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-3 bg-[#171a2e]/60 rounded-xl border border-[#B8C0FF]/15">
                <div className="text-xs text-[#B8C0FF]/60">In-Domain Video AUC</div>
                <div className="text-xl font-bold text-[#E7D8FF] mt-0.5">0.9899</div>
              </div>
              <div className="p-3 bg-[#171a2e]/60 rounded-xl border border-[#B8C0FF]/15">
                <div className="text-xs text-[#B8C0FF]/60">Equal Error Rate (EER)</div>
                <div className="text-xl font-bold text-[#E7D8FF] mt-0.5">4.34%</div>
              </div>
              <div className="p-3 bg-[#171a2e]/60 rounded-xl border border-[#B8C0FF]/15">
                <div className="text-xs text-[#B8C0FF]/60">Final Accuracy</div>
                <div className="text-xl font-bold text-emerald-400 mt-0.5">{benchmarks.training.final_accuracy}%</div>
              </div>
              <div className="p-3 bg-[#171a2e]/60 rounded-xl border border-[#B8C0FF]/15">
                <div className="text-xs text-[#B8C0FF]/60">Loss</div>
                <div className="text-xl font-bold text-[#E7D8FF] mt-0.5">{benchmarks.training.loss}</div>
              </div>
            </div>

            <div className="overflow-x-auto rounded-xl border border-[#B8C0FF]/20">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-[#171a2e] text-[#B8C0FF]/80 border-b border-[#B8C0FF]/20">
                  <tr>
                    <th className="p-2.5">Dataset & Split</th>
                    <th className="p-2.5">Level</th>
                    <th className="p-2.5">Model Head</th>
                    <th className="p-2.5">AUC-ROC</th>
                    <th className="p-2.5">Log-Loss</th>
                    <th className="p-2.5">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#B8C0FF]/10 text-zinc-300">
                  {benchmarks.evaluation.slice(0, 6).map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#B8C0FF]/5">
                      <td className="p-2.5 font-medium text-white">{row.set}</td>
                      <td className="p-2.5">{row.level}</td>
                      <td className="p-2.5 font-mono text-[#B8C0FF]">{row.head}</td>
                      <td className="p-2.5 font-semibold text-[#E7D8FF]">{row.AUC}</td>
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

      {/* Enhanced Rounded Footer Section with Dreamy Periwinkle Styling */}
      <footer className="w-full max-w-5xl mx-auto px-4 sm:px-6 pt-4 pb-12 mt-6">
        <div className="rounded-2xl border border-[#B8C0FF]/20 bg-[#121424]/60 backdrop-blur-md p-6 sm:p-8 space-y-6 shadow-xl shadow-black/30">
          
          {/* Top Row: Brand & Description */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#B8C0FF]/15 pb-6">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-[#B8C0FF] to-[#E7D8FF] text-[#0b0c16] flex items-center justify-center font-bold text-xs shadow-md shadow-[#B8C0FF]/20">
                ML
              </div>
              <div>
                <div className="font-semibold text-sm text-white">MayaLens Forensics</div>
                <div className="text-xs text-[#B8C0FF]/70">Integrated Deepfake Detection powered by Explainable AI</div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2 text-xs">
              <a
                href="/api/report/demo"
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-xl border border-[#B8C0FF]/25 bg-[#171a2e]/80 hover:bg-[#1f233e] text-[#E7D8FF] hover:text-white transition inline-flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5 text-[#B8C0FF]" />
                <span>Forensic PDF Report</span>
              </a>
              <a
                href="https://github.com/digitalengineer07/MayaLens"
                target="_blank"
                rel="noopener noreferrer"
                className="px-3 py-1.5 rounded-xl border border-[#B8C0FF]/25 bg-[#171a2e]/80 hover:bg-[#1f233e] text-[#E7D8FF] hover:text-white transition inline-flex items-center gap-1.5"
              >
                <span>GitHub</span>
                <ExternalLink className="w-3 h-3 text-[#B8C0FF]" />
              </a>
            </div>
          </div>

          {/* Middle Row: Model Architecture Badges */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-[#B8C0FF]">Pipeline Methodology:</div>
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="px-2.5 py-1 rounded-lg bg-[#171a2e]/80 border border-[#B8C0FF]/15 text-[#E7D8FF]">
                Spatial Stream: EfficientNet-B4 (1.15× crop)
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-[#171a2e]/80 border border-[#B8C0FF]/15 text-[#E7D8FF]">
                Context Stream: Vision Transformer (1.70× crop)
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-[#171a2e]/80 border border-[#B8C0FF]/15 text-[#E7D8FF]">
                Frequency Stream: FFT / DCT Residual ResNet
              </span>
              <span className="px-2.5 py-1 rounded-lg bg-[#171a2e]/80 border border-[#B8C0FF]/15 text-[#E7D8FF]">
                XAI: Grad-CAM · LIME · Saliency
              </span>
            </div>
          </div>

          {/* Bottom Row: Status & Copyright */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-4 border-t border-[#B8C0FF]/15 text-xs text-[#B8C0FF]/60">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-[#B8C0FF] animate-pulse" />
              <span>Engine Status: Online • Trained on FaceForensics++ & Celeb-DF v2</span>
            </div>
            <div>
              MayaLens Framework • Research & Verification Platform
            </div>
          </div>

        </div>
      </footer>

    </div>
  );
}
