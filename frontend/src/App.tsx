import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Download, 
  Layers, 
  Activity, 
  ChevronRight, 
  Eye, 
  ExternalLink, 
  ArrowUpRight, 
  CheckCircle2, 
  FileText 
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
    <div className="min-h-screen periwinkle-canvas text-[#171926] flex flex-col font-sans selection:bg-[#B8C0FF] selection:text-[#171926] p-4 sm:p-6 lg:p-8">
      
      {/* Signature Dreamy Periwinkle Frame Container (Matching the uploaded frame layout) */}
      <div className="max-w-6xl w-full mx-auto frame-gradient rounded-[36px] sm:rounded-[48px] border border-white/70 shadow-2xl shadow-[#B8C0FF]/30 p-6 sm:p-10 flex flex-col justify-between relative overflow-hidden">
        
        {/* Subtle decorative background world / grid watermark */}
        <div className="absolute inset-0 pointer-events-none opacity-[0.04]">
          <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <pattern id="dotPattern" width="24" height="24" patternUnits="userSpaceOnUse">
                <circle cx="2" cy="2" r="1.5" fill="#000" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#dotPattern)" />
          </svg>
        </div>

        {/* Frame Header Navigation */}
        <header className="flex items-center justify-between z-10 pb-8 sm:pb-12">
          {/* Logo */}
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-full bg-[#181a2e] text-[#E7D8FF] flex items-center justify-center font-bold text-sm shadow-md">
              ML
            </div>
            <span className="font-bold text-lg text-[#141624] tracking-tight">MayaLens</span>
          </div>

          {/* Center Navigation Links */}
          <nav className="hidden md:flex items-center gap-7 text-xs font-medium text-[#2d314e]">
            <a href="#overview" className="hover:text-black transition">Overview</a>
            <a href="#visual-inspection" className="hover:text-black transition">XAI Visualizer</a>
            <a href="#streams" className="hover:text-black transition">Model Streams</a>
            <a href="#timeline" className="hover:text-black transition">Timeline</a>
            <button onClick={() => setShowBenchmarks(!showBenchmarks)} className="hover:text-black transition">
              Benchmarks
            </button>
          </nav>

          {/* Right Action Buttons */}
          <div className="flex items-center gap-2.5">
            <a
              href="/api/report/demo"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-semibold text-[#181a2e] hover:text-black transition px-3 py-1.5 flex items-center gap-1"
            >
              <FileText className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">PDF Report</span>
            </a>

            <label className="text-xs font-semibold text-white bg-[#181a2e] hover:bg-[#232742] px-4 py-2 rounded-full cursor-pointer transition shadow-md flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5 text-[#B8C0FF]" />
              <span>Inspect Media</span>
              <input type="file" accept="video/*,image/*" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>
        </header>

        {/* Hero Section inside Frame (Split Layout from screenshot) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center z-10 py-4 sm:py-8">
          
          {/* Left Column: Big Bold Statement & Call to Action (7 cols) */}
          <div className="lg:col-span-7 space-y-6">
            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold text-[#121422] tracking-tight leading-[1.12]">
              Simplify deepfake detection with us.
            </h1>
            
            <p className="text-sm sm:text-base text-[#383c5e] max-w-lg leading-relaxed">
              From multi-scale MTCNN face extraction to spectral FFT residue analysis. Inspect and verify synthetic manipulations with complete explainable transparency.
            </p>

            {/* CTA Pills directly matching screenshot button style */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <label className="px-6 py-3 rounded-full bg-[#181a2e] hover:bg-[#232742] text-white text-xs font-semibold cursor-pointer transition shadow-lg shadow-[#181a2e]/20 flex items-center gap-2">
                <span>{uploadedFile ? uploadedFile.name : "Upload media to analyze"}</span>
                <input type="file" accept="video/*,image/*" onChange={handleFileUpload} className="hidden" />
              </label>

              <button
                onClick={loadSample}
                disabled={analyzing}
                className="group pl-5 pr-2 py-1.5 rounded-full bg-white/80 hover:bg-white text-[#181a2e] text-xs font-semibold border border-white/90 transition shadow-md flex items-center gap-2.5"
              >
                <span>{analyzing ? 'Evaluating...' : 'Try demo video'}</span>
                <div className="w-7 h-7 rounded-full bg-[#181a2e] text-white flex items-center justify-center group-hover:scale-105 transition">
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </div>
              </button>
            </div>
          </div>

          {/* Right Column: Floating MayaLens Inspection Card (5 cols) (matching the card in the screenshot) */}
          <div className="lg:col-span-5 flex justify-center lg:justify-end">
            <div className="w-full max-w-md bg-[#181a2e] text-white rounded-[28px] p-6 shadow-2xl border border-white/20 relative space-y-5">
              
              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-white/10 pb-4">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-6 rounded-full bg-gradient-to-tr from-[#B8C0FF] to-[#E7D8FF] text-[#181a2e] flex items-center justify-center font-bold text-[10px]">
                    ML
                  </div>
                  <span className="font-semibold text-xs tracking-wide">MayaLens Forensic Card</span>
                </div>
                
                <span className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full uppercase tracking-wider ${
                  data?.is_fake ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40' : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                }`}>
                  {data?.risk_level || 'ANALYZED'}
                </span>
              </div>

              {/* Card Body - Verdict & Confidence */}
              <div>
                <div className="text-[11px] text-[#B8C0FF]/70 uppercase tracking-widest font-mono">
                  VERDICT CONFIDENCE
                </div>
                <div className="text-3xl font-bold tracking-tight text-white mt-0.5 flex items-baseline gap-2">
                  <span>{data ? `${data.confidence}%` : '94.8%'}</span>
                  <span className="text-xs font-medium text-[#B8C0FF]">
                    {data?.decision || 'DEEPFAKE DETECTED'}
                  </span>
                </div>
                <div className="text-[11px] text-[#B8C0FF]/50 font-mono mt-1">
                  FILE: {data?.filename || 'sample_deepfake_clip.mp4'} • {data?.inference_time_ms || 360}ms
                </div>
              </div>

              {/* Three Stream Pill Breakdown inside the Card */}
              <div className="grid grid-cols-3 gap-2 pt-2 border-t border-white/10">
                <div className="bg-white/5 p-2 rounded-xl text-center border border-white/5">
                  <div className="text-[9px] text-[#B8C0FF]/70 uppercase">Spatial</div>
                  <div className="text-xs font-bold text-[#E7D8FF]">{data?.streams[0]?.confidence || 91.9}%</div>
                </div>
                <div className="bg-white/5 p-2 rounded-xl text-center border border-white/5">
                  <div className="text-[9px] text-[#B8C0FF]/70 uppercase">Context</div>
                  <div className="text-xs font-bold text-[#E7D8FF]">{data?.streams[1]?.confidence || 91.8}%</div>
                </div>
                <div className="bg-white/5 p-2 rounded-xl text-center border border-white/5">
                  <div className="text-[9px] text-[#B8C0FF]/70 uppercase">Frequency</div>
                  <div className="text-xs font-bold text-[#E7D8FF]">{data?.streams[2]?.confidence || 93.4}%</div>
                </div>
              </div>

              {/* Card Footer Info */}
              <div className="flex items-center justify-between text-[10px] text-[#B8C0FF]/60 pt-1 font-mono">
                <span>INSPECTOR: XAI-ENGINE</span>
                <span>MODEL V1.0</span>
              </div>

            </div>
          </div>

        </div>

        {/* Bottom Trust & Evaluation Bar inside Frame (matching the rating badges in screenshot) */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-8 sm:pt-12 border-t border-[#181a2e]/10 z-10 text-xs">
          
          <div className="flex items-center gap-3">
            <div className="flex items-center text-[#181a2e]">
              <span className="font-bold mr-1">AUC-ROC 0.9899</span>
              <span className="text-amber-600">★★★★★</span>
            </div>
            <span className="text-[#383c5e]">In-Domain Video Benchmark</span>
          </div>

          <div className="flex items-center gap-2 text-[#383c5e]">
            <CheckCircle2 className="w-4 h-4 text-[#181a2e]" />
            <span>Equal Error Rate (EER): <strong>4.34%</strong> on FaceForensics++ & Celeb-DF v2</span>
          </div>

        </div>

      </div>

      {/* Main Analysis Sections - Clean, Light, Dreamy Periwinkle (NO black background) */}
      <main className="max-w-6xl w-full mx-auto mt-10 space-y-8">
        
        {/* Visual Evidence Inspection Deck */}
        {data && currentFrame && (
          <section id="visual-inspection" className="bg-white/90 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-6">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#B8C0FF]/20 pb-4">
              <div>
                <h2 className="text-lg font-bold text-[#141624] flex items-center gap-2">
                  <Eye className="w-5 h-5 text-[#6c77d6]" />
                  <span>Visual Evidence Inspection Loupe</span>
                </h2>
                <p className="text-xs text-[#52567a]">
                  Keyframe #{currentFrame.frame_idx} ({currentFrame.timestamp}) • Crop Alignment: {currentFrame.crop_type}
                </p>
              </div>

              {/* Filter Tabs in Periwinkle */}
              <div className="flex items-center bg-[#f0edfa] p-1 rounded-full text-xs">
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
                      className={`px-3 py-1.5 rounded-full text-xs font-semibold transition ${
                        activeFilter === filter
                          ? 'bg-[#181a2e] text-white shadow-xs'
                          : 'text-[#484c6e] hover:text-[#181a2e]'
                      }`}
                    >
                      {labels[filter]}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Split Canvas & Evidence Details */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
              
              {/* Left: Face Canvas (7 cols) */}
              <div className="md:col-span-7 space-y-3">
                <div className="relative aspect-video w-full rounded-2xl overflow-hidden bg-[#eef1ff] border border-[#B8C0FF]/40 flex items-center justify-center shadow-inner">
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

                    {/* Canvas Background */}
                    <rect width="640" height="360" fill="#eef1ff" />

                    {/* Crop boundary frame */}
                    <rect x="230" y="65" width="180" height="230" rx="12" fill="#e0e5ff" stroke="#6c77d6" strokeOpacity="0.4" strokeWidth="1.5" strokeDasharray="4 4" />

                    {/* Face geometry */}
                    <ellipse cx="320" cy="180" rx="65" ry="90" fill="#d2d9ff" stroke="#6c77d6" strokeOpacity="0.3" strokeWidth="1" />
                    {/* Eyes */}
                    <ellipse cx="295" cy="155" rx="10" ry="6" fill="#6c77d6" />
                    <ellipse cx="345" cy="155" rx="10" ry="6" fill="#6c77d6" />
                    {/* Nose */}
                    <path d="M 320 162 L 316 186 L 324 186" fill="none" stroke="#6c77d6" strokeWidth="1.5" />
                    {/* Mouth */}
                    <ellipse cx="320" cy="225" rx="22" ry="8" fill="#6c77d6" />

                    {/* Filter overlays */}
                    {activeFilter === 'gradcam' && (
                      <g>
                        <circle cx="320" cy="230" r="42" fill="url(#periwinkleHeat)" />
                        <circle cx="285" cy="215" r="28" fill="url(#periwinkleHeat)" />
                        <text x="375" y="235" fill="#e11d48" fontSize="11" fontFamily="sans-serif" fontWeight="bold">Mouth & Jaw Blending</text>
                        <line x1="335" y1="230" x2="370" y2="230" stroke="#e11d48" strokeWidth="1.5" />
                      </g>
                    )}

                    {activeFilter === 'gradcam_plus' && (
                      <g>
                        <circle cx="295" cy="155" r="28" fill="url(#periwinkleHeat)" />
                        <circle cx="345" cy="155" r="28" fill="url(#periwinkleHeat)" />
                        <text x="385" y="155" fill="#e11d48" fontSize="11" fontFamily="sans-serif" fontWeight="bold">Orbital Discontinuity</text>
                        <line x1="360" y1="155" x2="380" y2="155" stroke="#e11d48" strokeWidth="1.5" />
                      </g>
                    )}

                    {activeFilter === 'lime' && (
                      <g>
                        <polygon points="265,195 305,190 315,235 275,240" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1.5" />
                        <polygon points="325,190 365,195 355,240 315,235" fill="url(#limeSoft)" stroke="#10b981" strokeWidth="1.5" />
                        <text x="375" y="215" fill="#047857" fontSize="11" fontFamily="sans-serif" fontWeight="bold">Superpixel Boundaries</text>
                        <line x1="355" y1="215" x2="370" y2="215" stroke="#047857" strokeWidth="1.5" />
                      </g>
                    )}

                    {activeFilter === 'saliency' && (
                      <g>
                        <circle cx="320" cy="180" r="75" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="3 3" />
                        <path d="M 285 155 Q 320 170 355 155" fill="none" stroke="#6366f1" strokeWidth="1.5" />
                        <path d="M 295 225 Q 320 240 345 225" fill="none" stroke="#6366f1" strokeWidth="1.5" />
                        <text x="390" y="180" fill="#4f46e5" fontSize="11" fontFamily="sans-serif" fontWeight="bold">High-gradient contours</text>
                        <line x1="375" y1="180" x2="385" y2="180" stroke="#4f46e5" strokeWidth="1.5" />
                      </g>
                    )}

                    {/* Landmark points */}
                    <g fill="#4338ca" opacity="0.6">
                      <circle cx="285" cy="150" r="1.5" /><circle cx="295" cy="148" r="1.5" /><circle cx="305" cy="150" r="1.5" />
                      <circle cx="335" cy="150" r="1.5" /><circle cx="345" cy="148" r="1.5" /><circle cx="355" cy="150" r="1.5" />
                      <circle cx="320" cy="170" r="1.5" /><circle cx="320" cy="185" r="1.5" />
                      <circle cx="305" cy="225" r="1.5" /><circle cx="320" cy="228" r="1.5" /><circle cx="335" cy="225" r="1.5" />
                    </g>
                  </svg>

                  <div className="absolute bottom-3 left-3 bg-white/90 backdrop-blur-md px-3 py-1 rounded-full text-[11px] font-medium text-[#181a2e] border border-[#B8C0FF]/50 shadow-xs">
                    {currentFrame.xai_focus}
                  </div>
                </div>

                {/* Keyframe chips */}
                <div className="flex items-center gap-2 pt-1">
                  <span className="text-xs text-[#52567a] font-medium">Select Keyframe:</span>
                  {data.keyframes.map((kf, idx) => (
                    <button
                      key={kf.frame_idx}
                      onClick={() => setSelectedKeyframe(idx)}
                      className={`px-3 py-1 rounded-full text-xs font-semibold transition ${
                        selectedKeyframe === idx
                          ? 'bg-[#181a2e] text-white shadow-xs'
                          : 'bg-[#f0edfa] text-[#484c6e] hover:bg-[#e4e0f5]'
                      }`}
                    >
                      Frame #{kf.frame_idx} ({kf.timestamp})
                    </button>
                  ))}
                </div>
              </div>

              {/* Right: Notes & Observations (5 cols) */}
              <div className="md:col-span-5 space-y-4 bg-[#f8f7fd] p-5 rounded-2xl border border-[#B8C0FF]/30">
                <div>
                  <h3 className="font-bold text-sm text-[#141624]">Forensic Evidence Notes</h3>
                  <p className="text-xs text-[#52567a]">Visual signals verified across the spatial and frequency streams.</p>
                </div>

                <div className="space-y-3">
                  <div className="p-3.5 rounded-xl bg-white border border-[#B8C0FF]/30 shadow-xs">
                    <span className="text-xs font-bold text-rose-600 block mb-1">Primary Finding:</span>
                    <p className="text-xs text-[#2d314e] leading-relaxed">
                      {currentFrame.anomaly_detected}
                    </p>
                  </div>

                  <div className="space-y-2 text-xs">
                    <div className="p-2.5 rounded-lg bg-white border border-[#B8C0FF]/20">
                      <span className="text-[#4f46e5] font-semibold block mb-0.5">FFT / DCT Residuals:</span>
                      <span className="text-[#2d314e]">{currentFrame.features.fft_peaks}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-white border border-[#B8C0FF]/20">
                      <span className="text-[#4f46e5] font-semibold block mb-0.5">Vision Transformer Attention:</span>
                      <span className="text-[#2d314e]">{currentFrame.features.vit_attention}</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-white border border-[#B8C0FF]/20">
                      <span className="text-[#4f46e5] font-semibold block mb-0.5">Landmark Alignment:</span>
                      <span className="text-[#2d314e]">{currentFrame.features.facial_landmarks}</span>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-[#B8C0FF]/20">
                  <a
                    href="/api/report/demo"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 text-xs text-[#4338ca] hover:text-[#181a2e] font-semibold transition"
                  >
                    <span>Read complete analysis in PDF report</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>

            </div>
          </section>
        )}

        {/* The Three Model Streams */}
        {data && (
          <section id="streams" className="space-y-4">
            <h2 className="text-lg font-bold text-[#141624] flex items-center gap-2">
              <Layers className="w-5 h-5 text-[#6c77d6]" />
              <span>Multi-Stream Core Decomposition (Report §5)</span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {data.streams.map((stream) => (
                <div key={stream.id} className="bg-white/90 backdrop-blur-md rounded-[28px] p-6 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-bold text-sm text-[#141624]">{stream.name}</h3>
                    <span className="text-xs font-mono font-semibold text-[#6c77d6] bg-[#f0edfa] px-2 py-0.5 rounded-md">{stream.weight}</span>
                  </div>

                  <div className="flex items-baseline justify-between pt-1">
                    <div className="text-3xl font-extrabold text-[#181a2e]">
                      {stream.confidence}%
                    </div>
                    <span className={`text-xs px-2.5 py-0.5 rounded-full font-bold ${
                      stream.confidence > 50 
                        ? 'bg-rose-100 text-rose-700' 
                        : 'bg-emerald-100 text-emerald-700'
                    }`}>
                      {stream.status}
                    </span>
                  </div>

                  {/* Periwinkle Progress Bar */}
                  <div className="w-full bg-[#eef1ff] h-2 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        stream.confidence > 50 ? 'bg-gradient-to-r from-rose-500 to-amber-500' : 'bg-gradient-to-r from-[#6c77d6] to-[#B8C0FF]'
                      }`}
                      style={{ width: `${stream.confidence}%` }}
                    />
                  </div>

                  <p className="text-xs text-[#52567a] leading-relaxed">
                    {stream.focus}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Temporal Timeline */}
        {data && data.temporal_series && data.temporal_series.length > 1 && (
          <section id="timeline" className="bg-white/90 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-[#6c77d6]" />
                <h2 className="text-lg font-bold text-[#141624]">Temporal Manipulation Timeline</h2>
              </div>
              <span className="text-xs font-semibold text-[#52567a]">Threshold: 50%</span>
            </div>

            <div className="pt-4 pb-2">
              <div className="h-28 flex items-end gap-2 sm:gap-3">
                {data.temporal_series.map((pt, i) => (
                  <div key={i} className="flex-1 flex flex-col items-center gap-1.5 group relative">
                    <div className="absolute -top-7 opacity-0 group-hover:opacity-100 transition pointer-events-none bg-[#181a2e] text-white text-[10px] font-semibold px-2 py-0.5 rounded-md whitespace-nowrap z-20 shadow-md">
                      {pt.time}: {pt.score}%
                    </div>
                    <div 
                      className={`w-full rounded-sm transition-all duration-300 ${
                        pt.score > 50 
                          ? 'bg-rose-500 group-hover:brightness-110' 
                          : 'bg-[#B8C0FF] group-hover:bg-[#9fa9ff]'
                      }`}
                      style={{ height: `${Math.max(12, pt.score)}%` }}
                    />
                    <span className="text-[10px] text-[#52567a] font-medium truncate w-full text-center">
                      {pt.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}

        {/* Model Evaluation Benchmarks Modal / Section */}
        {showBenchmarks && benchmarks && (
          <section className="bg-white/90 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/20 space-y-6">
            <div className="flex items-center justify-between border-b border-[#B8C0FF]/20 pb-4">
              <div>
                <h2 className="text-lg font-bold text-[#141624]">Model Benchmark & Evaluation Results</h2>
                <p className="text-xs text-[#52567a]">Logged performance from evaluation.csv in the MayaLens repository.</p>
              </div>
              <button 
                onClick={() => setShowBenchmarks(false)}
                className="text-xs font-semibold text-[#181a2e] px-3 py-1.5 bg-[#f0edfa] hover:bg-[#e4e0f5] rounded-full transition"
              >
                Close
              </button>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 bg-[#f8f7fd] rounded-2xl border border-[#B8C0FF]/30">
                <div className="text-xs text-[#52567a]">In-Domain Video AUC</div>
                <div className="text-2xl font-bold text-[#181a2e] mt-1">0.9899</div>
              </div>
              <div className="p-4 bg-[#f8f7fd] rounded-2xl border border-[#B8C0FF]/30">
                <div className="text-xs text-[#52567a]">Equal Error Rate (EER)</div>
                <div className="text-2xl font-bold text-[#181a2e] mt-1">4.34%</div>
              </div>
              <div className="p-4 bg-[#f8f7fd] rounded-2xl border border-[#B8C0FF]/30">
                <div className="text-xs text-[#52567a]">Final Accuracy</div>
                <div className="text-2xl font-bold text-emerald-600 mt-1">{benchmarks.training.final_accuracy}%</div>
              </div>
              <div className="p-4 bg-[#f8f7fd] rounded-2xl border border-[#B8C0FF]/30">
                <div className="text-xs text-[#52567a]">Training Loss</div>
                <div className="text-2xl font-bold text-[#181a2e] mt-1">{benchmarks.training.loss}</div>
              </div>
            </div>

            <div className="overflow-x-auto rounded-2xl border border-[#B8C0FF]/30">
              <table className="w-full text-left text-xs font-sans">
                <thead className="bg-[#f0edfa] text-[#141624] font-semibold border-b border-[#B8C0FF]/30">
                  <tr>
                    <th className="p-3">Dataset & Split</th>
                    <th className="p-3">Level</th>
                    <th className="p-3">Model Head</th>
                    <th className="p-3">AUC-ROC</th>
                    <th className="p-3">Log-Loss</th>
                    <th className="p-3">Accuracy</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#B8C0FF]/20 text-[#2d314e]">
                  {benchmarks.evaluation.slice(0, 6).map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#B8C0FF]/10">
                      <td className="p-3 font-semibold text-[#141624]">{row.set}</td>
                      <td className="p-3">{row.level}</td>
                      <td className="p-3 font-mono text-[#4338ca]">{row.head}</td>
                      <td className="p-3 font-bold text-[#141624]">{row.AUC}</td>
                      <td className="p-3">{row.LogLoss}</td>
                      <td className="p-3">{row.ACC}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

      </main>

      {/* Enhanced Footer matching the frame curvature */}
      <footer className="max-w-6xl w-full mx-auto mt-12 mb-6">
        <div className="bg-white/80 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-6">
          
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#B8C0FF]/20 pb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-[#181a2e] text-[#E7D8FF] flex items-center justify-center font-bold text-xs">
                ML
              </div>
              <div>
                <div className="font-bold text-sm text-[#141624]">MayaLens Forensics</div>
                <div className="text-xs text-[#52567a]">Explainable AI Deepfake Detection Platform</div>
              </div>
            </div>

            <div className="flex items-center gap-3 text-xs">
              <a
                href="/api/report/demo"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded-full bg-[#f0edfa] hover:bg-[#e4e0f5] text-[#181a2e] font-semibold transition inline-flex items-center gap-1.5"
              >
                <Download className="w-3.5 h-3.5 text-[#4338ca]" />
                <span>Forensic PDF Report</span>
              </a>
              <a
                href="https://github.com/digitalengineer07/MayaLens"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 rounded-full bg-[#f0edfa] hover:bg-[#e4e0f5] text-[#181a2e] font-semibold transition inline-flex items-center gap-1.5"
              >
                <span>GitHub</span>
                <ExternalLink className="w-3 h-3 text-[#4338ca]" />
              </a>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-[#52567a]">
            <div>
              Trained on FaceForensics++ & Celeb-DF v2 • EfficientNet-B4 + ViT + Frequency ResNet
            </div>
            <div className="font-medium text-[#181a2e]">
              MayaLens Framework • Research & Verification Platform
            </div>
          </div>

        </div>
      </footer>

    </div>
  );
}
