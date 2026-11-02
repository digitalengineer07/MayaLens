import React, { useState, useEffect } from 'react';
import { 
  Upload, 
  Download, 
  RefreshCw, 
  Crosshair, 
  Sliders,
  Layers,
  Activity
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
  const [activeXAI, setActiveXAI] = useState<'normal' | 'gradcam' | 'gradcam_plus' | 'lime' | 'saliency'>('gradcam');
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null);
  const [showDossier, setShowDossier] = useState(false);
  const [scrubIndex, setScrubIndex] = useState<number>(0);

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
      setScrubIndex(0);
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
      setScrubIndex(0);
    } catch (err) {
      console.error(err);
      alert('Analysis error. Backend may be offline.');
    } finally {
      setAnalyzing(false);
    }
  };

  const currentFrame = data?.keyframes[selectedKeyframe];

  return (
    <div className="min-h-screen bg-[#0e1115] text-[#cbd5e1] font-mono flex flex-col selection:bg-orange-500/30 selection:text-orange-200">
      
      {/* Precision Top Instrument Bar */}
      <header className="border-b border-[#21262d] bg-[#14181f] px-5 py-3 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="h-3 w-3 rounded-full bg-[#ff5500] shadow-[0_0_8px_#ff5500]" />
          <div className="flex items-baseline gap-2">
            <span className="text-white font-bold tracking-widest text-sm">MAYALENS</span>
            <span className="text-[#64748b] text-[11px] tracking-wider">OPTICAL FORENSIC SYSTEM</span>
          </div>
          <span className="text-[10px] text-[#475569] border border-[#262c36] px-1.5 py-0.5 rounded">
            SPEC: FFT·DCT·VIT·EFFNET
          </span>
        </div>

        {/* Global Controls */}
        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#1c222c] hover:bg-[#252d3a] border border-[#2d3748] text-slate-200 cursor-pointer transition">
            <Upload className="w-3.5 h-3.5 text-orange-400" />
            <span>INSPECT MEDIA</span>
            <input type="file" accept="video/*,image/*" onChange={handleFileUpload} className="hidden" />
          </label>

          <button
            onClick={loadSample}
            disabled={analyzing}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#1c222c] hover:bg-[#252d3a] border border-[#2d3748] text-slate-200 transition"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${analyzing ? 'animate-spin' : ''}`} />
            <span>{analyzing ? 'COMPUTING...' : 'SAMPLE'}</span>
          </button>

          <button
            onClick={() => setShowDossier(!showDossier)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition ${
              showDossier 
                ? 'bg-orange-500/10 border-orange-500/50 text-orange-300' 
                : 'bg-[#1c222c] hover:bg-[#252d3a] border-[#2d3748] text-slate-200'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>METRICS</span>
          </button>

          <a
            href="/api/report/demo"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-[#ff5500] hover:bg-[#ff661a] text-black font-semibold tracking-wide transition shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            <span>DOSSIER PDF</span>
          </a>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-[1500px] w-full mx-auto p-4 sm:p-6 grid grid-cols-1 lg:grid-cols-12 gap-5">
        
        {/* Left Column: The Optical Viewfinder & Light Table (7 cols) */}
        <section className="lg:col-span-7 flex flex-col gap-3">
          
          {/* Viewport Card */}
          <div className="bg-[#12161d] border border-[#222936] rounded-lg overflow-hidden flex flex-col">
            
            {/* Viewport Top Bezel */}
            <div className="bg-[#181d26] border-b border-[#222936] px-4 py-2 flex items-center justify-between text-xs text-[#94a3b8]">
              <div className="flex items-center gap-3">
                <span className="flex items-center gap-1 text-slate-300 font-semibold">
                  <Crosshair className="w-3.5 h-3.5 text-orange-400" />
                  OPTICAL LOUPE
                </span>
                <span className="text-[#64748b]">|</span>
                <span>FRAME: #{currentFrame ? currentFrame.frame_idx : '00'}</span>
                <span className="text-[#64748b]">|</span>
                <span>{currentFrame ? currentFrame.timestamp : '00:00.00'}</span>
              </div>

              <div className="flex items-center gap-2 text-[11px]">
                <span className="text-[#64748b]">CROP:</span>
                <span className="text-orange-400 font-medium">
                  {currentFrame ? currentFrame.crop_type : '1.15x TIGHT'}
                </span>
              </div>
            </div>

            {/* The Lens Canvas / Inspection Plate */}
            <div className="relative aspect-video w-full bg-[#0a0d11] light-table-grid flex items-center justify-center p-4">
              
              {/* Corner Registration Reticles (Camera Viewfinder style) */}
              <div className="absolute top-4 left-4 w-4 h-4 border-t-2 border-l-2 border-[#475569]/60 pointer-events-none" />
              <div className="absolute top-4 right-4 w-4 h-4 border-t-2 border-r-2 border-[#475569]/60 pointer-events-none" />
              <div className="absolute bottom-4 left-4 w-4 h-4 border-b-2 border-l-2 border-[#475569]/60 pointer-events-none" />
              <div className="absolute bottom-4 right-4 w-4 h-4 border-b-2 border-r-2 border-[#475569]/60 pointer-events-none" />

              {/* Center Crosshairs */}
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
                <div className="w-full h-px bg-white" />
                <div className="h-full w-px bg-white absolute" />
              </div>

              {/* SVG Optical Plate with Face Alignment & XAI Signature */}
              <svg className="w-full h-full max-h-[380px]" viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
                <defs>
                  {/* Grad-CAM Heat Signature */}
                  <radialGradient id="camHeat" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#ff5500" stopOpacity="0.85" />
                    <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.45" />
                    <stop offset="100%" stopColor="#f59e0b" stopOpacity="0" />
                  </radialGradient>
                  
                  {/* LIME Superpixel gradient */}
                  <radialGradient id="limeFill" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                    <stop offset="60%" stopColor="#059669" stopOpacity="0.3" />
                    <stop offset="100%" stopColor="#059669" stopOpacity="0" />
                  </radialGradient>
                </defs>

                {/* MTCNN Face Boundary Box */}
                <rect x="220" y="60" width="200" height="240" rx="4" fill="#131922" stroke="#334155" strokeWidth="1" strokeDasharray="3 3" />
                <text x="225" y="75" fill="#64748b" fontSize="9" fontFamily="monospace">ALIGNMENT: 68-PTS NORMALIZED</text>

                {/* Human Face Silhouette */}
                <ellipse cx="320" cy="180" rx="70" ry="95" fill="#1b222d" stroke="#2a3443" strokeWidth="1" />
                {/* Eyes */}
                <ellipse cx="295" cy="155" rx="10" ry="6" fill="#334155" />
                <ellipse cx="345" cy="155" rx="10" ry="6" fill="#334155" />
                {/* Nose bridge */}
                <path d="M 320 162 L 316 188 L 324 188" fill="none" stroke="#475569" strokeWidth="1.5" />
                {/* Mouth */}
                <ellipse cx="320" cy="225" rx="22" ry="8" fill="#334155" />

                {/* Overlays according to active filter */}
                {activeXAI === 'gradcam' && (
                  <g>
                    <circle cx="320" cy="230" r="42" fill="url(#camHeat)" />
                    <circle cx="280" cy="215" r="28" fill="url(#camHeat)" />
                    <line x1="335" y1="230" x2="385" y2="230" stroke="#ff5500" strokeWidth="1" />
                    <rect x="385" y="222" width="125" height="16" fill="#0f141c" stroke="#ff5500" strokeWidth="0.8" />
                    <text x="390" y="233" fill="#ff5500" fontSize="9" fontFamily="monospace">GRAD-CAM: JAW SEAM</text>
                  </g>
                )}

                {activeXAI === 'gradcam_plus' && (
                  <g>
                    <circle cx="295" cy="155" r="28" fill="url(#camHeat)" />
                    <circle cx="345" cy="155" r="28" fill="url(#camHeat)" />
                    <line x1="360" y1="155" x2="400" y2="155" stroke="#ff5500" strokeWidth="1" />
                    <rect x="400" y="147" width="135" height="16" fill="#0f141c" stroke="#ff5500" strokeWidth="0.8" />
                    <text x="405" y="158" fill="#ff5500" fontSize="9" fontFamily="monospace">GRAD-CAM++: PERIOCULAR</text>
                  </g>
                )}

                {activeXAI === 'lime' && (
                  <g>
                    <polygon points="265,195 305,190 315,235 275,240" fill="url(#limeFill)" stroke="#10b981" strokeWidth="1" />
                    <polygon points="325,190 365,195 355,240 315,235" fill="url(#limeFill)" stroke="#10b981" strokeWidth="1" />
                    <line x1="365" y1="210" x2="400" y2="210" stroke="#10b981" strokeWidth="1" />
                    <rect x="400" y="202" width="145" height="16" fill="#0f141c" stroke="#10b981" strokeWidth="0.8" />
                    <text x="405" y="213" fill="#10b981" fontSize="9" fontFamily="monospace">LIME: SLIC BOUNDARY</text>
                  </g>
                )}

                {activeXAI === 'saliency' && (
                  <g>
                    <circle cx="320" cy="180" r="82" fill="none" stroke="#38bdf8" strokeWidth="1" strokeDasharray="4 2" />
                    <path d="M 285 155 Q 320 170 355 155" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
                    <path d="M 295 225 Q 320 240 345 225" fill="none" stroke="#38bdf8" strokeWidth="1.5" />
                    <line x1="375" y1="180" x2="410" y2="180" stroke="#38bdf8" strokeWidth="1" />
                    <rect x="410" y="172" width="135" height="16" fill="#0f141c" stroke="#38bdf8" strokeWidth="0.8" />
                    <text x="415" y="183" fill="#38bdf8" fontSize="9" fontFamily="monospace">SALIENCY: GRADIENTS</text>
                  </g>
                )}

                {/* Landmark Coordinates */}
                <g fill="#ff5500" opacity="0.6">
                  <circle cx="285" cy="150" r="1.5" /><circle cx="295" cy="148" r="1.5" /><circle cx="305" cy="150" r="1.5" />
                  <circle cx="335" cy="150" r="1.5" /><circle cx="345" cy="148" r="1.5" /><circle cx="355" cy="150" r="1.5" />
                  <circle cx="320" cy="170" r="1.5" /><circle cx="320" cy="185" r="1.5" />
                  <circle cx="305" cy="225" r="1.5" /><circle cx="320" cy="228" r="1.5" /><circle cx="335" cy="225" r="1.5" />
                </g>
              </svg>

              {/* Status Stamp overlay */}
              <div className="absolute top-4 left-4 bg-[#0e1115]/80 backdrop-blur border border-[#2a3240] px-2 py-1 rounded text-[10px] text-[#94a3b8]">
                SOURCE: {data?.filename || 'clip_01.mp4'}
              </div>

              <div className="absolute bottom-4 right-4 bg-[#0e1115]/80 backdrop-blur border border-[#2a3240] px-2 py-1 rounded text-[10px] text-orange-400 font-mono">
                {currentFrame?.xai_focus || 'LANDMARK STABLE'}
              </div>
            </div>

            {/* Viewport Toolbar: Filter Selection */}
            <div className="bg-[#14181f] border-t border-[#222936] p-2 flex flex-wrap items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-[#64748b] text-[11px] mr-1">SPECTRAL LOUPE:</span>
                {(['normal', 'gradcam', 'gradcam_plus', 'lime', 'saliency'] as const).map((filter) => (
                  <button
                    key={filter}
                    onClick={() => setActiveXAI(filter)}
                    className={`px-2 py-1 rounded text-[11px] font-mono transition ${
                      activeXAI === filter 
                        ? 'bg-[#ff5500] text-black font-bold' 
                        : 'bg-[#1b212b] text-[#94a3b8] hover:bg-[#252c38] hover:text-white'
                    }`}
                  >
                    {filter.toUpperCase().replace('_', ' ')}
                  </button>
                ))}
              </div>

              <div className="text-[11px] text-[#64748b]">
                RES: 224×224 RGB
              </div>
            </div>

          </div>

          {/* Filmstrip Scrubber Deck */}
          <div className="bg-[#12161d] border border-[#222936] rounded-lg p-3 space-y-2">
            <div className="flex items-center justify-between text-xs text-[#94a3b8]">
              <span className="flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-orange-400" />
                TEMPORAL FILMSTRIP // SEQUENCE SCRUBBER
              </span>
              <span className="text-[11px] text-[#64748b]">CLICK TO AUDIT FRAME</span>
            </div>

            {/* Physical-feeling filmstrip thumbnails */}
            <div className="grid grid-cols-6 sm:grid-cols-8 gap-1.5">
              {data?.temporal_series?.slice(0, 8).map((pt, idx) => {
                const isSelected = scrubIndex === idx;
                return (
                  <button
                    key={idx}
                    onClick={() => {
                      setScrubIndex(idx);
                      if (data.keyframes[idx % data.keyframes.length]) {
                        setSelectedKeyframe(idx % data.keyframes.length);
                      }
                    }}
                    className={`p-1.5 rounded border text-left transition flex flex-col justify-between h-14 ${
                      isSelected 
                        ? 'bg-[#1e2532] border-[#ff5500] text-white shadow-[0_0_8px_rgba(255,85,0,0.2)]' 
                        : 'bg-[#101318] border-[#222834] text-[#64748b] hover:border-[#384252]'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[9px]">
                      <span>#{pt.frame}</span>
                      <div className={`w-1.5 h-1.5 rounded-full ${pt.score > 50 ? 'bg-[#ff5500]' : 'bg-[#10b981]'}`} />
                    </div>
                    <div className="text-[10px] font-bold text-slate-300">
                      {pt.score}%
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Temporal Barcode Track */}
            <div className="pt-2 border-t border-[#1c222c]">
              <div className="h-2 w-full flex gap-0.5 rounded overflow-hidden bg-[#0a0d11]">
                {data?.temporal_series?.map((pt, i) => (
                  <div
                    key={i}
                    className={`flex-1 transition-all ${
                      pt.score > 50 ? 'bg-[#ff5500]' : 'bg-[#1e293b]'
                    }`}
                    title={`Frame ${pt.frame}: ${pt.score}%`}
                  />
                ))}
              </div>
              <div className="flex justify-between text-[9px] text-[#64748b] pt-1">
                <span>0.00s</span>
                <span>CHRONOLOGICAL TEMPORAL SPECTRUM</span>
                <span>{data?.temporal_series?.slice(-1)[0]?.time || '3.00s'}</span>
              </div>
            </div>
          </div>

        </section>

        {/* Right Column: The Forensic Ledger & Instrument Deck (5 cols) */}
        <section className="lg:col-span-5 flex flex-col gap-3">
          
          {/* Stamped Verdict Plaque */}
          <div className="bg-[#12161d] border border-[#222936] rounded-lg p-5 relative overflow-hidden">
            {/* Top industrial marking */}
            <div className="flex items-center justify-between text-[11px] text-[#64748b] mb-3">
              <span>DOCUMENT REF // MAYALENS-XAI</span>
              <span className="text-orange-400 font-mono">CONFIDENTIAL REPORT</span>
            </div>

            {/* Verdict Box */}
            <div className={`border-2 p-4 rounded-md mb-4 ${
              data?.is_fake 
                ? 'border-[#ff5500]/60 bg-[#1a1312]' 
                : 'border-[#10b981]/60 bg-[#0d1714]'
            }`}>
              <div className="text-[10px] tracking-widest text-[#94a3b8] uppercase mb-1">
                SYSTEM VERDICT
              </div>
              <div className={`text-xl font-black tracking-wide ${
                data?.is_fake ? 'text-[#ff5500]' : 'text-[#10b981]'
              }`}>
                {data?.decision || 'COMPUTING...'}
              </div>
              <div className="flex items-center justify-between text-xs mt-2 pt-2 border-t border-white/10 text-slate-300">
                <span>CONFIDENCE SCORE:</span>
                <span className="font-bold text-white text-sm">{data?.confidence}%</span>
              </div>
            </div>

            {/* Concise Human Findings */}
            <p className="text-xs text-[#94a3b8] leading-relaxed mb-4">
              {data?.summary}
            </p>

            {/* Metric counters */}
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-[#181d26] p-2.5 rounded border border-[#242b38]">
                <div className="text-[10px] text-[#64748b]">INFERENCE LATENCY</div>
                <div className="text-sm font-bold text-white mt-0.5">{data?.inference_time_ms} ms</div>
              </div>
              <div className="bg-[#181d26] p-2.5 rounded border border-[#242b38]">
                <div className="text-[10px] text-[#64748b]">RISK PROFILE</div>
                <div className="text-sm font-bold text-orange-400 mt-0.5">{data?.risk_level}</div>
              </div>
            </div>
          </div>

          {/* Tri-Stream Instrument Decomposition */}
          <div className="bg-[#12161d] border border-[#222936] rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between text-xs text-[#94a3b8]">
              <span className="flex items-center gap-1.5 font-bold text-white">
                <Layers className="w-3.5 h-3.5 text-orange-400" />
                STREAM DECOMPOSITION (REPORT §5)
              </span>
              <span className="text-[10px] text-[#64748b]">3-TIER FUSION</span>
            </div>

            <div className="space-y-2.5">
              {data?.streams?.map((st) => (
                <div key={st.id} className="bg-[#181d26] border border-[#222937] p-3 rounded">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-semibold text-slate-200">{st.name}</span>
                    <span className="text-orange-400 font-bold">{st.confidence}%</span>
                  </div>

                  {/* Level meter */}
                  <div className="h-1.5 bg-[#0e1115] rounded-full overflow-hidden mb-2">
                    <div 
                      className={`h-full rounded-full ${st.confidence > 50 ? 'bg-[#ff5500]' : 'bg-[#10b981]'}`}
                      style={{ width: `${st.confidence}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[10px] text-[#64748b]">
                    <span className="truncate max-w-[240px]">{st.focus}</span>
                    <span className="text-slate-400 uppercase font-mono">{st.status}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Keyframe Evidence Dossier */}
          {currentFrame && (
            <div className="bg-[#12161d] border border-[#222936] rounded-lg p-4 space-y-2.5">
              <div className="flex items-center justify-between text-xs text-[#94a3b8]">
                <span className="font-bold text-white">KEYFRAME #{currentFrame.frame_idx} AUDIT</span>
                <span className="text-[10px] text-orange-400 font-mono">{currentFrame.crop_type}</span>
              </div>

              <div className="text-xs bg-[#181d26] p-3 rounded border border-[#242b38] space-y-2">
                <div>
                  <span className="text-[10px] text-[#64748b] block">ANOMALY DETECTED:</span>
                  <span className="text-slate-200">{currentFrame.anomaly_detected}</span>
                </div>
                <div className="border-t border-[#222834] pt-2">
                  <span className="text-[10px] text-[#64748b] block">SPECTRAL FFT/DCT:</span>
                  <span className="text-slate-300">{currentFrame.features.fft_peaks}</span>
                </div>
                <div className="border-t border-[#222834] pt-2">
                  <span className="text-[10px] text-[#64748b] block">ViT ATTENTION:</span>
                  <span className="text-slate-300">{currentFrame.features.vit_attention}</span>
                </div>
              </div>
            </div>
          )}

          {/* Model Benchmark Drawer (Collapsible) */}
          {showDossier && benchmarks && (
            <div className="bg-[#12161d] border border-orange-500/40 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between text-xs">
                <span className="font-bold text-orange-400">BENCHMARK DATA (evaluation.csv)</span>
                <span className="text-[10px] text-[#64748b]">IN-DOMAIN & CROSS-DATASET</span>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-[11px] font-mono">
                  <thead className="bg-[#181d26] text-[#64748b] border-b border-[#242b38]">
                    <tr>
                      <th className="p-1.5">Dataset</th>
                      <th className="p-1.5">Level</th>
                      <th className="p-1.5">AUC</th>
                      <th className="p-1.5">EER</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1e2430] text-slate-300">
                    {benchmarks.evaluation.slice(0, 6).map((r, i) => (
                      <tr key={i}>
                        <td className="p-1.5 font-bold text-white">{r.set}</td>
                        <td className="p-1.5">{r.level}</td>
                        <td className="p-1.5 text-orange-400">{r.AUC}</td>
                        <td className="p-1.5 text-[#64748b]">{r.EER}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </section>

      </main>

      {/* Industrial Bottom Status Strip */}
      <footer className="border-t border-[#21262d] bg-[#14181f] px-5 py-2.5 flex items-center justify-between text-[11px] text-[#64748b]">
        <div className="flex items-center gap-3">
          <span>PORT: 5173 / 8000</span>
          <span>•</span>
          <span>XAI: GRAD-CAM / LIME / SALIENCY</span>
          <span>•</span>
          <span>BACKEND: FASTAPI</span>
        </div>
        <div className="flex items-center gap-2">
          <span>MayaLens Forensic Core</span>
        </div>
      </footer>

    </div>
  );
}
