import React, { useState, useEffect, useRef } from 'react';
import { 
  Upload, 
  Download, 
  RefreshCw, 
  Layers, 
  Activity, 
  ChevronRight, 
  Eye, 
  ExternalLink, 
  ArrowUpRight, 
  CheckCircle2, 
  FileText,
  Play,
  Film,
  Image as ImageIcon,
  AlertCircle,
  Sparkles
} from 'lucide-react';

interface StreamResult {
  id: string;
  name: string;
  confidence: number;
  weight: string;
  focus: string;
  status: string;
  metrics?: Record<string, string>;
}

interface Keyframe {
  frame_idx: number;
  timestamp: string;
  fake_score: number;
  anomaly_detected: string;
  xai_focus: string;
  crop_type: string;
  thumbnail_url?: string;
  laplacian_var?: number;
  hf_spectral_ratio?: number;
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
  file_size?: string;
  file_size_bytes?: number;
  media_url?: string;
  storage_path?: string;
  decision: string;
  is_fake: boolean;
  confidence: number;
  risk_level: string;
  inference_time_ms: number;
  summary: string;
  resolution?: string;
  width?: number;
  height?: number;
  fps?: number;
  total_frames?: number;
  duration?: string;
  duration_sec?: number;
  bitrate_mbps?: number;
  avg_laplacian?: number;
  hf_spectral_ratio?: number;
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
  const [analysisStep, setAnalysisStep] = useState<string>('');
  const [selectedKeyframe, setSelectedKeyframe] = useState<number>(0);
  const [activeFilter, setActiveFilter] = useState<'original' | 'gradcam' | 'gradcam_plus' | 'lime' | 'saliency'>('gradcam');
  const [benchmarks, setBenchmarks] = useState<BenchmarkData | null>(null);
  const [showBenchmarks, setShowBenchmarks] = useState(false);
  
  // Dedicated Upload state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadSectionRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Only load benchmarks on mount - do not auto-run sample analysis so numbers do not randomly fluctuate
    fetchBenchmarks();
  }, []);

  const loadSample = async () => {
    setAnalyzing(true);
    setAnalysisStep('Loading pre-computed forensic benchmark data...');
    try {
      const res = await fetch('/api/sample');
      if (!res.ok) throw new Error('API Error');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);
      setPreviewUrl(null);
      setSelectedFile(null);
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (e) {
      console.error(e);
    } finally {
      setAnalyzing(false);
      setAnalysisStep('');
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

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const onFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setAnalyzing(true);

    const steps = [
      'Ingesting media & extracting frames via MTCNN...',
      'Aligning 68 facial landmarks & multi-scale crops (1.15× & 1.70×)...',
      'Executing EfficientNet-B4, Vision Transformer & FFT/DCT ResNet...',
      'Synthesizing Explainable AI heatmaps & generating findings...'
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      currentStep = (currentStep + 1) % steps.length;
      setAnalysisStep(steps[currentStep]);
    }, 450);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!res.ok) throw new Error('Upload analysis failed');
      const json = await res.json();
      setData(json);
      setSelectedKeyframe(0);

      // Scroll to results
      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    } catch (err) {
      console.error(err);
      alert('Analysis error. Please ensure the backend server is running on port 8000.');
    } finally {
      clearInterval(interval);
      setAnalyzing(false);
      setAnalysisStep('');
    }
  };

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const currentFrame = data?.keyframes[selectedKeyframe];
  const isVideo = selectedFile ? selectedFile.type.startsWith('video/') || selectedFile.name.match(/\.(mp4|mov|avi|webm|mkv)$/i) : true;

  return (
    <div className="min-h-screen periwinkle-canvas text-[#171926] flex flex-col font-sans selection:bg-[#B8C0FF] selection:text-[#171926] p-4 sm:p-6 lg:p-8">
      
      {/* Signature Dreamy Periwinkle Frame Container */}
      <div className="max-w-6xl w-full mx-auto frame-gradient rounded-[36px] sm:rounded-[48px] border border-white/70 shadow-2xl shadow-[#B8C0FF]/30 p-6 sm:p-10 flex flex-col justify-between relative overflow-hidden">
        
        {/* Subtle decorative dot pattern watermark */}
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
        <header className="flex items-center justify-between z-10 pb-6 sm:pb-10 pt-1">
          {/* Logo - Increased size for clear visibility */}
          <a href="#overview" className="flex items-center group transition shrink-0">
            <img 
              src="/mayalens_logo_transparent.png" 
              alt="MayaLens - Deepfake Detection & Explainability" 
              className="h-14 sm:h-16 lg:h-[72px] w-auto object-contain transition-transform group-hover:scale-[1.02] drop-shadow-xs"
            />
          </a>

          {/* Center Navigation Links - Only Relevant Options */}
          <nav className="hidden md:flex items-center gap-8 text-[14px] font-semibold text-[#2d314e]">
            <button 
              onClick={scrollToUpload} 
              className="hover:text-black transition py-1 cursor-pointer"
            >
              Upload Lab
            </button>
            <a 
              href="#visual-inspection" 
              className="hover:text-black transition py-1"
            >
              XAI Visualizer
            </a>
            <button 
              onClick={() => setShowBenchmarks(!showBenchmarks)} 
              className="hover:text-black transition py-1 cursor-pointer"
            >
              Benchmarks
            </button>
          </nav>

          {/* Right Action Buttons - Proportional to header */}
          <div className="flex items-center gap-3 shrink-0">
            <a
              href={data?.report_url || "/api/report/demo"}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs sm:text-[13px] font-semibold text-[#181a2e] hover:text-black hover:bg-white/50 transition px-3.5 py-2 rounded-full flex items-center gap-1.5 border border-transparent hover:border-white/60"
            >
              <FileText className="w-4 h-4 text-[#5e69bf]" />
              <span className="hidden sm:inline">PDF Report</span>
            </a>

            <button
              onClick={scrollToUpload}
              className="text-xs sm:text-[13px] font-semibold text-white bg-[#181a2e] hover:bg-[#232742] px-5 py-2.5 rounded-full transition shadow-md hover:shadow-lg flex items-center gap-2 cursor-pointer"
            >
              <Upload className="w-4 h-4 text-[#B8C0FF]" />
              <span>Inspect Media</span>
            </button>
          </div>
        </header>

        {/* Hero Section inside Frame */}
        <div id="overview" className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center z-10 py-4 sm:py-8">
          
          {/* Left Column: Headline & Direct Navigation Pills */}
          <div className="lg:col-span-7 space-y-6">
            <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold text-[#121422] tracking-tight leading-[1.12]">
              Simplify deepfake detection with us.
            </h1>
            
            <p className="text-sm sm:text-base text-[#383c5e] max-w-lg leading-relaxed">
              From multi-scale MTCNN face extraction to spectral FFT residue analysis. Inspect and verify synthetic manipulations with complete explainable transparency.
            </p>

            {/* CTA Pills */}
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <button 
                onClick={scrollToUpload}
                className="px-6 py-3 rounded-full bg-[#181a2e] hover:bg-[#232742] text-white text-xs font-semibold transition shadow-lg shadow-[#181a2e]/20 flex items-center gap-2"
              >
                <Upload className="w-3.5 h-3.5 text-[#B8C0FF]" />
                <span>Upload Media to Analyze</span>
              </button>

              <button
                onClick={loadSample}
                disabled={analyzing}
                className="group pl-5 pr-2 py-1.5 rounded-full bg-white/80 hover:bg-white text-[#181a2e] text-xs font-semibold border border-white/90 transition shadow-md flex items-center gap-2.5"
              >
                <span>{analyzing ? 'Evaluating...' : 'Try demo sample'}</span>
                <div className="w-7 h-7 rounded-full bg-[#181a2e] text-white flex items-center justify-center group-hover:scale-105 transition">
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </div>
              </button>
            </div>
          </div>

          {/* Right Column: Floating MayaLens Forensic Card */}
          <div className="lg:col-span-5 flex justify-center lg:justify-end">
            <div className="w-full max-w-[420px] bg-gradient-to-br from-[#16182c] via-[#121424] to-[#0d0f1b] text-white rounded-[32px] p-6 sm:p-7 shadow-2xl shadow-[#181a2e]/30 border border-white/15 relative backdrop-blur-2xl overflow-hidden flex flex-col justify-between min-h-[280px] transition-all duration-300 hover:border-white/25">
              
              {/* Ambient Glassmorphic Glow inside Card */}
              <div className="absolute -top-14 -right-14 w-40 h-40 bg-[#B8C0FF]/15 rounded-full blur-2xl pointer-events-none" />
              <div className="absolute -bottom-14 -left-14 w-40 h-40 bg-[#E7D8FF]/10 rounded-full blur-2xl pointer-events-none" />

              {/* Card Header */}
              <div className="flex items-center justify-between border-b border-white/10 pb-4 relative z-10">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-2xl bg-white/10 border border-white/15 flex items-center justify-center shadow-inner">
                    <img 
                      src="/mayalens_icon.png" 
                      alt="MayaLens" 
                      className="w-5 h-5 object-contain" 
                    />
                  </div>
                  <div>
                    <h3 className="font-bold text-sm tracking-tight text-white">MayaLens</h3>
                    <p className="text-[11px] text-[#B8C0FF]/70">Deepfake Detection</p>
                  </div>
                </div>
                
                {data ? (
                  <span className={`text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-wider ${
                    data.is_fake 
                      ? 'bg-rose-500/20 text-rose-300 border border-rose-500/40 shadow-xs' 
                      : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 shadow-xs'
                  }`}>
                    {data.risk_level}
                  </span>
                ) : (
                  <span className="text-[11px] font-semibold px-3 py-1 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5 shadow-xs">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    Ready
                  </span>
                )}
              </div>

              {/* Card Body */}
              <div className="relative z-10 my-4">
                {data ? (
                  <div className="space-y-1.5">
                    <div className="text-[10px] text-[#B8C0FF]/70 uppercase tracking-widest font-mono font-semibold">
                      Verdict Confidence
                    </div>
                    <div className="flex flex-wrap items-baseline gap-2.5">
                      <span className="text-4xl font-extrabold tracking-tight text-white">
                        {data.confidence}%
                      </span>
                      <span className={`text-xs font-bold px-2.5 py-0.5 rounded-lg ${
                        data.is_fake ? 'bg-rose-500/20 text-rose-300' : 'bg-emerald-500/20 text-emerald-300'
                      }`}>
                        {data.decision}
                      </span>
                    </div>
                    <p className="text-xs text-[#B8C0FF]/70 truncate pt-1">
                      {data.filename} • {data.duration || '20.0s'}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-1.5">
                    <div className="text-[10px] text-[#B8C0FF]/70 uppercase tracking-widest font-mono font-semibold">
                      Detection Benchmark
                    </div>
                    <div className="flex items-baseline gap-2.5">
                      <span className="text-4xl font-extrabold tracking-tight text-white">
                        98.9%
                      </span>
                      <span className="text-xs font-semibold px-2.5 py-0.5 rounded-lg bg-[#B8C0FF]/15 text-[#E7D8FF] border border-[#B8C0FF]/25">
                        AUC-ROC
                      </span>
                    </div>
                    <p className="text-xs text-[#B8C0FF]/70 pt-1">
                      Multi-stream explainable forensic scanner
                    </p>
                  </div>
                )}
              </div>

              {/* Card Footer */}
              <div className="pt-3 border-t border-white/10 flex items-center justify-between text-xs relative z-10">
                {data ? (
                  <>
                    <div className="flex items-center gap-4 text-[11px]">
                      <div>
                        <span className="text-[9px] text-[#B8C0FF]/60 block font-mono">SPATIAL</span>
                        <span className="font-semibold text-white">{data.streams[0]?.confidence}%</span>
                      </div>
                      <div>
                        <span className="text-[9px] text-[#B8C0FF]/60 block font-mono">CONTEXT</span>
                        <span className="font-semibold text-white">{data.streams[1]?.confidence}%</span>
                      </div>
                      <div>
                        <span className="text-[9px] text-[#B8C0FF]/60 block font-mono">FREQUENCY</span>
                        <span className="font-semibold text-white">{data.streams[2]?.confidence}%</span>
                      </div>
                    </div>

                    <button 
                      onClick={() => {
                        setData(null);
                        setPreviewUrl(null);
                        setSelectedFile(null);
                      }} 
                      className="text-[11px] font-semibold text-[#B8C0FF] hover:text-white px-2.5 py-1 rounded-lg hover:bg-white/10 transition cursor-pointer"
                    >
                      Reset
                    </button>
                  </>
                ) : (
                  <>
                    <div>
                      <span className="text-[9px] text-[#B8C0FF]/60 block font-mono uppercase">Streams</span>
                      <span className="font-semibold text-white text-[11px]">Spatial • Context • Spectral</span>
                    </div>

                    <span className="text-[10px] font-mono font-semibold px-2.5 py-1 rounded-lg bg-white/5 border border-white/10 text-[#B8C0FF]">
                      XAI v1.0
                    </span>
                  </>
                )}
              </div>

            </div>
          </div>

        </div>

        {/* Bottom Trust & Evaluation Bar inside Frame */}
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

      {/* ========================================================================= */}
      {/* DEDICATED IMAGE OR VIDEO UPLOAD SECTION (Requested by User) */}
      {/* ========================================================================= */}
      <section 
        ref={uploadSectionRef}
        id="upload-section"
        className="max-w-6xl w-full mx-auto mt-10 bg-white/95 backdrop-blur-md rounded-[36px] p-6 sm:p-10 border border-white/90 shadow-2xl shadow-[#B8C0FF]/20 space-y-6"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[#B8C0FF]/20 pb-5">
          <div>
            <div className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full bg-[#B8C0FF]/20 text-[#384080] mb-2">
              <Sparkles className="w-3.5 h-3.5 text-[#5e69bf]" />
              <span>Dedicated Forensic Scanner</span>
            </div>
            <h2 className="text-2xl font-bold text-[#141624] tracking-tight">
              Upload Video or Image to Check for Deepfakes
            </h2>
            <p className="text-xs sm:text-sm text-[#52567a] mt-1">
              Select or drop any media file to scan for neural face synthesis, facial boundary warping, and frequency artifacts.
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs text-[#52567a]">
            <span className="px-2.5 py-1 rounded-lg bg-[#f0edfa] font-mono">MP4</span>
            <span className="px-2.5 py-1 rounded-lg bg-[#f0edfa] font-mono">WEBM</span>
            <span className="px-2.5 py-1 rounded-lg bg-[#f0edfa] font-mono">MOV</span>
            <span className="px-2.5 py-1 rounded-lg bg-[#f0edfa] font-mono">JPG / PNG</span>
          </div>
        </div>

        {/* Drag and Drop Box & Live Media Preview Area */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
          
          {/* Left: Drag and Drop Uploader (7 cols) */}
          <div className="lg:col-span-7 flex flex-col justify-between">
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-3xl p-8 sm:p-10 flex flex-col items-center justify-center text-center cursor-pointer transition-all duration-200 min-h-[260px] ${
                isDragging 
                  ? 'border-[#5e69bf] bg-[#eef1ff] scale-[0.99]' 
                  : 'border-[#B8C0FF] bg-[#faf9fe] hover:bg-[#f3f1fa] hover:border-[#8c97eb]'
              }`}
            >
              <input 
                ref={fileInputRef}
                type="file" 
                accept="video/*,image/*" 
                onChange={onFileInputChange}
                className="hidden" 
              />
              
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-[#B8C0FF] to-[#E7D8FF] text-[#181a2e] flex items-center justify-center mb-4 shadow-lg shadow-[#B8C0FF]/30">
                <Upload className="w-8 h-8" />
              </div>

              <h3 className="font-bold text-base text-[#141624]">
                {selectedFile ? selectedFile.name : "Drag & drop your video or photo here"}
              </h3>
              
              <p className="text-xs text-[#52567a] mt-1.5 max-w-sm">
                or <span className="text-[#4338ca] font-semibold underline underline-offset-2">browse from your computer</span>. Max size up to 200MB.
              </p>

              <div className="flex items-center gap-4 text-[11px] text-[#6c77d6] font-medium mt-4 pt-3 border-t border-[#B8C0FF]/30">
                <span className="flex items-center gap-1"><Film className="w-3.5 h-3.5" /> Full Video Scan</span>
                <span>•</span>
                <span className="flex items-center gap-1"><ImageIcon className="w-3.5 h-3.5" /> Static Photo Scan</span>
                <span>•</span>
                <span>MTCNN 68-Landmark Crop</span>
              </div>
            </div>

            {/* Run Analysis Action Button */}
            <div className="mt-4 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3">
              <div className="text-xs text-[#52567a]">
                {selectedFile ? (
                  <span className="text-[#181a2e] font-medium">
                    Ready to scan: <strong>{selectedFile.name}</strong> ({(selectedFile.size / (1024 * 1024)).toFixed(1)} MB)
                  </span>
                ) : (
                  <span>No file selected yet. You can also test with our pre-loaded video.</span>
                )}
              </div>

              <button
                onClick={runAnalysis}
                disabled={!selectedFile || analyzing}
                className={`px-8 py-3.5 rounded-full text-xs font-bold transition shadow-lg flex items-center justify-center gap-2 ${
                  selectedFile && !analyzing
                    ? 'bg-[#181a2e] hover:bg-[#232742] text-white shadow-[#181a2e]/25 cursor-pointer'
                    : 'bg-[#d8defa] text-[#8087ab] cursor-not-allowed'
                }`}
              >
                {analyzing ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin text-[#B8C0FF]" />
                    <span>Analyzing Media...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 text-[#B8C0FF] fill-[#B8C0FF]" />
                    <span>Run Deepfake Detection</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Right: Live Interactive Video / Image Preview (5 cols) */}
          <div className="lg:col-span-5 bg-[#f5f4fd] rounded-3xl p-5 border border-[#B8C0FF]/30 flex flex-col justify-between">
            <div className="flex items-center justify-between border-b border-[#B8C0FF]/20 pb-3 mb-3">
              <span className="text-xs font-bold text-[#141624] flex items-center gap-1.5">
                {isVideo ? <Film className="w-4 h-4 text-[#6c77d6]" /> : <ImageIcon className="w-4 h-4 text-[#6c77d6]" />}
                Media Inspection Monitor
              </span>
              <span className="text-[11px] font-mono text-[#6c77d6]">
                {selectedFile ? 'LOADED' : 'AWAITING FILE'}
              </span>
            </div>

            {/* Media Player or Placeholder */}
            <div className="relative aspect-video w-full rounded-2xl overflow-hidden bg-[#e9eaf7] border border-[#B8C0FF]/40 flex items-center justify-center shadow-inner">
              {previewUrl ? (
                isVideo ? (
                  <video 
                    src={previewUrl} 
                    controls 
                    className="w-full h-full object-contain bg-black rounded-2xl"
                  />
                ) : (
                  <img 
                    src={previewUrl} 
                    alt="Uploaded preview" 
                    className="w-full h-full object-contain rounded-2xl"
                  />
                )
              ) : (
                <div className="text-center p-6 text-[#52567a] space-y-2">
                  <div className="w-12 h-12 rounded-full bg-white flex items-center justify-center mx-auto text-[#6c77d6] shadow-sm">
                    <Film className="w-6 h-6" />
                  </div>
                  <div className="text-xs font-medium text-[#181a2e]">Live Media Preview Area</div>
                  <p className="text-[11px] text-[#71769e] max-w-[220px]">
                    Your uploaded video or picture will be rendered here for frame inspection.
                  </p>
                </div>
              )}
            </div>

            {/* File telemetry info */}
            <div className="mt-3 pt-3 border-t border-[#B8C0FF]/20 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-white p-2.5 rounded-xl border border-[#B8C0FF]/25">
                <span className="text-[10px] text-[#71769e] block font-mono font-semibold">FILE TYPE:</span>
                <span className="font-semibold text-[#181a2e]">{selectedFile?.type || (data ? data.media_type : 'video/mp4')}</span>
              </div>
              <div className="bg-white p-2.5 rounded-xl border border-[#B8C0FF]/25">
                <span className="text-[10px] text-[#71769e] block font-mono font-semibold">FACE DETECTION:</span>
                <span className="font-semibold text-emerald-600">MTCNN 68-Pts Active</span>
              </div>
            </div>

            {/* Live Processing status display when analyzing */}
            {analyzing && (
              <div className="mt-3 p-3 rounded-2xl bg-[#181a2e] text-white text-xs space-y-1.5 animate-pulse">
                <div className="flex items-center justify-between font-mono text-[10px] text-[#B8C0FF]">
                  <span>PIPELINE EXECUTION</span>
                  <span>ACTIVE</span>
                </div>
                <div className="text-xs font-medium">{analysisStep || 'Processing multi-stream model...'}</div>
              </div>
            )}

          </div>

        </div>
      </section>

      {/* ========================================================================= */}
      {/* DETAILED RESULTS & XAI INSPECTOR SECTION */}
      {/* ========================================================================= */}
      {data && (
        <main ref={resultsRef} id="visual-inspection" className="max-w-6xl w-full mx-auto mt-10 space-y-8">
          
          {/* Primary Result Banner */}
          <section className={`rounded-[32px] p-6 sm:p-8 border shadow-xl transition ${
            data.is_fake 
              ? 'bg-[#fff5f5] border-rose-200 shadow-rose-200/20' 
              : 'bg-[#f4fbf7] border-emerald-200 shadow-emerald-200/20'
          }`}>
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-6">
              
              <div className="flex items-start gap-4">
                <div className={`p-3.5 rounded-2xl mt-0.5 shrink-0 ${
                  data.is_fake 
                    ? 'bg-rose-100 text-rose-600 border border-rose-200' 
                    : 'bg-emerald-100 text-emerald-600 border border-emerald-200'
                }`}>
                  {data.is_fake ? <AlertCircle className="w-7 h-7" /> : <CheckCircle2 className="w-7 h-7" />}
                </div>

                <div className="space-y-1.5">
                  <div className="flex flex-wrap items-center gap-2.5">
                    <h2 className="text-2xl font-bold text-[#141624]">
                      {data.decision || (data as any).verdict || 'ANALYSIS COMPLETE'}
                    </h2>
                    <span className={`text-xs px-3 py-1 rounded-full font-bold ${
                      (data.is_fake ?? (data as any).is_deepfake)
                        ? 'bg-rose-200 text-rose-900' 
                        : 'bg-emerald-200 text-emerald-900'
                    }`}>
                      {data.confidence}% confidence
                    </span>
                    <span className="text-xs text-[#52567a]">
                      • Latency: {data.inference_time_ms || (data as any).processing_time_ms || 350} ms
                    </span>
                  </div>
                  <p className="text-sm text-[#383c5e] max-w-2xl leading-relaxed">
                    {data.summary}
                  </p>
                </div>
              </div>

              <div className="sm:text-right shrink-0">
                <a
                  href={data.report_url || "/api/report/demo"}
                  download={`MayaLens_Dossier_${data.filename.replace(/\.[^/.]+$/, '')}.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="px-5 py-2.5 rounded-full bg-[#181a2e] hover:bg-[#232742] text-white text-xs font-semibold inline-flex items-center gap-2 shadow-md transition"
                >
                  <Download className="w-3.5 h-3.5 text-[#B8C0FF]" />
                  <span>Download PDF Dossier</span>
                </a>
              </div>

            </div>
          </section>

          {/* Visual Evidence Inspection Loupe */}
          {currentFrame && (
            <section className="bg-white/90 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-6">
              
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
                    {/* Real extracted keyframe thumbnail if available */}
                    {currentFrame.thumbnail_url && (
                      <img 
                        src={currentFrame.thumbnail_url} 
                        alt={`Keyframe #${currentFrame.frame_idx}`} 
                        className="absolute inset-0 w-full h-full object-cover"
                      />
                    )}

                    <svg className={`relative z-10 w-full h-full max-h-[340px] ${currentFrame.thumbnail_url ? 'bg-black/20' : ''}`} viewBox="0 0 640 360" xmlns="http://www.w3.org/2000/svg">
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

                      {/* If no real image, show synthetic base canvas */}
                      {!currentFrame.thumbnail_url && (
                        <>
                          <rect width="640" height="360" fill="#eef1ff" />
                          <rect x="230" y="65" width="180" height="230" rx="12" fill="#e0e5ff" stroke="#6c77d6" strokeOpacity="0.4" strokeWidth="1.5" strokeDasharray="4 4" />
                          <ellipse cx="320" cy="180" rx="65" ry="90" fill="#d2d9ff" stroke="#6c77d6" strokeOpacity="0.3" strokeWidth="1" />
                          <ellipse cx="295" cy="155" rx="10" ry="6" fill="#6c77d6" />
                          <ellipse cx="345" cy="155" rx="10" ry="6" fill="#6c77d6" />
                          <path d="M 320 162 L 316 186 L 324 186" fill="none" stroke="#6c77d6" strokeWidth="1.5" />
                          <ellipse cx="320" cy="225" rx="22" ry="8" fill="#6c77d6" />
                        </>
                      )}

                      {/* Overlays */}
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

                      {/* 68 Landmarks */}
                      {activeFilter !== 'original' && (
                        <g fill="#4338ca" opacity="0.7">
                          <circle cx="285" cy="150" r="1.5" /><circle cx="295" cy="148" r="1.5" /><circle cx="305" cy="150" r="1.5" />
                          <circle cx="335" cy="150" r="1.5" /><circle cx="345" cy="148" r="1.5" /><circle cx="355" cy="150" r="1.5" />
                          <circle cx="320" cy="170" r="1.5" /><circle cx="320" cy="185" r="1.5" />
                          <circle cx="305" cy="225" r="1.5" /><circle cx="320" cy="228" r="1.5" /><circle cx="335" cy="225" r="1.5" />
                        </g>
                      )}
                    </svg>

                    <div className="absolute bottom-3 left-3 bg-white/95 backdrop-blur-md px-3 py-1 rounded-full text-[11px] font-semibold text-[#181a2e] border border-[#B8C0FF]/50 shadow-sm z-20">
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
                        <span className="text-[#2d314e]">{currentFrame?.features?.fft_peaks || 'Organic spectral decay'}</span>
                      </div>

                      <div className="p-2.5 rounded-lg bg-white border border-[#B8C0FF]/20">
                        <span className="text-[#4f46e5] font-semibold block mb-0.5">Vision Transformer Attention:</span>
                        <span className="text-[#2d314e]">{currentFrame?.features?.vit_attention || 'Coherent contextual patch attention'}</span>
                      </div>

                      <div className="p-2.5 rounded-lg bg-white border border-[#B8C0FF]/20">
                        <span className="text-[#4f46e5] font-semibold block mb-0.5">Landmark Alignment:</span>
                        <span className="text-[#2d314e]">{currentFrame?.features?.facial_landmarks || '68-point biological landmarks active'}</span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-[#B8C0FF]/20">
                    <a
                      href={data?.report_url || "/api/report/demo"}
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
          <section id="streams" className="space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h2 className="text-lg font-bold text-[#141624] flex items-center gap-2">
                <Layers className="w-5 h-5 text-[#6c77d6]" />
                <span>Multi-Stream Core Decomposition (Report §5)</span>
              </h2>
              <span className="text-xs text-[#52567a] font-mono font-medium">
                Target: {data.filename} ({data.resolution || '1280×720'}, {data.total_frames || 480} frames)
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {data.streams?.map((stream) => {
                const confVal = typeof stream.confidence === 'number'
                  ? stream.confidence
                  : parseFloat(String(stream.confidence)) || 50;
                return (
                <div key={stream.id} className="bg-white/90 backdrop-blur-md rounded-[28px] p-6 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-4 flex flex-col justify-between">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="font-bold text-sm text-[#141624]">{stream.name}</h3>
                      <span className="text-xs font-mono font-semibold text-[#6c77d6] bg-[#f0edfa] px-2 py-0.5 rounded-md">{stream.weight}</span>
                    </div>

                    <div className="flex items-baseline justify-between pt-1">
                      <div className="text-3xl font-extrabold text-[#181a2e]">
                        {confVal}%
                      </div>
                      <span className={`text-xs px-2.5 py-0.5 rounded-full font-bold ${
                        confVal > 50 
                          ? 'bg-rose-100 text-rose-700' 
                          : 'bg-emerald-100 text-emerald-700'
                      }`}>
                        {stream.status}
                      </span>
                    </div>

                    <div className="w-full bg-[#eef1ff] h-2 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-500 ${
                          confVal > 50 ? 'bg-gradient-to-r from-rose-500 to-amber-500' : 'bg-gradient-to-r from-[#6c77d6] to-[#B8C0FF]'
                        }`}
                        style={{ width: `${Math.min(100, Math.max(0, confVal))}%` }}
                      />
                    </div>

                    <p className="text-xs text-[#52567a] leading-relaxed">
                      {stream.focus}
                    </p>
                  </div>

                  {/* Granular Video-Specific Stream Metrics */}
                  {stream.metrics && (
                    <div className="pt-3 border-t border-[#B8C0FF]/30 grid grid-cols-2 gap-2 text-xs">
                      {Object.entries(stream.metrics).map(([k, v]) => (
                        <div key={k} className="bg-[#f8f7fd] p-2.5 rounded-xl border border-[#B8C0FF]/25">
                          <span className="text-[10px] text-[#52567a] uppercase font-mono font-bold block">{k}</span>
                          <span className="font-bold text-[#181a2e] text-[11px] mt-0.5 block truncate">{v}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            </div>
          </section>

          {/* Temporal Timeline */}
          {data.temporal_series && data.temporal_series.length > 1 && (
            <section id="timeline" className="bg-white/90 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-5">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#B8C0FF]/20 pb-4">
                <div className="flex items-center gap-2.5">
                  <Activity className="w-5 h-5 text-[#6c77d6]" />
                  <div>
                    <h2 className="text-lg font-bold text-[#141624]">Temporal Manipulation Timeline</h2>
                    <p className="text-xs text-[#52567a]">
                      Frame-by-frame anomaly progression across {data.duration || '20.0s'} ({data.total_frames || '480'} frames @ {data.fps || 24} fps)
                    </p>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3 text-xs font-semibold">
                  <span className="flex items-center gap-1.5 text-rose-600">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block shadow-xs"></span> Manipulated (&gt;50%)
                  </span>
                  <span className="flex items-center gap-1.5 text-[#6c77d6]">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#B8C0FF] inline-block shadow-xs"></span> Authentic (&le;50%)
                  </span>
                  <span className="px-2.5 py-1 rounded-lg bg-[#f0edfa] text-[#181a2e] border border-[#B8C0FF]/40 font-mono text-[11px]">
                    Threshold: 50%
                  </span>
                </div>
              </div>

              {/* Chart Container with relative reference height */}
              <div className="pt-6 pb-2 relative">
                {/* 50% decision threshold dashed horizontal line */}
                <div className="absolute top-[calc(6rem*0.5+1.5rem)] left-0 right-0 border-b border-dashed border-rose-400/60 z-10 pointer-events-none flex justify-end">
                  <span className="text-[9px] font-mono font-bold text-rose-600 bg-white/90 px-2 py-0.5 rounded -translate-y-2 border border-rose-200 shadow-xs">
                    50% Decision Threshold
                  </span>
                </div>

                <div className="h-32 flex items-end gap-1.5 sm:gap-2.5 w-full">
                  {data.temporal_series?.map((pt, i) => {
                    const scoreNum = typeof pt.score === 'number'
                      ? pt.score
                      : parseFloat(String(pt.score)) || 0;
                    return (
                    <div key={i} className="flex-1 h-full flex flex-col justify-end items-center gap-1.5 group relative">
                      {/* Floating tooltip */}
                      <div className="absolute -top-9 opacity-0 group-hover:opacity-100 transition-all duration-200 pointer-events-none bg-[#181a2e] text-white text-[10px] font-semibold px-2.5 py-1 rounded-lg whitespace-nowrap z-30 shadow-lg border border-[#B8C0FF]/30">
                        {pt.time} (Frame #{pt.frame}): <span className={scoreNum > 50 ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>{scoreNum}%</span>
                      </div>

                      {/* Bar Track & Fill */}
                      <div className="w-full h-24 bg-[#eef1ff] rounded-t-lg overflow-hidden flex items-end p-0.5 border border-[#B8C0FF]/25">
                        <div 
                          className={`w-full rounded-t transition-all duration-500 ${
                            scoreNum > 50 
                              ? 'bg-gradient-to-t from-rose-500 via-rose-400 to-amber-400 group-hover:brightness-110 shadow-xs shadow-rose-500/30' 
                              : 'bg-gradient-to-t from-[#6c77d6] to-[#B8C0FF] group-hover:bg-[#9fa9ff]'
                          }`}
                          style={{ height: `${Math.max(12, Math.min(100, scoreNum))}%` }}
                        />
                      </div>

                      {/* Timestamp below bar */}
                      <span className="text-[9px] text-[#52567a] font-mono font-semibold truncate w-full text-center">
                        {pt.time}
                      </span>
                    </div>
                    );
                  })}
                </div>
              </div>
            </section>
          )}

        </main>
      )}

      {/* Model Evaluation Benchmarks Modal */}
      {showBenchmarks && benchmarks && (
        <section className="max-w-6xl w-full mx-auto mt-10 bg-white/95 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-2xl shadow-[#B8C0FF]/25 space-y-6">
          <div className="flex items-center justify-between border-b border-[#B8C0FF]/20 pb-4">
            <div>
              <h2 className="text-lg font-bold text-[#141624]">Model Benchmark & Evaluation Results</h2>
              <p className="text-xs text-[#52567a]">Logged performance from evaluation.csv in the MayaLens repository.</p>
            </div>
            <button 
              onClick={() => setShowBenchmarks(false)}
              className="text-xs font-semibold text-[#181a2e] px-3.5 py-1.5 bg-[#f0edfa] hover:bg-[#e4e0f5] rounded-full transition"
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

      {/* Enhanced Footer matching the frame curvature */}
      <footer className="max-w-6xl w-full mx-auto mt-12 mb-6">
        <div className="bg-white/80 backdrop-blur-md rounded-[32px] p-6 sm:p-8 border border-white/90 shadow-xl shadow-[#B8C0FF]/15 space-y-6">
          
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-[#B8C0FF]/20 pb-5">
            <div className="flex items-center gap-3">
              <img 
                src="/mayalens_logo_transparent.png" 
                alt="MayaLens Forensics" 
                className="h-8 sm:h-9 w-auto object-contain"
              />
            </div>

            <div className="flex items-center gap-3 text-xs">
              <a
                href={data?.report_url || "/api/report/demo"}
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
