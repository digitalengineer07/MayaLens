import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[MayaLens] Uncaught UI error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null });
    window.location.reload();
  };

  public render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#F2EBFC] text-[#171926] flex items-center justify-center p-6 font-sans">
          <div className="max-w-md w-full bg-white/95 backdrop-blur-md rounded-[32px] p-8 border border-white/80 shadow-2xl shadow-[#B8C0FF]/30 text-center space-y-5">
            <div className="w-14 h-14 mx-auto rounded-2xl bg-rose-100 text-rose-600 flex items-center justify-center">
              <AlertCircle className="w-8 h-8" />
            </div>
            
            <div className="space-y-2">
              <h2 className="text-xl font-bold text-[#141624]">Rendering Interrupted</h2>
              <p className="text-xs text-[#52567a] leading-relaxed">
                MayaLens encountered an unexpected display issue while formatting the forensic report.
              </p>
            </div>

            {this.state.error && (
              <div className="p-3 rounded-xl bg-[#f8f7fd] border border-[#B8C0FF]/30 text-[11px] font-mono text-rose-700 text-left overflow-x-auto max-h-24">
                {this.state.error.message}
              </div>
            )}

            <button
              onClick={this.handleReset}
              className="w-full py-3 px-5 rounded-full bg-[#181a2e] hover:bg-[#232742] text-white text-xs font-semibold inline-flex items-center justify-center gap-2 shadow-md transition"
            >
              <RefreshCw className="w-4 h-4 text-[#B8C0FF]" />
              <span>Reload Scanner</span>
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
