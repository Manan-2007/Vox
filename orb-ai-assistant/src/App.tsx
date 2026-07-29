import { Mic, MicOff } from 'lucide-react';
import { Background } from './components/Background';
import { Orb } from './components/Orb';
import { useLiveAPI } from './hooks/useLiveAPI';

export default function App() {
  const { state, isConnected, error, connect, disconnect, micVolume, speakerVolume } = useLiveAPI();

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center font-sans text-white overflow-hidden">
      <Background />
      
      <div className="z-10 flex flex-col items-center justify-center gap-16">
        <Orb state={state} micVolume={micVolume} speakerVolume={speakerVolume} />
        
        <div className="flex flex-col items-center gap-6">
          <button
            onClick={isConnected ? disconnect : connect}
            className={`flex items-center gap-2 px-8 py-4 rounded-full font-medium transition-all duration-300 shadow-lg ${
              isConnected 
                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30 border border-red-500/50' 
                : 'bg-white/10 text-white hover:bg-white/20 border border-white/20 backdrop-blur-md'
            }`}
          >
            {isConnected ? (
              <>
                <MicOff className="w-5 h-5" />
                Disconnect
              </>
            ) : (
              <>
                <Mic className="w-5 h-5" />
                Wake Assistant
              </>
            )}
          </button>
          
          {error && (
            <div className="text-red-400 text-sm max-w-md text-center bg-red-950/50 p-3 rounded-lg border border-red-900/50">
              {error}
            </div>
          )}
          
          {isConnected && (
            <div className="text-white/40 text-xs uppercase tracking-[0.2em] font-mono">
              System State: {state}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
