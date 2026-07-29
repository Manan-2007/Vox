import { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, Type } from '@google/genai';
import { AudioRecorder, AudioPlayer } from '../lib/audioUtils';

export type OrbState = 'idle' | 'listening' | 'thinking' | 'speaking' | 'confused' | 'happy';

export function useLiveAPI() {
  const [state, setState] = useState<OrbState>('idle');
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const sessionRef = useRef<any>(null);
  const recorderRef = useRef<AudioRecorder | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);
  
  const [micVolume, setMicVolume] = useState(0);
  const [speakerVolume, setSpeakerVolume] = useState(0);
  
  const stateTimeoutRef = useRef<number | null>(null);

  const setOrbState = useCallback((newState: OrbState, duration?: number) => {
    setState(newState);
    if (stateTimeoutRef.current) {
      window.clearTimeout(stateTimeoutRef.current);
      stateTimeoutRef.current = null;
    }
    if (duration) {
      stateTimeoutRef.current = window.setTimeout(() => {
        setState('idle');
      }, duration);
    }
  }, []);

  const connect = useCallback(async () => {
    try {
      setError(null);
      const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
      
      recorderRef.current = new AudioRecorder();
      playerRef.current = new AudioPlayer();
      playerRef.current.init();

      const sessionPromise = ai.live.connect({
        model: "gemini-3.1-flash-live-preview",
        callbacks: {
          onopen: () => {
            setIsConnected(true);
            setOrbState('idle');
            
            recorderRef.current?.start((base64Data) => {
              sessionPromise.then((session) => {
                session.sendRealtimeInput({
                  audio: { data: base64Data, mimeType: 'audio/pcm;rate=16000' }
                });
              });
            });
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.modelTurn) {
              setOrbState('speaking');
              const parts = message.serverContent.modelTurn.parts;
              for (const part of parts) {
                if (part.inlineData?.data) {
                  playerRef.current?.play(part.inlineData.data);
                }
              }
            }
            
            if (message.serverContent?.interrupted) {
              playerRef.current?.stop();
              playerRef.current?.init();
              setOrbState('listening');
            }
            
            if (message.serverContent?.turnComplete) {
               // Wait a bit to see if audio is still playing before going idle
               setTimeout(() => {
                 if (playerRef.current?.getVolume() === 0) {
                   setOrbState('idle');
                 }
               }, 1000);
            }

            if (message.toolCall) {
              const calls = message.toolCall.functionCalls;
              if (calls) {
                for (const call of calls) {
                  if (call.name === 'indicateConfusion') {
                    setOrbState('confused', 3000);
                    sessionPromise.then(session => {
                      session.sendToolResponse({
                        functionResponses: [{
                          id: call.id,
                          name: call.name,
                          response: { result: "Confusion indicated to user." }
                        }]
                      });
                    });
                  } else if (call.name === 'indicateHappiness') {
                    setOrbState('happy', 3000);
                    sessionPromise.then(session => {
                      session.sendToolResponse({
                        functionResponses: [{
                          id: call.id,
                          name: call.name,
                          response: { result: "Happiness indicated to user." }
                        }]
                      });
                    });
                  }
                }
              }
            }
          },
          onerror: (err) => {
            console.error("Live API Error:", err);
            setError("Connection error occurred.");
            disconnect();
          },
          onclose: () => {
            setIsConnected(false);
            setOrbState('idle');
            disconnect();
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: "Zephyr" } },
          },
          systemInstruction: "You are a helpful, calm, and intelligent AI assistant represented as a glowing orb. Keep your responses concise and conversational. If you don't understand something, or if the user's input is ambiguous, call the 'indicateConfusion' function and ask for clarification. If the user shares good news or you successfully complete a complex task, call the 'indicateHappiness' function.",
          tools: [{
            functionDeclarations: [
              {
                name: 'indicateConfusion',
                description: 'Call this function when you do not understand the user, the input is ambiguous, or you need clarification.',
                parameters: { type: Type.OBJECT, properties: {} }
              },
              {
                name: 'indicateHappiness',
                description: 'Call this function when the user shares good news, or you successfully complete a task and want to show a happy expression.',
                parameters: { type: Type.OBJECT, properties: {} }
              }
            ]
          }]
        },
      });

      sessionRef.current = await sessionPromise;

    } catch (err: any) {
      console.error("Failed to connect:", err);
      setError(err.message || "Failed to connect to AI.");
      disconnect();
    }
  }, [setOrbState]);

  const disconnect = useCallback(() => {
    if (sessionRef.current) {
      sessionRef.current.close();
      sessionRef.current = null;
    }
    if (recorderRef.current) {
      recorderRef.current.stop();
      recorderRef.current = null;
    }
    if (playerRef.current) {
      playerRef.current.stop();
      playerRef.current = null;
    }
    setIsConnected(false);
    setOrbState('idle');
  }, [setOrbState]);

  // Volume polling loop
  useEffect(() => {
    let animationFrameId: number;
    
    const updateVolumes = () => {
      let mVol = 0;
      let sVol = 0;
      
      if (recorderRef.current) {
        mVol = recorderRef.current.getVolume();
      }
      if (playerRef.current) {
        sVol = playerRef.current.getVolume();
      }
      
      setMicVolume(mVol);
      setSpeakerVolume(sVol);
      
      // Auto-state transitions based on volume
      setState((prevState) => {
        if (prevState === 'confused' || prevState === 'happy') return prevState; // Let these states time out
        
        if (sVol > 5) {
          return 'speaking';
        } else if (mVol > 15 && prevState !== 'speaking') {
          return 'listening';
        } else if (prevState === 'listening' && mVol <= 15) {
          return 'thinking'; // Briefly think after listening
        } else if (prevState === 'thinking' && sVol === 0) {
           // Stay thinking until speaking or timeout
           return prevState;
        }
        
        return prevState;
      });

      animationFrameId = requestAnimationFrame(updateVolumes);
    };
    
    if (isConnected) {
      updateVolumes();
    }
    
    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, [isConnected]);

  // Thinking timeout
  useEffect(() => {
    if (state === 'thinking') {
      const t = setTimeout(() => {
        setState(s => s === 'thinking' ? 'idle' : s);
      }, 2000);
      return () => clearTimeout(t);
    }
  }, [state]);

  return {
    state,
    isConnected,
    error,
    connect,
    disconnect,
    micVolume,
    speakerVolume
  };
}
