/**
 * WebSocket link to the inference backend.
 *
 * Sends `{"landmarks": [...126 raw floats]}` frames and
 * `{"config": {...}}` settings; receives one of:
 *   {"word": "hello", "confidence": 0.97}                    a confirmed word
 *   {"status":"listening", "top", "confidence", "stable_for"} live best guess
 *   {"status":"listening", "buffered", "needed"}             window still filling
 *   {"status":"config", "confidence_threshold"}               settings ack
 *   {"error": "..."}                                          bad frame / no model
 *
 * Normalization happens server-side in ml/normalize.py — the single shared
 * implementation. Never normalize here.
 */
import { useCallback, useEffect, useRef, useState } from "react";

const WS_URL =
  (import.meta.env.VITE_VOX_WS as string | undefined) ?? "ws://localhost:8000/ws";
const RECONNECT_MS = 2000;

export type SocketState = "connecting" | "open" | "closed";

export interface LiveGuess {
  top: string;
  confidence: number;
  stableFor: number;
}

export interface ConfirmedWord {
  word: string;
  confidence: number;
}

export interface VoxSocket {
  socket: SocketState;
  live: LiveGuess | null;
  buffered: { have: number; need: number } | null;
  error: string | null;
  /** Backend is up but has no trained model — recognition disabled. */
  noModel: boolean;
  send: (vector: Float32Array) => void;
  /** Set the backend's per-connection confidence threshold (re-sent on reconnect). */
  setThreshold: (value: number) => void;
}

/** `onWord` fires once per confirmed word (the backend already de-duplicates). */
export function useVoxSocket(onWord: (word: ConfirmedWord) => void): VoxSocket {
  const onWordRef = useRef(onWord);
  onWordRef.current = onWord;

  const wsRef = useRef<WebSocket | null>(null);
  const thresholdRef = useRef<number | null>(null);
  const [socket, setSocket] = useState<SocketState>("connecting");
  const [live, setLive] = useState<LiveGuess | null>(null);
  const [buffered, setBuffered] = useState<{ have: number; need: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [noModel, setNoModel] = useState(false);

  useEffect(() => {
    let disposed = false;
    let reconnectTimer: number | undefined;

    const connect = () => {
      if (disposed) return;
      setSocket("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) return;
        setSocket("open");
        setError(null);
        setNoModel(false);
        // A non-default threshold survives reconnects.
        if (thresholdRef.current !== null) {
          ws.send(
            JSON.stringify({
              config: { confidence_threshold: thresholdRef.current },
            }),
          );
        }
      };

      ws.onclose = () => {
        if (disposed) return;
        setSocket("closed");
        setLive(null);
        setBuffered(null);
        // The backend is often started after the page; retry quietly.
        reconnectTimer = window.setTimeout(connect, RECONNECT_MS);
      };

      ws.onmessage = (event) => {
        if (disposed) return;
        const msg = JSON.parse(event.data as string);
        if (msg.word) {
          onWordRef.current({ word: msg.word, confidence: msg.confidence });
          setBuffered(null);
        } else if (msg.error) {
          setError(msg.error);
        } else if (msg.status === "config") {
          // ack only — nothing to render
        } else if (msg.status === "no-model") {
          setNoModel(true);
        } else if (msg.top !== undefined) {
          setLive({
            top: msg.top,
            confidence: msg.confidence,
            stableFor: msg.stable_for,
          });
          setBuffered(null);
        } else if (msg.buffered !== undefined) {
          setBuffered({ have: msg.buffered, need: msg.needed });
          setLive(null);
        }
      };
    };

    connect();

    return () => {
      disposed = true;
      window.clearTimeout(reconnectTimer);
      const ws = wsRef.current;
      if (ws) {
        ws.onclose = null; // don't schedule a reconnect on our own teardown
        ws.close();
      }
      wsRef.current = null;
    };
  }, []);

  const send = useCallback((vector: Float32Array) => {
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ landmarks: Array.from(vector) }));
    }
  }, []);

  const setThreshold = useCallback((value: number) => {
    thresholdRef.current = value;
    const ws = wsRef.current;
    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ config: { confidence_threshold: value } }));
    }
  }, []);

  return { socket, live, buffered, error, noModel, send, setThreshold };
}
