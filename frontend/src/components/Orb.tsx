/**
 * Vox orb — the assistant face that reacts while the app listens, thinks, and
 * speaks. Ported from orb-ai-assistant/ (its Gemini hook and Tailwind styling
 * removed) and recolored to the Vox palette.
 *
 * States map to real app activity:
 *   idle       breathing float + occasional blink
 *   listening  mic is capturing speech / user is typing a phrase
 *   thinking   an ISL clip sequence is playing
 *   speaking   TTS is reading a sentence aloud
 *   happy      a sign was just recognized (brief hop)
 *   confused   backend/mic error (brief wobble)
 *
 * Neither SpeechSynthesis nor SpeechRecognition exposes an output/input level,
 * so `energy` is synthesized: a requestAnimationFrame sine-with-jitter runs
 * while the orb is in an active state, standing in for the volume the original
 * component read from the Gemini audio stream.
 */
import { motion, type Variants } from "motion/react";
import { useEffect, useRef, useState } from "react";

export type OrbState =
  | "idle"
  | "listening"
  | "thinking"
  | "speaking"
  | "happy"
  | "confused";

interface OrbProps {
  state: OrbState;
  /** Diameter of the orb body in px (glow extends beyond). */
  size?: number;
}

/** Palette glows per state (Vox colours, not the original purples). */
const GLOW: Record<OrbState, string> = {
  idle: "rgba(120, 155, 123, 0.45)",       // primary
  listening: "rgba(214, 174, 130, 0.60)",  // accent
  thinking: "rgba(168, 185, 157, 0.55)",   // secondary
  speaking: "rgba(91, 154, 105, 0.60)",    // success
  happy: "rgba(214, 174, 130, 0.70)",      // accent, wide
  confused: "rgba(197, 139, 58, 0.55)",    // warning
};

export function Orb({ state, size = 150 }: OrbProps) {
  const [isBlinking, setIsBlinking] = useState(false);
  const [energy, setEnergy] = useState(0);
  const raf = useRef(0);

  /* Blink only when calm, like the original. */
  useEffect(() => {
    if (state !== "idle" && state !== "listening") return;
    let timeout: number;
    const scheduleBlink = () => {
      timeout = window.setTimeout(() => {
        setIsBlinking(true);
        window.setTimeout(() => setIsBlinking(false), 150);
        scheduleBlink();
      }, Math.random() * 3000 + 3000);
    };
    scheduleBlink();
    return () => window.clearTimeout(timeout);
  }, [state]);

  /* Synthesized activity level while listening / speaking. */
  useEffect(() => {
    const active = state === "speaking" || state === "listening";
    if (!active) {
      cancelAnimationFrame(raf.current);
      setEnergy(0);
      return;
    }
    const tick = () => {
      const t = performance.now() / 1000;
      const wave =
        0.5 +
        0.3 * Math.sin(t * 5.1) +
        0.2 * Math.sin(t * 13.7 + 1.3) * Math.random();
      setEnergy(Math.max(0, Math.min(1, wave)));
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [state]);

  const containerVariants: Variants = {
    idle: {
      y: [0, -8, 0],
      scale: 1,
      rotate: 0,
      rotateY: 0,
      transition: { duration: 4, repeat: Infinity, ease: "easeInOut" },
    },
    listening: {
      y: 0,
      scale: 1.04 + energy * 0.08,
      rotate: 0,
      rotateY: 0,
      transition: { type: "spring", stiffness: 100, damping: 10 },
    },
    thinking: {
      y: [0, -4, 0],
      scale: 1,
      rotate: 0,
      rotateY: [0, 14, -14, 0], // gentle 3D sway while clips play
      transition: { duration: 2.4, repeat: Infinity, ease: "easeInOut" },
    },
    speaking: {
      y: 0,
      scale: 1 + energy * 0.13,
      rotate: 0,
      rotateY: energy * 10 - 5,
      transition: { type: "spring", stiffness: 200, damping: 15 },
    },
    happy: {
      y: [0, -14, 0],
      scale: 1.1,
      rotate: 0,
      rotateY: 0,
      transition: { duration: 0.6, ease: "easeOut" },
    },
    confused: {
      y: 0,
      scale: 1,
      rotate: [-5, 5, -5, 5, 0],
      rotateY: 0,
      transition: { duration: 0.5, ease: "easeInOut" },
    },
  };

  const eyeVariants = (side: "left" | "right"): Variants => ({
    idle: { scaleY: isBlinking ? 0.1 : 1, scaleX: 1, y: 0, rotate: 0 },
    listening: { scaleY: isBlinking ? 0.1 : 1.2, scaleX: 1.1, y: 0, rotate: 0 },
    thinking: { scaleY: 1, scaleX: 1, y: -5, rotate: 0 },
    speaking: { scaleY: 1 + energy * 0.25, scaleX: 1, y: 0, rotate: 0 },
    happy: { scaleY: 1.3, scaleX: 1.2, y: -3, rotate: 0 },
    confused: {
      scaleY: side === "left" ? 0.8 : 0.5,
      scaleX: 1,
      y: 0,
      rotate: side === "left" ? 15 : -10,
    },
  });

  const glowSize =
    state === "happy" ? 46 :
    state === "speaking" ? 34 + energy * 26 :
    state === "listening" ? 28 + energy * 22 :
    state === "thinking" ? 26 : 22;
  const glow = GLOW[state];

  return (
    <div
      className="orb"
      style={{ width: size, height: size, perspective: 600 }}
      aria-hidden
    >
      <motion.div
        className="orb__aura"
        animate={{
          boxShadow: `0 0 ${glowSize}px ${glowSize / 2}px ${glow}`,
          rotate: state === "thinking" ? 360 : 0,
        }}
        transition={{
          boxShadow: { type: "spring", stiffness: 50, damping: 10 },
          rotate: { duration: 3, repeat: Infinity, ease: "linear" },
        }}
      />

      <motion.div
        className="orb__body"
        variants={containerVariants}
        animate={state}
        style={{
          background:
            state === "thinking"
              ? "conic-gradient(from 0deg, rgba(120,155,123,0.9), rgba(168,185,157,0.9), rgba(214,174,130,0.85), rgba(120,155,123,0.9))"
              : undefined,
        }}
      >
        <div className="orb__sheen" />
        <div className="orb__highlight" />
        <motion.div className="orb__eye" variants={eyeVariants("left")} animate={state} />
        <motion.div className="orb__eye" variants={eyeVariants("right")} animate={state} />
      </motion.div>
    </div>
  );
}
