import { motion, Variants } from 'motion/react';
import { useEffect, useState } from 'react';
import { OrbState } from '../hooks/useLiveAPI';

interface OrbProps {
  state: OrbState;
  micVolume: number;
  speakerVolume: number;
}

export function Orb({ state, micVolume, speakerVolume }: OrbProps) {
  const [isBlinking, setIsBlinking] = useState(false);

  useEffect(() => {
    if (state !== 'idle' && state !== 'listening') return;
    
    let timeout: number;
    const scheduleBlink = () => {
      timeout = window.setTimeout(() => {
        setIsBlinking(true);
        setTimeout(() => setIsBlinking(false), 150);
        scheduleBlink();
      }, Math.random() * 3000 + 3000);
    };
    
    scheduleBlink();
    return () => clearTimeout(timeout);
  }, [state]);

  const containerVariants: Variants = {
    idle: {
      y: [0, -15, 0],
      scale: 1,
      rotate: 0,
      transition: { duration: 4, repeat: Infinity, ease: "easeInOut" }
    },
    listening: {
      y: 0,
      scale: 1.05 + (micVolume / 255) * 0.1,
      rotate: 0,
      transition: { type: "spring", stiffness: 100, damping: 10 }
    },
    thinking: {
      y: [0, -5, 0],
      scale: 1,
      rotate: 0,
      transition: { duration: 2, repeat: Infinity, ease: "easeInOut" }
    },
    speaking: {
      y: 0,
      scale: 1 + (speakerVolume / 255) * 0.15,
      rotate: 0,
      transition: { type: "spring", stiffness: 200, damping: 15 }
    },
    confused: {
      y: 0,
      scale: 1,
      rotate: [-5, 5, -5, 5, 0],
      transition: { duration: 0.5, ease: "easeInOut" }
    },
    happy: {
      y: [0, -20, 0],
      scale: 1.1,
      rotate: 0,
      transition: { duration: 0.6, ease: "easeOut" }
    }
  };

  const leftEyeVariants: Variants = {
    idle: { scaleY: isBlinking ? 0.1 : 1, scaleX: 1, y: 0, x: 0, rotate: 0 },
    listening: { scaleY: isBlinking ? 0.1 : 1.2, scaleX: 1.1, y: 0, x: 0, rotate: 0 },
    thinking: { scaleY: 1, scaleX: 1, y: -8, x: 8, rotate: 0 },
    speaking: { scaleY: 1 + (speakerVolume/255)*0.2, scaleX: 1, y: 0, x: 0, rotate: 0 },
    confused: { scaleY: 0.8, scaleX: 1, y: 0, x: 0, rotate: 15 },
    happy: { scaleY: 1.3, scaleX: 1.2, y: -4, x: 0, rotate: 0 },
  };

  const rightEyeVariants: Variants = {
    idle: { scaleY: isBlinking ? 0.1 : 1, scaleX: 1, y: 0, x: 0, rotate: 0 },
    listening: { scaleY: isBlinking ? 0.1 : 1.2, scaleX: 1.1, y: 0, x: 0, rotate: 0 },
    thinking: { scaleY: 1, scaleX: 1, y: -8, x: 8, rotate: 0 },
    speaking: { scaleY: 1 + (speakerVolume/255)*0.2, scaleX: 1, y: 0, x: 0, rotate: 0 },
    confused: { scaleY: 0.5, scaleX: 1, y: 0, x: 0, rotate: -10 },
    happy: { scaleY: 1.3, scaleX: 1.2, y: -4, x: 0, rotate: 0 },
  };

  const getGlow = () => {
    const baseColor = 'rgba(139, 92, 246, 0.5)'; // Purple
    const listeningColor = 'rgba(56, 189, 248, 0.6)'; // Blue
    const speakingColor = 'rgba(45, 212, 191, 0.6)'; // Teal
    const confusedColor = 'rgba(245, 158, 11, 0.5)'; // Amber
    const happyColor = 'rgba(236, 72, 153, 0.6)'; // Pink

    let color = baseColor;
    let size = 40;

    switch (state) {
      case 'listening':
        color = listeningColor;
        size = 50 + (micVolume / 255) * 40;
        break;
      case 'speaking':
        color = speakingColor;
        size = 60 + (speakerVolume / 255) * 50;
        break;
      case 'thinking':
        color = baseColor;
        size = 45;
        break;
      case 'confused':
        color = confusedColor;
        size = 30;
        break;
      case 'happy':
        color = happyColor;
        size = 80;
        break;
      default:
        size = 40;
    }

    return `0 0 ${size}px ${size/2}px ${color}, inset 0 0 ${size/2}px ${color}`;
  };

  return (
    <div className="relative flex items-center justify-center w-64 h-64">
      {/* Aura / Glow */}
      <motion.div
        className="absolute inset-0 rounded-full blur-xl"
        animate={{
          boxShadow: getGlow(),
          rotate: state === 'thinking' ? 360 : 0,
        }}
        transition={{
          boxShadow: { type: "spring", stiffness: 50, damping: 10 },
          rotate: { duration: 3, repeat: Infinity, ease: "linear" }
        }}
      />

      {/* Main Orb Body */}
      <motion.div
        variants={containerVariants}
        animate={state}
        className="relative w-48 h-48 rounded-full bg-gradient-to-br from-indigo-500/80 via-purple-600/80 to-teal-500/80 backdrop-blur-md border border-white/20 shadow-2xl overflow-hidden flex items-center justify-center gap-8"
        style={{
          background: state === 'thinking' 
            ? 'conic-gradient(from 0deg, rgba(99,102,241,0.8), rgba(168,85,247,0.8), rgba(20,184,166,0.8), rgba(99,102,241,0.8))'
            : undefined
        }}
      >
        {/* Glassy reflection */}
        <div className="absolute inset-0 rounded-full bg-gradient-to-tr from-transparent via-white/10 to-white/30 pointer-events-none" />
        <div className="absolute top-4 left-8 w-16 h-8 bg-white/20 rounded-full blur-md transform -rotate-45 pointer-events-none" />

        {/* Left Eye */}
        <motion.div
          variants={leftEyeVariants}
          animate={state}
          className="w-3 h-12 bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.8)]"
        />

        {/* Right Eye */}
        <motion.div
          variants={rightEyeVariants}
          animate={state}
          className="w-3 h-12 bg-white rounded-full shadow-[0_0_10px_rgba(255,255,255,0.8)]"
        />
      </motion.div>
    </div>
  );
}
