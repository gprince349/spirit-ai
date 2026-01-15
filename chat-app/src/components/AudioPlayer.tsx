"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Slider } from "@/components/ui/slider";

interface AudioPlayerProps {
  audioChunks: string[];
  onPlayStateChange?: (isPlaying: boolean) => void;
}

export function AudioPlayer({ audioChunks, onPlayStateChange }: AudioPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentChunkIndex, setCurrentChunkIndex] = useState(0);
  const [chunkProgress, setChunkProgress] = useState(0);
  const [totalDuration, setTotalDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const durationsRef = useRef<number[]>([]);
  const isPlayingRef = useRef(false);

  // Calculate total duration from chunk durations
  useEffect(() => {
    // Estimate duration: ~100ms per word, assume ~5 words per chunk on average
    // This is a rough estimate since we can't know actual duration without loading
    const estimatedDuration = audioChunks.length * 3; // ~3 seconds per chunk
    setTotalDuration(estimatedDuration);
  }, [audioChunks]);

  // Create audio URL from base64
  const createAudioUrl = useCallback((base64: string): string => {
    const audioData = atob(base64);
    const arrayBuffer = new ArrayBuffer(audioData.length);
    const view = new Uint8Array(arrayBuffer);
    for (let i = 0; i < audioData.length; i++) {
      view[i] = audioData.charCodeAt(i);
    }
    const blob = new Blob([arrayBuffer], { type: "audio/wav" });
    return URL.createObjectURL(blob);
  }, []);

  // Play a specific chunk
  const playChunk = useCallback(async (index: number) => {
    if (index >= audioChunks.length) {
      // Done playing all chunks
      setIsPlaying(false);
      isPlayingRef.current = false;
      setCurrentChunkIndex(0);
      setChunkProgress(0);
      setCurrentTime(0);
      onPlayStateChange?.(false);
      return;
    }

    const url = createAudioUrl(audioChunks[index]);
    const audio = new Audio(url);
    audioRef.current = audio;

    audio.addEventListener("loadedmetadata", () => {
      durationsRef.current[index] = audio.duration;
    });

    audio.addEventListener("timeupdate", () => {
      setChunkProgress(audio.currentTime);
      // Calculate overall time
      const previousDuration = durationsRef.current.slice(0, index).reduce((a, b) => a + b, 0);
      setCurrentTime(previousDuration + audio.currentTime);
    });

    audio.addEventListener("ended", () => {
      URL.revokeObjectURL(url);
      if (isPlayingRef.current) {
        setCurrentChunkIndex(index + 1);
        playChunk(index + 1);
      }
    });

    audio.addEventListener("error", () => {
      URL.revokeObjectURL(url);
      // Skip to next chunk on error
      if (isPlayingRef.current) {
        setCurrentChunkIndex(index + 1);
        playChunk(index + 1);
      }
    });

    try {
      await audio.play();
    } catch (e) {
      console.error("Playback failed:", e);
    }
  }, [audioChunks, createAudioUrl, onPlayStateChange]);

  // Toggle play/pause
  const togglePlay = () => {
    if (isPlaying) {
      // Pause
      if (audioRef.current) {
        audioRef.current.pause();
      }
      setIsPlaying(false);
      isPlayingRef.current = false;
      onPlayStateChange?.(false);
    } else {
      // Play
      setIsPlaying(true);
      isPlayingRef.current = true;
      onPlayStateChange?.(true);
      playChunk(currentChunkIndex);
    }
  };

  // Seek to chunk
  const handleSeek = (value: number[]) => {
    const targetTime = value[0];
    let accumulatedTime = 0;
    let targetChunk = 0;

    for (let i = 0; i < audioChunks.length; i++) {
      const chunkDuration = durationsRef.current[i] || 3; // Default 3s per chunk
      if (accumulatedTime + chunkDuration > targetTime) {
        targetChunk = i;
        break;
      }
      accumulatedTime += chunkDuration;
      targetChunk = i + 1;
    }

    // Stop current playback
    if (audioRef.current) {
      audioRef.current.pause();
    }

    setCurrentChunkIndex(Math.min(targetChunk, audioChunks.length - 1));
    setCurrentTime(targetTime);

    // Resume if was playing
    if (isPlayingRef.current) {
      playChunk(Math.min(targetChunk, audioChunks.length - 1));
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  // Format time as mm:ss
  const formatTime = (time: number): string => {
    if (!isFinite(time) || time < 0) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  if (audioChunks.length === 0) return null;

  return (
    <div className="flex items-center gap-3 mt-3 p-2 bg-zinc-800/50 rounded-lg">
      {/* Play/Pause Button */}
      <button
        onClick={togglePlay}
        className="w-8 h-8 flex items-center justify-center rounded-full bg-amber-600 hover:bg-amber-500 transition-colors"
      >
        {isPlaying ? (
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white" stroke="none">
            <rect x="6" y="4" width="4" height="16" rx="1"/>
            <rect x="14" y="4" width="4" height="16" rx="1"/>
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="white" stroke="none">
            <polygon points="6 3 20 12 6 21 6 3"/>
          </svg>
        )}
      </button>

      {/* Time - Current */}
      <span className="text-xs text-zinc-400 w-10 text-right font-mono">
        {formatTime(currentTime)}
      </span>

      {/* Progress Slider */}
      <div className="flex-1">
        <Slider
          value={[currentTime]}
          min={0}
          max={totalDuration || 1}
          step={0.1}
          onValueChange={handleSeek}
          className="cursor-pointer"
        />
      </div>

      {/* Time - Duration */}
      <span className="text-xs text-zinc-400 w-10 font-mono">
        {formatTime(totalDuration)}
      </span>

      {/* Chunk indicator */}
      <span className="text-xs text-zinc-500">
        {currentChunkIndex + 1}/{audioChunks.length}
      </span>
    </div>
  );
}
