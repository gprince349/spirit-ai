'use client';

import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { SpeechService } from '@/services/speechService';

export default function Home() {
  const [isListening, setIsListening] = useState(true);
  const [transcript, setTranscript] = useState('');
  const [finalTranscript, setFinalTranscript] = useState('');
  const speechServiceRef = useRef<SpeechService | null>(null);
  const muteSoundRef = useRef<HTMLAudioElement | null>(null);
  const unmuteSoundRef = useRef<HTMLAudioElement | null>(null);

  // Initialize audio elements
  useEffect(() => {
    muteSoundRef.current = new Audio('/assets/rt-pause.wav');
    unmuteSoundRef.current = new Audio('/assets/rt-ready.wav');
    
    // Preload audio files
    muteSoundRef.current.load();
    unmuteSoundRef.current.load();
    
    // Cleanup function
    return () => {
      if (muteSoundRef.current) {
        muteSoundRef.current.pause();
        muteSoundRef.current = null;
      }
      if (unmuteSoundRef.current) {
        unmuteSoundRef.current.pause();
        unmuteSoundRef.current = null;
      }
    };
  }, []);

  // Auto-start microphone when component mounts
  useEffect(() => {
    // Small delay to ensure component is fully mounted
    const timer = setTimeout(() => {
      startListening();
    }, 100);
    
    return () => clearTimeout(timer);
  }, []); // Empty dependency array means this runs once on mount

  const handleCancel = () => {
    console.log('Cancel button clicked');
    if (isListening) {
      // Play mute sound when canceling
      if (muteSoundRef.current) {
        muteSoundRef.current.currentTime = 0; // Reset to start
        muteSoundRef.current.play().catch(err => console.log('Audio play failed:', err));
      }
      stopListening();
    }
  };

  const startListening = useCallback(async () => {
    try {
      // Get API key from environment
      const apiKey = process.env.NEXT_PUBLIC_SONIOX_API_KEY;
      if (!apiKey) {
        console.error('Missing SONIOX_API_KEY. Please set NEXT_PUBLIC_SONIOX_API_KEY in your .env.local file');
        return;
      }

      // Create speech service with default config
      const config = SpeechService.getDefaultConfig(apiKey);
      speechServiceRef.current = new SpeechService(
        config,
        // Transcript callback
        (final, live) => {
          setFinalTranscript(final);
          setTranscript(live);
        },
        // Connection callback
        (connected) => {
          // Connection status is handled internally by the service
        },
        // Error callback
        (error) => {
          console.error('Speech service error:', error);
        }
      );

      // Start the service
      await speechServiceRef.current.startListening();
      setIsListening(true);
      
      // Play unmute sound for auto-start
      if (unmuteSoundRef.current) {
        unmuteSoundRef.current.currentTime = 0; // Reset to start
        unmuteSoundRef.current.play().catch(err => console.log('Audio play failed:', err));
      }
      
      console.log('Microphone started - listening...');
      
    } catch (error) {
      console.error('Error starting microphone:', error);
      setIsListening(false);
    }
  }, []);

  const stopListening = useCallback(() => {
    console.log('Stopping microphone...');
    
    if (speechServiceRef.current) {
      speechServiceRef.current.stopListening();
    }
    
    setIsListening(false);
    setTranscript('');
    setFinalTranscript('');
    console.log('Microphone stopped');
  }, []);

  const handleMicrophone = () => {
    if (isListening) {
      // Play mute sound
      if (muteSoundRef.current) {
        muteSoundRef.current.currentTime = 0; // Reset to start
        muteSoundRef.current.play().catch(err => console.log('Audio play failed:', err));
      }
      stopListening();
    } else {
      // Play unmute sound
      if (unmuteSoundRef.current) {
        unmuteSoundRef.current.currentTime = 0; // Reset to start
        unmuteSoundRef.current.play().catch(err => console.log('Audio play failed:', err));
      }
      startListening();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800 flex flex-col items-center justify-center p-4">
      {/* Dotted Circle Animation */}
      <div className="relative w-32 h-32 mb-8">
        <div className="absolute inset-0 rounded-full border-2 border-slate-300 dark:border-slate-600"></div>
        <div className="absolute inset-2 rounded-full bg-slate-200/50 dark:bg-slate-700/50"></div>
        {/* Dots */}
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1.5 h-1.5 bg-slate-400 dark:bg-slate-500 rounded-full animate-pulse"
            style={{
              left: `${50 + 40 * Math.cos(i * 0.314)}%`,
              top: `${50 + 40 * Math.sin(i * 0.314)}%`,
              animationDelay: `${i * 0.1}s`,
            }}
          />
        ))}
      </div>

      {/* Text Prompt & Transcription - Floating above buttons */}
      <div className="absolute bottom-36 left-1/2 transform -translate-x-1/2 text-center w-full max-w-2xl px-4">
        {(finalTranscript || transcript) ? (
          <div>
            <p className="text-xl font-medium text-[#21808d] leading-relaxed">
              {finalTranscript}
              <span className="italic">
                {transcript}
              </span>
            </p>
          </div>
        ) : (
          <p className="text-xl font-medium" style={{ color: !isListening ? undefined : '#21808d' }}>
            {!isListening ? 'Muted' : 'Say something...'}
          </p>
        )}
      </div>

      {/* Action Buttons */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 flex gap-6">
        {/* Cancel Button */}
        <Button
          onClick={handleCancel}
          variant="outline"
          size="icon"
          className="w-16 h-16 rounded-full"
        >
          <span className="text-2xl font-bold">×</span>
        </Button>

        {/* Microphone Button */}
        <Button
          onClick={handleMicrophone}
          variant={!isListening ? "default" : "outline"}
          size="icon"
          className="w-16 h-16 rounded-full"
        >
          <svg
            className="w-7 h-7"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            {!isListening ? (
              <>
                {/* Muted microphone icon with cut-through line */}
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
                />
                {/* Diagonal cut-through line */}
                <line
                  x1="5"
                  y1="19"
                  x2="19"
                  y2="5"
                  stroke="currentColor"
                  strokeWidth={3}
                  strokeLinecap="round"
                />
              </>
            ) : (
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            )}
          </svg>
        </Button>
      </div>
    </div>
  );
}
