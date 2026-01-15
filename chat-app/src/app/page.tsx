"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AudioPlayer } from "@/components/AudioPlayer";

interface Message {
  id: string;
  type: "user" | "assistant";
  content: string;
  audioChunks?: string[];
  isStreaming?: boolean;
  autoPlayComplete?: boolean; // True when initial auto-play is finished
}

interface SentenceResult {
  index: number;
  caption: string;
  audio: string;
  format: string;
  duration_ms: number;
  error?: string;
}

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8001/conversation";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [playingMessageId, setPlayingMessageId] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const currentAssistantIdRef = useRef<string | null>(null);
  
  // Audio player state (for streaming playback)
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioQueueRef = useRef<string[]>([]);
  const isAutoPlayingRef = useRef(false);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setIsConnected(true);
      console.log("Connected to orchestrator");
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("Disconnected from orchestrator");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleServerMessage(data);
    };

    wsRef.current = ws;
  }, []);

  // Handle messages from server
  const handleServerMessage = (data: any) => {
    if (data.type === "sentence") {
      const result = data as SentenceResult;
      
      setMessages((prev) => {
        return prev.map((msg) => {
          if (msg.type === "assistant" && msg.id === currentAssistantIdRef.current) {
            const newAudioChunks = [...(msg.audioChunks || [])];
            if (result.audio && !result.error) {
              newAudioChunks.push(result.audio);
            }
            return {
              ...msg,
              content: msg.content + (msg.content ? " " : "") + result.caption,
              audioChunks: newAudioChunks,
            };
          }
          return msg;
        });
      });

      // Auto-play during streaming
      if (result.audio && !result.error && isAutoPlayingRef.current) {
        audioQueueRef.current.push(result.audio);
        playNextFromQueue();
      }
    } else if (data.type === "complete") {
      setIsLoading(false);
      isAutoPlayingRef.current = false;
      setMessages((prev) => {
        return prev.map((msg) => {
          if (msg.type === "assistant" && msg.id === currentAssistantIdRef.current) {
            return { ...msg, isStreaming: false };
          }
          return msg;
        });
      });
    } else if (data.type === "error") {
      setIsLoading(false);
      isAutoPlayingRef.current = false;
      console.error("Server error:", data.message);
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          type: "assistant",
          content: `Error: ${data.message}`,
        },
      ]);
    }
  };

  // Play next audio from queue (for streaming)
  const playNextFromQueue = async () => {
    if (currentAudioRef.current || audioQueueRef.current.length === 0) {
      // Queue is empty - check if auto-play is done
      if (audioQueueRef.current.length === 0 && !isAutoPlayingRef.current && currentAssistantIdRef.current) {
        // Mark auto-play as complete
        const assistantId = currentAssistantIdRef.current;
        setMessages((prev) => {
          return prev.map((msg) => {
            if (msg.id === assistantId) {
              return { ...msg, autoPlayComplete: true };
            }
            return msg;
          });
        });
        setPlayingMessageId(null);
      }
      return;
    }

    const audioBase64 = audioQueueRef.current.shift()!;
    await playAudioChunk(audioBase64, () => {
      playNextFromQueue();
    });
  };

  // Play a single audio chunk
  const playAudioChunk = (audioBase64: string, onEnd?: () => void): Promise<void> => {
    return new Promise((resolve) => {
      try {
        const audioData = atob(audioBase64);
        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
          view[i] = audioData.charCodeAt(i);
        }

        const blob = new Blob([arrayBuffer], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        currentAudioRef.current = audio;

        audio.onended = () => {
          URL.revokeObjectURL(url);
          currentAudioRef.current = null;
          onEnd?.();
          resolve();
        };

        audio.onerror = () => {
          URL.revokeObjectURL(url);
          currentAudioRef.current = null;
          onEnd?.();
          resolve();
        };

        audio.play();
      } catch (error) {
        console.error("Audio playback error:", error);
        currentAudioRef.current = null;
        onEnd?.();
        resolve();
      }
    });
  };

  // Stop current streaming playback
  const stopPlayback = () => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause();
      currentAudioRef.current = null;
    }
    audioQueueRef.current = [];
    setPlayingMessageId(null);
  };

  // Send query
  const sendQuery = () => {
    if (!input.trim() || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Stop any current playback
    stopPlayback();

    const userMessage: Message = {
      id: Date.now().toString(),
      type: "user",
      content: input,
    };

    const assistantId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantId,
      type: "assistant",
      content: "",
      audioChunks: [],
      isStreaming: true,
    };

    currentAssistantIdRef.current = assistantId;
    isAutoPlayingRef.current = true;
    setPlayingMessageId(assistantId);
    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsLoading(true);

    wsRef.current.send(
      JSON.stringify({
        type: "query",
        query: input,
        language: "en",
      })
    );

    setInput("");
  };

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Connect on mount
  useEffect(() => {
    connect();
    return () => {
      stopPlayback();
      wsRef.current?.close();
    };
  }, [connect]);

  return (
    <div className="flex flex-col h-screen bg-linear-to-b from-zinc-900 to-zinc-950">
      {/* Header */}
      <header className="border-b border-zinc-800 p-4">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-white">Spirit AI</h1>
            <p className="text-sm text-zinc-400">Wisdom of Osho</p>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? "bg-green-500" : "bg-red-500"
              }`}
            />
            <span className="text-sm text-zinc-400">
              {isConnected ? "Connected" : "Disconnected"}
            </span>
          </div>
        </div>
      </header>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-20">
              <p className="text-zinc-500 text-lg">Ask Osho anything...</p>
              <p className="text-zinc-600 text-sm mt-2">
                "What is love?", "How to meditate?", "What is the meaning of life?"
              </p>
            </div>
          )}
          
          {messages.map((msg) => (
            <Card
              key={msg.id}
              className={`p-4 ${
                msg.type === "user"
                  ? "bg-zinc-800 border-zinc-700 ml-auto max-w-[80%]"
                  : "bg-zinc-900 border-zinc-800 mr-auto max-w-[90%]"
              }`}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                    msg.type === "user"
                      ? "bg-blue-600 text-white"
                      : "bg-amber-600 text-white"
                  }`}
                >
                  {msg.type === "user" ? "You" : "🕉️"}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-zinc-100 whitespace-pre-wrap">
                    {msg.content || (msg.isStreaming && "...")}
                  </p>
                  {msg.isStreaming && (
                    <span className="inline-block w-2 h-4 bg-amber-500 animate-pulse ml-1" />
                  )}
                  
                  {/* Show "Playing..." indicator during auto-play */}
                  {msg.type === "assistant" && !msg.autoPlayComplete && msg.audioChunks && msg.audioChunks.length > 0 && (
                    <div className="mt-2 text-xs text-zinc-500 flex items-center gap-2">
                      <span className="inline-block w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                      Playing audio...
                    </div>
                  )}
                  
                  {/* Audio Player - shown after auto-play completes */}
                  {msg.type === "assistant" && msg.autoPlayComplete && msg.audioChunks && msg.audioChunks.length > 0 && (
                    <AudioPlayer 
                      audioChunks={msg.audioChunks} 
                      onPlayStateChange={(isPlaying) => {
                        setPlayingMessageId(isPlaying ? msg.id : null);
                      }}
                    />
                  )}
                </div>
              </div>
            </Card>
          ))}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="border-t border-zinc-800 p-4">
        <div className="max-w-3xl mx-auto flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendQuery()}
            placeholder="Ask your question..."
            disabled={!isConnected || isLoading}
            className="bg-zinc-800 border-zinc-700 text-white placeholder:text-zinc-500"
          />
          <Button
            onClick={sendQuery}
            disabled={!isConnected || isLoading || !input.trim()}
            className="bg-amber-600 hover:bg-amber-700 text-white"
          >
            {isLoading ? "..." : "Send"}
          </Button>
        </div>
      </div>
    </div>
  );
}
