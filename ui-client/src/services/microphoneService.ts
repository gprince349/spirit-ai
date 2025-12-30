export type AudioDataCallback = (audioData: ArrayBuffer) => void;
export type ErrorCallback = (error: string) => void;

export interface MicrophoneConfig {
  sampleRate: number;
  channelCount: number;
  echoCancellation: boolean;
  noiseSuppression: boolean;
}

export class MicrophoneService {
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private processor: ScriptProcessorNode | null = null;
  private onAudioData: AudioDataCallback;
  private onError: ErrorCallback;
  private config: MicrophoneConfig;

  constructor(
    config: MicrophoneConfig,
    onAudioData: AudioDataCallback,
    onError: ErrorCallback
  ) {
    this.config = config;
    this.onAudioData = onAudioData;
    this.onError = onError;
  }

  async start(): Promise<void> {
    try {
      console.log('Starting microphone...');
      
      // Get microphone access
      this.stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channelCount,
          echoCancellation: this.config.echoCancellation,
          noiseSuppression: this.config.noiseSuppression,
        } 
      });
      
      // Create audio context for processing
      this.audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
      
      // Create source from stream
      const source = this.audioContext.createMediaStreamSource(this.stream);
      
      // Create script processor for audio data
      this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      
      // Connect audio nodes
      source.connect(this.processor);
      this.processor.connect(this.audioContext.destination);
      
      // Process audio data
      this.processor.onaudioprocess = (event) => {
        const inputData = event.inputBuffer.getChannelData(0);
        
        // Convert float32 to int16
        const int16Array = new Int16Array(inputData.length);
        for (let i = 0; i < inputData.length; i++) {
          int16Array[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
        }
        
        // Send audio data to callback
        this.onAudioData(int16Array.buffer);
      };
      
      console.log('Microphone started successfully');
      
    } catch (error) {
      console.error('Error starting microphone:', error);
      this.onError('Failed to start microphone');
      throw error;
    }
  }

  stop(): void {
    console.log('Stopping microphone...');
    
    // Stop media stream
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    
    // Close audio context
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    // Clean up processor
    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }
    
    console.log('Microphone stopped');
  }

  isActive(): boolean {
    return this.stream !== null && this.audioContext !== null;
  }

  getStream(): MediaStream | null {
    return this.stream;
  }
}
