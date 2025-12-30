import { MicrophoneService, MicrophoneConfig } from './microphoneService';
import { SonioxService, SonioxConfig } from './sonioxService';

export interface SpeechServiceConfig {
  microphone: MicrophoneConfig;
  soniox: SonioxConfig;
}

export type TranscriptCallback = (finalTranscript: string, liveTranscript: string) => void;
export type ConnectionCallback = (isConnected: boolean) => void;
export type ErrorCallback = (error: string) => void;

export class SpeechService {
  private microphoneService: MicrophoneService;
  private sonioxService: SonioxService;
  private isListening: boolean = false;
  private isConnected: boolean = false;

  constructor(
    config: SpeechServiceConfig,
    onTranscript: TranscriptCallback,
    onConnectionChange: ConnectionCallback,
    onError: ErrorCallback
  ) {
    // Initialize microphone service
    this.microphoneService = new MicrophoneService(
      config.microphone,
      (audioData) => this.handleAudioData(audioData),
      (error) => this.handleMicrophoneError(error)
    );

    // Initialize Soniox service
    this.sonioxService = new SonioxService(
      config.soniox,
      onTranscript,
      onConnectionChange,
      onError
    );
  }

  private handleAudioData(audioData: ArrayBuffer): void {
    if (this.isConnected) {
      this.sonioxService.sendAudio(audioData);
    }
  }

  private handleMicrophoneError(error: string): void {
    console.error('Microphone error:', error);
    this.isListening = false;
  }

  async startListening(): Promise<void> {
    try {
      // Start microphone first
      await this.microphoneService.start();
      this.isListening = true;

      // Then connect to Soniox
      await this.sonioxService.connect();
      this.isConnected = true;

      console.log('Speech service started successfully');
    } catch (error) {
      console.error('Error starting speech service:', error);
      this.stopListening();
      throw error;
    }
  }

  stopListening(): void {
    console.log('Stopping speech service...');
    
    // Stop microphone
    this.microphoneService.stop();
    this.isListening = false;
    
    // Disconnect from Soniox
    this.sonioxService.disconnect();
    this.isConnected = false;
    
    // Clear transcripts
    this.sonioxService.clearTranscripts();
    
    console.log('Speech service stopped');
  }

  isActive(): boolean {
    return this.isListening && this.isConnected;
  }

  getMicrophoneStatus(): boolean {
    return this.isListening;
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  // Method to get default configuration
  static getDefaultConfig(apiKey: string): SpeechServiceConfig {
    return {
      microphone: {
        sampleRate: 16000,
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
      soniox: {
        api_key: apiKey,
        model: "stt-rt-preview",
        language_hints: ["en", "hi"],
        enable_language_identification: true,
        enable_speaker_diarization: true,
        enable_endpoint_detection: true,
        audio_format: "pcm_s16le",
        sample_rate: 16000,
        num_channels: 1,
      }
    };
  }
}
