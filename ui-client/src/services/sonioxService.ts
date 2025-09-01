export interface SonioxConfig {
  api_key: string;
  model: string;
  language_hints: string[];
  enable_language_identification: boolean;
  enable_speaker_diarization: boolean;
  enable_endpoint_detection: boolean;
  audio_format: string;
  sample_rate: number;
  num_channels: number;
}

export interface SonioxToken {
  text?: string;
  is_final?: boolean;
  speaker?: string;
  language?: string;
}

export interface SonioxResponse {
  tokens?: SonioxToken[];
  error_code?: string;
  error_message?: string;
  finished?: boolean;
}

export type TranscriptCallback = (finalTranscript: string, liveTranscript: string) => void;
export type ConnectionCallback = (isConnected: boolean) => void;
export type ErrorCallback = (error: string) => void;

export class SonioxService {
  private ws: WebSocket | null = null;
  private config: SonioxConfig;
  private onTranscript: TranscriptCallback;
  private onConnectionChange: ConnectionCallback;
  private onError: ErrorCallback;
  private finalTranscript: string = '';
  private liveTranscript: string = '';

  constructor(
    config: SonioxConfig,
    onTranscript: TranscriptCallback,
    onConnectionChange: ConnectionCallback,
    onError: ErrorCallback
  ) {
    this.config = config;
    this.onTranscript = onTranscript;
    this.onConnectionChange = onConnectionChange;
    this.onError = onError;
  }

  private getConfig(): SonioxConfig {
    return {
      api_key: this.config.api_key,
      model: "stt-rt-preview",
      language_hints: ["en", "hi"],
      enable_language_identification: true,
      enable_speaker_diarization: true,
      enable_endpoint_detection: true,
      audio_format: "pcm_s16le",
      sample_rate: 16000,
      num_channels: 1,
    };
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        console.log('Connecting to Soniox WebSocket...');
        this.ws = new WebSocket('wss://stt-rt.soniox.com/transcribe-websocket');
        
        this.ws.onopen = () => {
          console.log('WebSocket connected to Soniox');
          this.onConnectionChange(true);
          
          // Clear previous transcripts for new session
          this.finalTranscript = '';
          this.liveTranscript = '';
          this.onTranscript(this.finalTranscript, this.liveTranscript);
          
          // Send config
          const config = this.getConfig();
          this.ws!.send(JSON.stringify(config));
          console.log('Config sent to Soniox');
          
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const response: SonioxResponse = JSON.parse(event.data);
            console.log('Soniox response:', response);
            
            // Handle errors
            if (response.error_code) {
              const errorMsg = `Soniox error: ${response.error_code} - ${response.error_message}`;
              console.error(errorMsg);
              this.onError(errorMsg);
              return;
            }
            
            // Handle tokens
            if (response.tokens) {
              let currentTranscript = '';
              let hasNewFinalTokens = false;
              
              response.tokens.forEach((token: SonioxToken) => {
                if (token.text && token.text !== '<end>') {
                  currentTranscript += token.text;
                  if (token.is_final) {
                    hasNewFinalTokens = true;
                  }
                }
              });
              
              if (currentTranscript.trim()) {
                console.log('Transcript:', currentTranscript);
                
                if (hasNewFinalTokens) {
                  // Update final transcript with new final tokens
                  this.finalTranscript += currentTranscript;
                  this.liveTranscript = ''; // Clear live transcript for next iteration
                } else {
                  // Update live transcript with non-final tokens
                  this.liveTranscript = currentTranscript;
                }
                
                // Notify callback
                this.onTranscript(this.finalTranscript, this.liveTranscript);
              }
            }
            
            // Handle session completion
            if (response.finished) {
              console.log('Soniox session finished');
              // Add any remaining live transcript to final
              if (this.liveTranscript.trim()) {
                this.finalTranscript += this.liveTranscript;
                this.liveTranscript = '';
                this.onTranscript(this.finalTranscript, this.liveTranscript);
              }
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            this.onError('Error parsing WebSocket message');
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.onConnectionChange(false);
          this.onError('WebSocket connection error');
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket connection closed');
          this.onConnectionChange(false);
        };

      } catch (error) {
        console.error('Error creating WebSocket:', error);
        this.onError('Failed to create WebSocket connection');
        reject(error);
      }
    });
  }

  sendAudio(audioData: ArrayBuffer): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(audioData);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.onConnectionChange(false);
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  clearTranscripts(): void {
    this.finalTranscript = '';
    this.liveTranscript = '';
    this.onTranscript(this.finalTranscript, this.liveTranscript);
  }
}
