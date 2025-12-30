# Soniox Speech-to-Text Setup

## Prerequisites

1. **Soniox API Key**: Get your API key from [console.soniox.com](https://console.soniox.com)

## Environment Setup

1. Create a `.env.local` file in the `ui-client` directory:
```bash
# Soniox API Key for Speech-to-Text
NEXT_PUBLIC_SONIOX_API_KEY=your_actual_api_key_here
```

2. Replace `your_actual_api_key_here` with your real Soniox API key

## How It Works

The microphone button now:
- ✅ Records audio from your microphone
- ✅ Connects to Soniox WebSocket API
- ✅ Streams audio data in real-time
- ✅ Receives speech-to-text transcriptions
- ✅ Logs all responses to the browser console

## Features

- **Real-time transcription**: Audio is processed and sent to Soniox as you speak
- **WebSocket connection**: Maintains persistent connection for low-latency transcription
- **Audio processing**: Converts microphone audio to the format Soniox expects
- **Error handling**: Gracefully handles connection issues and API errors
- **Visual feedback**: Button changes color and text updates when listening

## Testing

1. Click the microphone button to start listening
2. Speak into your microphone
3. Check the browser console for transcriptions
4. Click the microphone button again to stop listening

## Console Output

You'll see logs for:
- WebSocket connection status
- Audio processing
- Soniox API responses
- Transcribed text
- Any errors or connection issues

## Troubleshooting

- **Microphone access denied**: Make sure to allow microphone access in your browser
- **API key error**: Verify your `.env.local` file has the correct API key
- **Connection issues**: Check your internet connection and Soniox service status
