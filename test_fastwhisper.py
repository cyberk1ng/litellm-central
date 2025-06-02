from fastwhisper_service import FastWhisperService
import sys
import time

def test_transcription(audio_path: str):
    """Test FastWhisper transcription with a given audio file."""
    print(f"Testing transcription with file: {audio_path}")
    
    # Initialize FastWhisper service
    service = FastWhisperService()
    
    # Print available models
    print("\nAvailable models:")
    for model in service.get_available_models():
        print(f"  - {model}")
    
    # Test transcription with small model
    print("\nTranscribing with 'small' model...")
    start_time = time.time()
    
    try:
        result = service.transcribe(
            audio_path,
            model_size="small",
            language=None  # Auto-detect language
        )
        
        duration = time.time() - start_time
        print(f"\nTranscription completed in {duration:.2f} seconds")
        print(f"\nDetected language: {result['language']} (confidence: {result['language_probability']:.2f})")
        print("\nTranscription:")
        print(result['text'])
        
        print("\nSegments:")
        for segment in result['segments']:
            print(f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}")
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_fastwhisper.py <path_to_audio_file>")
        sys.exit(1)
    
    test_transcription(sys.argv[1]) 