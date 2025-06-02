from faster_whisper import WhisperModel
from typing import Dict, Any, Optional
import os
import torch

class FastWhisperService:
    def __init__(self):
        self.model = None
        self.available_models = [
            "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
        ]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

    def load_model(self, model_size: str = "small") -> None:
        """Load a Whisper model of specified size."""
        if model_size not in self.available_models:
            raise ValueError(f"Model size must be one of {self.available_models}")
        
        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=self.compute_type
        )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: str = "small"
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using FastWhisper.
        
        Args:
            audio_path (str): Path to the audio file
            language (Optional[str]): Language code (e.g., 'en', 'de')
            model_size (str): Model size to use
            
        Returns:
            Dict[str, Any]: Transcription results including text, language, and segments
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self.model is None or self.model.model_size != model_size:
            self.load_model(model_size)

        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True
        )

        # Collect all segments
        all_segments = []
        for segment in segments:
            all_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "average_logprob": getattr(segment, 'average_logprob', None)
            })

        return {
            "text": " ".join(seg["text"] for seg in all_segments),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": all_segments
        }

    def get_available_models(self) -> list:
        """Get list of available model sizes."""
        return self.available_models 