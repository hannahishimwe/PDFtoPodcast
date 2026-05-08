import os, re, mimetypes, struct, base64, tempfile
from google import genai
from google.genai import types
from app.settings import client, model

##This code was taken from Google AI Studio!
class ExportTranscript:
    OUTPUT_BASENAME = "podcast"           # generates podcast_000.mp3, etc.
    PAUSE_MS = 350                        # silence between chunks 

    # Maps speakers to Gemini prebuilt generative voices
    PREBUILT_VOICES = {
        "Speaker 1": "sadaltager",
        "Speaker 2": "achernar",
    }

    def init(self, transcript_file_name):
        self.transcript = self.load_transcript(transcript_file_name)

    def load_transcript(self, file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                tagged_transcript = f.read().strip()
        except FileNotFoundError:
            print("Error: transcript.txt not found. Please create this file.")
            tagged_transcript = ""
        return tagged_transcript

  
    def save_binary_file(self, file_name: str, data: bytes):
        with open(file_name, "wb") as f:
            f.write(data)
        print(f"Saved: {file_name}")

    def convert_to_wav(self, audio_data: bytes, mime_type: str) -> bytes:
        """Wrap raw PCM into WAV based on mime parameters like audio/L16;rate=24000"""
        params = self.parse_audio_mime_type(mime_type)
        bits_per_sample = params["bits_per_sample"]
        sample_rate = params["rate"]
        num_channels = 1
        data_size = len(audio_data)
        block_align = num_channels * (bits_per_sample // 8)
        byte_rate = sample_rate * block_align
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,
            1,
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        return header + audio_data

    def parse_audio_mime_type(self, mime_type: str) -> dict:
        bits_per_sample = 16
        rate = 24000
        for part in mime_type.split(";"):
            p = part.strip()
            if p.lower().startswith("rate="):
                try:
                    rate = int(p.split("=", 1)[1])
                except Exception:
                    pass
            if p.startswith("audio/L"):
                try:
                    bits_per_sample = int(p.split("L", 1)[1])
                except Exception:
                    pass
        return {"bits_per_sample": bits_per_sample, "rate": rate}

    def build_multi_speaker_config(self):
        return types.SpeechConfig(
            multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(
                        speaker=spk,
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        ),
                    )
                    for spk, voice in self.PREBUILT_VOICES.items()
                ]
            ),
            # You can also set global speaking rate / pitch here if desired
            # speaking_rate=1.02, pitch_semitones=0.0,
        )

    def build_contents_from_transcript(self, text: str) -> types.Content:
        # The generative TTS endpoint can parse speaker tags if included literally.
        # Ensure lines start with "Speaker 1:" or "Speaker 2:" etc.
        text = self.transcript if isinstance(self.transcript, str) else str(self.transcript)
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        )



    def generate(self):
        content = self.build_contents_from_transcript(self.transcript) #type:ignore
        generate_content_config = types.GenerateContentConfig(
            temperature=1.0,
            response_modalities=["audio"],
            speech_config=self.build_multi_speaker_config(),
            # You can hint an output format; if not honoured, we’ll derive from mime type.
            # audio_config=types.AudioConfig(mime_type="audio/mp3"),
        )

        file_index = 0
        for chunk in client.models.generate_content_stream(
        model=model,
        contents=content,
        config=generate_content_config,
    ):
            if not chunk.candidates:
                continue

            candidate = chunk.candidates[0]
            content_obj = getattr(candidate, "content", None)
            if not content_obj or not getattr(content_obj, "parts", None):
                continue

            for part in content_obj.parts:
                # Always check type and presence before accessing fields
                inline = getattr(part, "inline_data", None)
                text_field = getattr(part, "text", None)

                # ---- Handle audio parts ----
                if inline and getattr(inline, "data", None):
                    data = inline.data
                    try:
                        audio_bytes = base64.b64decode(data) if isinstance(data, str) else data
                    except Exception as e:
                        print(f"⚠️ Could not decode audio chunk: {e}")
                        continue

                    mime_type = getattr(inline, "mime_type", "audio/wav") or "audio/wav"
                    ext = mimetypes.guess_extension(mime_type) or ".wav"

                    if mime_type.startswith("audio/L"):
                        audio_bytes = self.convert_to_wav(audio_bytes, mime_type)
                        ext = ".wav"

                    out_name = f"{self.OUTPUT_BASENAME}_{file_index:03d}{ext}"
                    file_index += 1
                    self.save_binary_file(out_name, audio_bytes)

                # ---- Handle text parts (logs) ----
                elif text_field:
                    print(text_field)
