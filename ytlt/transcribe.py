"""Etapa 2a: reconhecimento de fala com dois modelos independentes.

- WhisperX large-v3 (GPU, float16) → `asr_whisperx.json` (tempos por palavra)
- NVIDIA Parakeet TDT 0.6B v3 via onnx-asr (CPU) → `asr_parakeet.json` (texto pontuado por trecho de fala)

A transcrição final (`transcript.json`) é produzida por `consensus.py`, que combina estes dois com as
legendas automáticas do YouTube.
"""

import gc
import json
from pathlib import Path

from . import config
from .llm import unload_ollama


def fmt_ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{h:02d}:{m:02d}:{s:02d}"


def free_gpu() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def transcribe_whisperx(vdir: Path, force: bool = False) -> Path:
    out = vdir / "asr_whisperx.json"
    if out.exists() and not force:
        return out

    import whisperx

    unload_ollama()
    audio = whisperx.load_audio(str(vdir / "audio.wav"))
    model = whisperx.load_model(
        config.WHISPER_MODEL, config.DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE, language=config.WHISPER_LANGUAGE,
        # Sem hotwords/initial_prompt: nos testes, eles fizeram o Whisper trocar frases inteiras por
        # listas de termos do prompt (alucinação detectada pelo consenso).
        asr_options={"beam_size": 5},
    )
    result = model.transcribe(audio, batch_size=config.WHISPER_BATCH_SIZE)
    language = result["language"]
    del model
    free_gpu()

    align_model, align_meta = whisperx.load_align_model(language_code=language, device=config.DEVICE)
    aligned = whisperx.align(result["segments"], align_model, align_meta, audio, config.DEVICE)
    del align_model
    free_gpu()

    payload = {"language": language, "model": config.WHISPER_MODEL, "segments": _clean_segments(aligned["segments"])}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    return out


def _clean_segments(segments: list[dict]) -> list[dict]:
    return [
        {
            "start": round(s["start"], 3),
            "end": round(s["end"], 3),
            "text": s["text"].strip(),
            "words": [{k: w[k] for k in ("word", "start", "end", "score") if k in w} for w in s.get("words", [])],
        }
        for s in segments
    ]


def transcribe_parakeet(vdir: Path, force: bool = False) -> Path:
    out = vdir / "asr_parakeet.json"
    if out.exists() and not force:
        return out

    import onnx_asr

    # onnxruntime-gpu 1.30 exige CUDA 13; na CPU o modelo 0.6B processa ~14 min de áudio em ~1 min.
    model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3", providers=["CPUExecutionProvider"])
    vad = onnx_asr.load_vad("silero")
    pipeline = model.with_vad(vad, max_speech_duration_s=25, min_silence_duration_ms=250, speech_pad_ms=100)
    segments = [
        {"start": round(r.start, 3), "end": round(r.end, 3), "text": r.text.strip()}
        for r in pipeline.recognize(str(vdir / "audio.wav"))
    ]
    out.write_text(json.dumps({"model": "nemo-parakeet-tdt-0.6b-v3", "segments": segments}, ensure_ascii=False, indent=1))
    return out


def realign(vdir: Path, segments: list[dict], language: str) -> list[dict]:
    """Recalcula os tempos por palavra (wav2vec2) para um texto já corrigido."""
    import whisperx

    audio = whisperx.load_audio(str(vdir / "audio.wav"))
    align_model, align_meta = whisperx.load_align_model(language_code=language, device=config.DEVICE)
    aligned = whisperx.align(segments, align_model, align_meta, audio, config.DEVICE)
    del align_model
    free_gpu()
    return _clean_segments(aligned["segments"])


def write_srt(segments: list[dict], path: Path) -> None:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines += [str(i), f"{fmt_ts(s['start'], srt=True)} --> {fmt_ts(s['end'], srt=True)}", s["text"], ""]
    path.write_text("\n".join(lines))
