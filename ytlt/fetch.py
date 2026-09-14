"""Etapas 1: listar playlist, salvar metadados e baixar o áudio (yt-dlp) + converter (ffmpeg)."""

import json
import subprocess
from pathlib import Path

import yt_dlp

from . import config

METADATA_FIELDS = [
    "id", "title", "description", "channel", "channel_id", "uploader", "upload_date",
    "duration", "view_count", "like_count", "comment_count", "tags", "categories",
    "chapters", "language", "webpage_url", "thumbnail",
]


def list_playlist(url: str | None = None) -> list[dict]:
    url = url or config.PLAYLIST_URL
    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = []
    for i, e in enumerate(info.get("entries") or [], start=1):
        entries.append({"index": i, "id": e["id"], "title": e.get("title"), "duration": e.get("duration")})
    return entries


def list_videos(urls: list[str]) -> list[dict]:
    """Vídeos avulsos (sem playlist), na ordem dada."""
    entries = []
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        for i, url in enumerate(urls, start=1):
            info = ydl.extract_info(url, download=False)
            entries.append({"index": i, "id": info["id"], "title": info.get("title"), "duration": info.get("duration")})
    return entries


def save_metadata(index: int, video_id: str, playlist_title: str | None = None, force: bool = False) -> Path:
    out = config.video_dir(index, video_id) / "metadata.json"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    meta = {k: info.get(k) for k in METADATA_FIELDS}
    meta["playlist_index"] = index
    meta["playlist_title"] = playlist_title
    out.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return out


def download_auto_captions(index: int, video_id: str, force: bool = False) -> Path | None:
    """Baixa as legendas automáticas do YouTube (3º reconhecedor independente, só para votação)."""
    vdir = config.video_dir(index, video_id)
    out = vdir / "asr_youtube.json"
    if out.exists() and not force:
        return out
    vdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "skip_download": True, "writeautomaticsub": True, "subtitleslangs": ["en-orig", "en"],
        "subtitlesformat": "json3", "outtmpl": str(vdir / "yt_auto.%(ext)s"), "quiet": True, "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
    candidates = [vdir / "yt_auto.en-orig.json3", vdir / "yt_auto.en.json3"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        return None
    segments = []
    for ev in json.loads(src.read_text()).get("events", []):
        base = ev.get("tStartMs", 0)
        for seg in ev.get("segs") or []:
            text = seg.get("utf8", "").strip()
            if text:
                segments.append({"start": (base + seg.get("tOffsetMs", 0)) / 1000, "text": text})
    out.write_text(json.dumps({"source": src.name, "segments": segments}, ensure_ascii=False, indent=1))
    for p in candidates:
        p.unlink(missing_ok=True)
    return out


def download_audio(index: int, video_id: str, force: bool = False) -> Path:
    """Baixa só a trilha de áudio e converte para WAV 16 kHz mono (formato nativo do Whisper)."""
    vdir = config.video_dir(index, video_id)
    wav = vdir / "audio.wav"
    if wav.exists() and not force:
        return wav
    vdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(vdir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "overwrites": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
        src = Path(ydl.prepare_filename(info))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)],
        check=True,
    )
    src.unlink(missing_ok=True)
    return wav
