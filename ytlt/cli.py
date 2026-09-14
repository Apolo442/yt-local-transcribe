"""ytlt — transcrição local e auditável de vídeos do YouTube.

Exemplos:
  ytlt init ~/yt-projects/aulas --playlist "https://www.youtube.com/playlist?list=..." --profile transcript
  ytlt -p ~/yt-projects/aulas list
  ytlt -p ~/yt-projects/aulas run --only 1            # processa só o primeiro vídeo
  ytlt -p ~/yt-projects/aulas run --isolate            # um processo por vídeo (libera RAM/VRAM entre vídeos)
  ytlt -p ~/yt-projects/easy-jeff run --stages analyze,render --force analyze,render
  ytlt -p ~/yt-projects/easy-jeff export
  ytlt -p ~/yt-projects/easy-jeff i18n check
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from . import config

STAGES = ["metadata", "audio", "captions", "whisperx", "parakeet", "consensus", "references", "analyze", "emphasis", "render"]
PROJECT_TEMPLATE = """[project]
name = "{name}"
# playlist OU lista de vídeos avulsos
playlist = "{playlist}"
videos = []
# transcript (só transcrição, sem LLM) | tier_list (vídeos de ranking em tiers)
profile = "{profile}"
# idioma dos textos gerados pela análise
output_language = "English"

[asr]
whisper_model = "large-v3"
compute_type = "float16"   # float16 em GPUs com 8 GB; int8_float16 para economizar VRAM
batch_size = 4
# language = "en"          # descomente para fixar o idioma (padrão: detecção automática)

[llm]
model = "qwen3:30b-a3b-instruct-2507-q4_K_M"
num_thread = 4
num_ctx = 8192
"""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_selection(spec: str | None, total: int) -> list[int]:
    if not spec:
        return list(range(1, total + 1))
    picked: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            picked.update(range(int(a), int(b) + 1))
        else:
            picked.add(int(part))
    return sorted(i for i in picked if 1 <= i <= total)


def _entries() -> list[dict]:
    from . import fetch

    if config.PLAYLIST_URL:
        return fetch.list_playlist(config.PLAYLIST_URL)
    return fetch.list_videos(config.VIDEO_URLS)


def cmd_init(args) -> None:
    root = Path(args.dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    toml = root / "project.toml"
    if toml.exists() and not args.force:
        raise SystemExit(f"{toml} já existe (use --force para sobrescrever)")
    toml.write_text(PROJECT_TEMPLATE.format(name=args.name or root.name, playlist=args.playlist or "", profile=args.profile))
    log(f"projeto criado: {toml}")


def cmd_list(args) -> None:
    for e in _entries():
        print(f"{e['index']:>3}  {e['id']}  {e['duration'] or 0:>6}s  {e['title']}")


def cmd_run(args) -> None:
    from . import consensus, fetch, references, transcribe
    from .profiles import get_profile

    profile = get_profile(config.PROFILE)
    entries = _entries()
    (config.DATA_DIR).mkdir(parents=True, exist_ok=True)
    (config.DATA_DIR / "playlist.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    stages = STAGES if not args.stages else args.stages.split(",")
    force = set(args.force.split(",")) if args.force else set()
    selected = parse_selection(args.only, len(entries))

    if args.isolate and len(selected) > 1:
        # um processo por vídeo: RAM e VRAM são devolvidas ao sistema entre vídeos
        base = [sys.executable, "-m", "ytlt.cli", "-p", str(config.PROJECT_DIR), "run"]
        extra = (["--stages", args.stages] if args.stages else []) + (["--force", args.force] if args.force else [])
        for idx in selected:
            code = subprocess.call(base + ["--only", str(idx)] + extra + ["--keep-going", "--no-index"])
            if code and not args.keep_going:
                raise SystemExit(code)
        if "render" in stages:
            log(f"índice: {profile.render_index()}")
        return

    for idx in selected:
        e = entries[idx - 1]
        vdir = config.video_dir(idx, e["id"])
        log(f"#{idx} {e['title']}")
        try:
            if "metadata" in stages:
                fetch.save_metadata(idx, e["id"], force="metadata" in force)
                log("  metadados ok")
            if "audio" in stages:
                fetch.download_audio(idx, e["id"], force="audio" in force)
                log("  áudio ok")
            if "captions" in stages:
                ok = fetch.download_auto_captions(idx, e["id"], force="captions" in force)
                log(f"  legendas automáticas {'ok' if ok else 'indisponíveis'}")
            if "whisperx" in stages:
                t0 = time.time()
                transcribe.transcribe_whisperx(vdir, force="whisperx" in force)
                log(f"  whisperx ok ({time.time() - t0:.0f}s)")
            if "parakeet" in stages:
                t0 = time.time()
                transcribe.transcribe_parakeet(vdir, force="parakeet" in force)
                log(f"  parakeet ok ({time.time() - t0:.0f}s)")
            if "consensus" in stages:
                t0 = time.time()
                consensus.build(vdir, force="consensus" in force, progress=log)
                log(f"  consenso ok ({time.time() - t0:.0f}s)")
            if "references" in stages:
                t0 = time.time()
                references.build(vdir, force="references" in force, progress=log)
                log(f"  referências ok ({time.time() - t0:.0f}s)")
            if "analyze" in stages:
                t0 = time.time()
                profile.analyze(vdir, force="analyze" in force, progress=log)
                log(f"  análise ({config.PROFILE}) ok ({time.time() - t0:.0f}s)")
            if "emphasis" in stages and hasattr(profile, "emphasis"):
                t0 = time.time()
                profile.emphasis(vdir, force="emphasis" in force, progress=log)
                log(f"  ênfase ok ({time.time() - t0:.0f}s)")
            if "render" in stages:
                log(f"  documento: {profile.render(vdir)}")
        except Exception as exc:
            log(f"  ERRO: {exc!r}")
            if not args.keep_going:
                raise
    if "render" in stages and not args.no_index:
        log(f"índice: {profile.render_index()}")


def cmd_export(args) -> None:
    from .profiles import get_profile

    log(f"export: {get_profile(config.PROFILE).export()}")


def cmd_i18n(args) -> None:
    from . import i18n

    if args.action == "extract":
        for path in i18n.extract():
            log(f"extraído: {path}")
        return
    problems = i18n.check()
    for pr in problems:
        print(pr)
    log(f"{len(problems)} problema(s)")
    if problems:
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ytlt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--project", help="pasta do projeto (padrão: $YTLT_PROJECT ou a pasta atual)")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("init", help="cria um projeto (project.toml)")
    i.add_argument("dir")
    i.add_argument("--name")
    i.add_argument("--playlist")
    i.add_argument("--profile", default="transcript", choices=["transcript", "tier_list"])
    i.add_argument("--force", action="store_true")
    i.set_defaults(func=cmd_init, needs_project=False)

    sub.add_parser("list", help="lista os vídeos do projeto").set_defaults(func=cmd_list, needs_project=True)

    r = sub.add_parser("run", help="executa o pipeline")
    r.add_argument("--only", help="vídeos: 1 | 1,3 | 2-5 (padrão: todos)")
    r.add_argument("--stages", help=f"subconjunto de etapas: {','.join(STAGES)}")
    r.add_argument("--force", help="refaz etapas já concluídas, ex.: analyze,render")
    r.add_argument("--keep-going", action="store_true", help="não parar no primeiro erro")
    r.add_argument("--isolate", action="store_true", help="um processo por vídeo")
    r.add_argument("--no-index", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run, needs_project=True)

    sub.add_parser("export", help="exportação agregada do perfil").set_defaults(func=cmd_export, needs_project=True)

    t = sub.add_parser("i18n", help="tradução da análise (extract | check)")
    t.add_argument("action", choices=["extract", "check"])
    t.set_defaults(func=cmd_i18n, needs_project=True)

    args = p.parse_args(argv)
    if args.needs_project:
        root = config.load_project(args.project)
        if not (root / "project.toml").exists():
            raise SystemExit(f"nenhum project.toml em {root} (crie com: ytlt init {root})")
    args.func(args)


if __name__ == "__main__":
    main()
