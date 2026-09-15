# yt-local-transcribe

Transcrição de vídeos do YouTube **100% local e auditável**. O objetivo não é só gerar texto, é gerar texto
em que dá para confiar sem reassistir o vídeo:

- a transcrição sai do **consenso de três reconhecedores de fala** independentes, com os pontos de dúvida marcados;
- toda análise feita por LLM é **verificada contra a própria fala** (citações localizadas, afirmações com evidência);
- referências científicas só são ligadas a uma fala quando **detalhes verificáveis** coincidem.

Vídeo e áudio nunca saem da máquina. Os únicos acessos externos são o download (yt-dlp) e, opcionalmente,
consultas a PubMed / Crossref / OpenAlex com identificadores de artigos.

## Etapas

| Etapa | O que faz | Ferramentas | Saída por vídeo (`data/videos/NN-<id>/`) |
|---|---|---|---|
| `metadata` | título, capítulos, descrição | yt-dlp | `metadata.json` |
| `audio` | áudio WAV 16 kHz mono | yt-dlp + ffmpeg | `audio.wav` |
| `captions` | legendas automáticas (3º voto) | yt-dlp | `asr_youtube.json` |
| `whisperx` | reconhecimento principal + tempo por palavra | WhisperX large-v3 (GPU) + wav2vec2 | `asr_whisperx.json` |
| `parakeet` | segundo reconhecedor, arquitetura diferente | NVIDIA Parakeet TDT 0.6B v3 (CPU, ONNX) | `asr_parakeet.json` |
| `consensus` | alinha as três fontes, redecodifica disputas, vota, realinha | código próprio + Whisper | `transcript.json`, `transcript.srt`, `consensus.json` |
| `references` | estudos da descrição → PubMed/Crossref/OpenAlex; liga falas a estudos | NCBI, Crossref, OpenAlex + LLM local | `references.json` |
| `analyze` | análise do perfil do projeto | Ollama (Qwen3 30B-A3B) | `analysis.json` |
| `emphasis` | (tier_list) parte do músculo mais trabalhada, só se dita, com citação verificada | Ollama | `analysis.json` |
| `render` | documento legível + índice | — | Markdown |

Etapas concluídas são puladas (os arquivos são o cache). `--force <etapas>` refaz.

## Requisitos

- Linux, Python 3.12 (via [uv](https://docs.astral.sh/uv/)), `ffmpeg`
- GPU NVIDIA com ~8 GB (testado: RTX 2070 Super) e ~32 GB de RAM para o LLM de análise
- [Ollama](https://ollama.com) local para os perfis com LLM:
  ```bash
  ollama serve &
  ollama pull qwen3:30b-a3b-instruct-2507-q4_K_M   # ~19 GB, divide GPU e RAM
  ```

Os modelos de fala (Whisper, Parakeet, wav2vec2) são baixados automaticamente na primeira execução (~5 GB).

## Uso

```bash
uv sync

# 1. crie um projeto (uma pasta de trabalho, fora deste repositório)
uv run ytlt init ~/yt-projects/minhas-aulas --playlist "https://www.youtube.com/playlist?list=..." --profile transcript

# 2. rode
uv run ytlt -p ~/yt-projects/minhas-aulas list
uv run ytlt -p ~/yt-projects/minhas-aulas run --only 1          # valide com um vídeo
uv run ytlt -p ~/yt-projects/minhas-aulas run --isolate         # todos, um processo por vídeo

# refazer só a análise e os documentos
uv run ytlt -p ~/yt-projects/minhas-aulas run --stages analyze,render --force analyze,render
```

Configuração em `project.toml` (modelos, idioma, perfil); variáveis `YTLT_<NOME>` sobrescrevem
(ex.: `YTLT_WHISPER_BATCH_SIZE=2`). Veja [`examples/`](examples/).

## Perfis de análise

| Perfil | LLM | Para quê | Saídas |
|---|---|---|---|
| `transcript` | não | só a transcrição por consenso | `transcripts/*.md` |
| `tier_list` | sim | vídeos que ranqueiam itens em tiers (S…F) | `analysis.json`, `encyclopedia/*.md`, `export/<projeto>.json` |

Criar um perfil novo: [`docs/PROFILES.md`](docs/PROFILES.md).

## Tradução da análise

```bash
uv run ytlt -p <projeto> i18n extract   # translation/en/*.json com chaves estáveis
# traduza para translation/pt/*.json (mesmas chaves), com glossário em translation/glossary.json
uv run ytlt -p <projeto> i18n check     # chaves, números, tiers, siglas, nomes do glossário
```

Transcrição e citações literais ficam sempre no idioma original.

## Documentação

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — como cada etapa funciona e por quê
- [`docs/LESSONS.md`](docs/LESSONS.md) — o que deu errado nos testes e como foi resolvido
- [`docs/PROFILES.md`](docs/PROFILES.md) — escrever um perfil de análise

## Licenças

O código deste repositório é MIT ([`LICENSE`](LICENSE)). Os modelos e ferramentas usados têm licenças próprias,
baixados por você na primeira execução:

| Componente | Uso | Licença |
|---|---|---|
| [OpenAI Whisper](https://github.com/openai/whisper) large-v3 (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)) | reconhecimento principal | MIT |
| [WhisperX](https://github.com/m-bain/whisperX) | lote, VAD e alinhamento | BSD-2-Clause |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | detecção de voz | MIT |
| [wav2vec2 base 960h](https://pytorch.org/audio/stable/pipelines.html) (torchaudio) | alinhamento por palavra | ver termos do modelo no torchaudio |
| [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (via [onnx-asr](https://github.com/istupakov/onnx-asr)) | segundo reconhecedor | CC-BY-4.0 (modelo) · MIT (onnx-asr) |
| [Silero VAD](https://github.com/snakers4/silero-vad) | detecção de voz para o Parakeet | MIT |
| [Qwen3](https://huggingface.co/Qwen) via [Ollama](https://ollama.com) | análise | Apache 2.0 |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | download | Unlicense |

Confira os termos de cada modelo antes de redistribuir resultados comercialmente.

## Uso responsável

A ferramenta processa conteúdo de terceiros. Transcrições completas são para uso pessoal; ao publicar algo
derivado, prefira resumos, trechos curtos e links para o vídeo original.
