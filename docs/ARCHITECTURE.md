# Arquitetura

```
yt-dlp ─► metadata.json, audio.wav, asr_youtube.json
             │
             ├─► WhisperX large-v3 (GPU) ─► asr_whisperx.json (tempo por palavra)
             ├─► Parakeet TDT 0.6B v3 (CPU) ─► asr_parakeet.json
             ▼
         consensus ─► transcript.json / .srt + consensus.json (cada decisão registrada)
             │
             ├─► references (opcional) ─► references.json
             ▼
         perfil de análise (LLM local, verificado) ─► analysis.json ─► render / export / i18n
```

Código: `ytlt/` (núcleo) e `ytlt/profiles/<perfil>/`. Dados: sempre na pasta do projeto, fora do repositório.

## Configuração por projeto (`config.py`)
`project.toml` → variáveis `config.X`, sobrescritas por `YTLT_<NOME>`. Os módulos leem `config.X` em tempo de
execução, então `load_project()` na CLI basta para trocar de projeto.

## Reconhecimento de fala (`transcribe.py`)
- **WhisperX**: faster-whisper large-v3 em lote na GPU (beam 5), VAD do pyannote, alinhamento wav2vec2 por palavra.
  Sem `hotwords`/`initial_prompt` (causaram alucinação — ver LESSONS).
- **Parakeet** (onnx-asr): arquitetura TDT, erros descorrelacionados do Whisper. Roda em CPU (onnxruntime-gpu
  exigia CUDA 13); ~1 min para 14 min de áudio.
- **Legendas automáticas do YouTube**: fracas sozinhas, úteis como terceiro voto.
- A VRAM do Ollama é liberada antes das etapas de GPU (`llm.unload_ollama`).

## Consenso (`consensus.py`)
1. Cada fonte vira tokens canônicos: minúsculas, sem pontuação, números por extenso, `gonna`→`going to`,
   hesitações removidas. Diferenças só de grafia deixam de contar.
2. Parakeet e YouTube são alinhados ao WhisperX (difflib). `BMap` guarda início/fim separados para que
   inserções puras fiquem dentro da região.
3. Regiões divergentes:
   - WhisperX concorda com alguma fonte → mantém;
   - senão, o trecho é **redecodificado isolado** (beam 10, contexto anterior como prompt);
   - variantes de grafia são agrupadas (similaridade ≥ 0,8);
   - termo presente no título/capítulos/descrição vence (glossário do vídeo);
   - maioria simples vence se tiver apoio de fonte **independente** (Parakeet/YouTube);
   - empate → mantém WhisperX; marca ❓ se as opções diferem em palavra de conteúdo.
4. Segmentos sem pontuação (artefato do modo em lote) recebem a pontuação do Parakeet, só em palavras idênticas.
5. Texto final dividido em frases e realinhado por palavra.

## LLM local (`llm.py`)
JSON Schema do Ollama (`format`), `num_predict` limitado, `repeat_penalty`, 3 tentativas com leve variação;
falha persistente → `LLMOutputError` (a etapa decide o que fazer, nunca derruba o lote).

## Fidelidade (`grounding.py`, `textmatch.py`)
- `Locator.locate(trecho)`: similaridade e segundo inicial de qualquer citação na transcrição.
- `verify_support`: itens gerados só ficam se (a) compartilham radicais com o trecho de origem e (b) um
  verificador aponta **evidência literal** que o `Locator` encontra de fato na transcrição.
- `strip_llm_wrappers`, `looks_portuguese`: limpeza e detecção de troca de idioma.

## Referências (`references.py`)
Descrição → grupos + links → PubMed (PMID), Crossref (DOI, título do link, ISSN/volume), PMC (idconv),
metadados `citation_doi` da página, `reference_overrides.json` (identificações manuais). Abstracts via PubMed/
OpenAlex só para verificação local. Uma fala só é ligada a um estudo com números com unidade ou ≥2 termos
raros no catálogo do vídeo.

## Tradução (`i18n.py`)
Chaves estáveis definidas pelo perfil (`units`). `check` valida chaves, números, letras de tier, siglas e nomes
do glossário. Metodologia usada: glossário verificado → tradutores → revisor independente → correções conferidas.
