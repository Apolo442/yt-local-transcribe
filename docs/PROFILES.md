# Perfis de análise

Um perfil é um pacote em `ytlt/profiles/<nome>/` registrado em `ytlt/profiles/__init__.py` (`PROFILES`).
Ele recebe a pasta de um vídeo já transcrito (`transcript.json`, `metadata.json`, `consensus.json`,
opcionalmente `references.json`) e deve expor:

| Função | Retorno | Obrigatória |
|---|---|---|
| `analyze(vdir, force=False, progress=print)` | caminho de `analysis.json` | sim |
| `render(vdir)` | documento legível do vídeo | sim |
| `render_index()` | índice do projeto | sim |
| `units(vdir)` | `{chave estável: texto}` traduzível | sim (pode ser `{}`) |
| `export()` | exportação agregada | não (levante `SystemExit` se não houver) |

## Regras que valem para qualquer perfil
- Use `ytlt.llm.chat(system, user, Schema, num_ctx=...)` com um modelo Pydantic: nunca parse texto livre.
- Tudo que o LLM afirma sobre a fala precisa ser verificável: peça citações literais e confira com
  `textmatch.Locator`; itens resumidos passam por `grounding.verify_support`.
- Divida a transcrição por capítulo (ponto médio da frase) para manter o contexto pequeno e 100% na GPU.
- Nada de estado global além de `config`; escreva só dentro de `config.PROJECT_DIR`.
- Falhas do LLM (`LLMOutputError`) degradam o item, não o vídeo.

## Exemplo mínimo
`ytlt/profiles/transcript/` — sem LLM, só Markdown com capítulos, timestamps e trechos incertos.

## Exemplo completo
`ytlt/profiles/tier_list/` — análise por capítulo, tier lido da fala com pontuação de contexto, verificação de
evidência, classificação ciência × opinião, Markdown, export JSON bilíngue.
