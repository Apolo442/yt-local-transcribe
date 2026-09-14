# Lições dos testes

Problemas reais encontrados ao processar a primeira playlist, e a correção adotada.

| Problema | Sintoma | Correção |
|---|---|---|
| `hotwords` / `initial_prompt` no Whisper | frases inteiras trocadas por listas de termos do prompt | não usar; o consenso detectou todos os casos |
| Modo em lote do WhisperX | segmentos de ~30 s sem pontuação nem maiúsculas | pontuação emprestada do Parakeet em palavras idênticas |
| LLM como desempate do consenso | jargão errado: "pec"→"peck", "Pendlay"→"pen lay" | sem LLM: grafias agrupadas, glossário do vídeo, maioria com fonte independente |
| Maioria 2-1-1 enviada ao desempate | "raise"→"rays", "delt is"→"delta" | maioria simples decide quando há apoio independente |
| LLM 8B extraindo justificativas | 624 de 983 "justificativas" eram frases copiadas da fala | modelo maior + itens copiados viram citação literal |
| LLM 30B resumindo | dicas genéricas inventadas ("controle a fase excêntrica") | `verify_support` com evidência literal conferida |
| Capítulos curtos | frase do capítulo seguinte vazava para o anterior | frase pertence ao capítulo do seu ponto médio |
| Tier por regra fixa | "quase S, mas vou baixar para A+" virava S | pontuação de contexto (decisão > hipótese); gabarito 41/41 |
| LLM escolhendo o tier | 35/41 no gabarito, pior que a regra | descartado |
| Encerramento como exercício | "there were no F tier exercises" virou tier F | títulos de introdução/encerramento ignorados |
| Repetição infinita no LLM | ~157 mil caracteres, JSON inválido, 24 min presos | `num_predict`, `repeat_penalty`, novas tentativas |
| Troca de idioma | resposta em português com saída pedida em inglês | detecção por item + nova tentativa |
| Vínculo fala↔estudo pelo LLM | estudo agudo de EMG ligado a "estudo de 9 semanas" | confirmação determinística (números com unidade, termos raros) |
| Nome do melhor exercício | "Faceaway" ≠ "Face Away" | comparação sem espaços/hífens |
| Monitor de memória da sessão | processo morto com 27 GB livres (cache de disco contado como uso) | `run --isolate`; processo desvinculado; vigia próprio de `MemAvailable` |
| Tradutor anotando dúvidas | "[trecho ambíguo no original]" dentro do texto | limpar a origem antes de traduzir; revisor independente |

Números da playlist de referência (7 vídeos, ~100 min): concordância WhisperX×Parakeet 98,9–99,8%, 8 trechos
incertos no total, 160 exercícios, 21 estudos identificados.
