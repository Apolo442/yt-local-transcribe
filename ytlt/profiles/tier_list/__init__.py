"""Perfil tier_list: vídeos que ranqueiam itens em tiers (S, A, B, C, D, F).

- tier lido da fala com pontuação de contexto ("close to S tier, but… A tier");
- justificativas/dicas resumidas pelo LLM e verificadas contra o capítulo;
- ênfase muscular (cabeça/porção) só quando dita no capítulo, com citação verificada;
- afirmações classificadas (estudo citado, biomecânica, opinião, experiência);
- Markdown por vídeo + índice; export JSON bilíngue para um web app.
"""

from .analyze import analyze
from .emphasis import build as emphasis
from .export import build as export
from .render import render_index, render_video as render
from .units import units

__all__ = ["analyze", "emphasis", "export", "render", "render_index", "units"]
