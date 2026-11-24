# Código dos replays

Estes snapshots preservam os executores históricos necessários à reexecução dos experimentos arquivados. Cada diretório inclui a licença Apache-2.0 da origem. O interpretador de cada alvo importa seu próprio snapshot CelLM e usa o pacote `celnn` fixado no lock do projeto.

| Diretório | Proveniência |
| --- | --- |
| `text8/` | `CelNN-Text-Generation`, revisão `0ff1fb02b6cf65ead6a75562a15f4de8c10beca5`, diretório `src/celllm`. O notebook publicado usou libPyCelNN na revisão `1685319256e5f167ea1fef9d5f9e48a7e8b69578`. O replay local usa a API compatível de `celnn==1.0.0`. |
| `d4/` | Subconjuntos disponíveis de `experiment-chat-good-4M/d20` e `experiment-mixer-replication/src/celllm`, ambos de `CelNN-Text-Generation`. |
| `lamp/` | Subconjunto de `CelNN-Text-Generation/src/celllm`, ferramentas de avaliação e configuração de `experiment-celllm-v3-600m-chat`; revisão base `d230ea06aee7dcabefd69c9cf37cd3544dfbff80` com alterações do snapshot histórico. |

O núcleo dos experimentos novos fica em `src/cellm_poc/`.
