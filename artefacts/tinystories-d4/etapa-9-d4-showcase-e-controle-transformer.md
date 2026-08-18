# Etapa 9 — D4: showcase escalado e controle Transformer

**Datas:** 2026-08-17 a 2026-08-18
**Hardware:** RTX 4060 Laptop (8 GB), 32 GB RAM · conda `ag`, torch 2.13.0+cu130
**Código:** `code/CelNN-Text-Generation/experiment-chat-good-4M/`

> **ERRATA (2026-08-20).** A seção 11 lista `cellm_window64` como incompleto por
> OOM em `batch=2`, e isso foi repetido como prova de que o eixo `n` estava bloqueado
> nesta placa. **Está errado.** O preflight da
> [`etapa-13`](etapa-13-escada-de-n.md) mediu a mesma configuração cabendo em 5,79 GB
> (71 % da placa). O OOM original ocorreu depois da avaliação do passo 0, dentro do
> laço de treino, com 13,25 MiB livres, e `window32` havia rodado antes no mesmo
> processo: era memória não reclamada, não exigência de `n=64`.

> **ERRATA (2026-08-21) — `mixer=dense`.** Toda menção a `mixer=dense` neste
> documento descreve uma **imposição do código**, não uma escolha de projeto:
> `src/celllm/model.py` executava `replace(cfg, mixer="dense")` em quatro classes,
> descartando silenciosamente o mixer pedido. O eixo era código morto desde a
> Etapa 3. Ver seção 2 de
> [`etapa-14`](etapa-14-mixer-largura-e-raio-hebbiano.md).

---

## 0. Natureza desta etapa

O D4 **não é um experimento causal da linha B0.x.** É uma branch de demonstração
cujo objetivo era escalar a arquitetura validada e ver se ela vira um chatbot.
Usa cronograma pragmático e muda várias coisas entre segmentos.

Duas coisas o tornaram cientificamente relevante além do showcase:

1. produziu o **primeiro controle Transformer pareado em parâmetros** da linha;
2. reproduziu, em escala 3× maior, os achados de esquecimento do B0.2.

Os números do D4 não devem ser misturados às conclusões causais do B0.x.

> Nota de nomenclatura: os artefatos em disco dizem `d20_showcase`, `d20_a_language`,
> etc. O alvo original era 20M parâmetros; foi reduzido para ~3,5M por restrição de
> VRAM (secção 2) e os nomes não foram renomeados para não quebrar o `resume` de um
> treino em curso. **O modelo tem 3.337.865 parâmetros, não 20M.**

---

## 1. Resumo executivo

| Pergunta | Resposta |
| --- | --- |
| A CellLM escalada aprende linguagem? | **SIM** — BPC 3,1237 → 1,0392 em 50M tokens |
| Vira um chatbot? | **NÃO** — respostas com forma de assistente e conteúdo incoerente |
| Empata com um Transformer de mesmo tamanho? | **NÃO** — 1,0392 vs 0,7603 (37% pior) |
| Usa menos memória que o Transformer? | **SIM** — 0,179 GB vs 0,610 GB em 8k tokens (3,4×) |
| Chain-of-thought funciona? | Formato sim (98%), raciocínio não (48%) |

---

## 2. Arquitetura: a escolha e a restrição

### 2.1 Não há knob de profundidade

Inspecionando a implementação: **uma única célula compartilhada aplicada `k` vezes.**
Não há blocos empilháveis. Capacidade escala apenas por largura (`d`), vocabulário e
largura de memória.

Consequência: um alvo de 20M viraria 63–76% tabela de embedding. Registro da busca em
`architecture_search.json`.

### 2.2 O critério mudou durante a busca

O critério inicial era `min |N − 20M|`. Ele escolheu `d=2048, vocab=6144, mem=192`
(19,95M, **63,1% embedding**). Revisado para lexicográfico:

1. total dentro da faixa;
2. estado plástico por sequência dentro do teto;
3. **minimizar `embedding_fraction`**;
4. desempate por distância ao alvo.

A restrição (2) foi acrescentada depois de medir que o estado plástico cresce com
`memory_size²` e é FP32 por protocolo: `mem=512` daria 40% de embedding mas **16,81 MB
por sequência**, contra 4,21 MB de `mem=256`.

### 2.3 A largura, não o número de parâmetros, limita o batch

Medido em seq 256:

| d | mem | params | batch máx | pico VRAM | tok/s | 50M em |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2304 | 256 | 19.493.641 | 1 | 5,00 GB | 112 | 124 h |
| 1024 | 96 | 3.945.481 | 4 | 3,44 GB | 391 | 35,5 h |
| **896** | **96** | **3.337.865** | **8** | 6,65 GB | **~1000** | **~13 h** |
| 512 | 64 | 2.631.177 | 16 | 6,50 GB | 1504 | 9,2 h |

As ativações atravessam os 15 passos de dinâmica, então escalam com `d`. `d=1024` tem
18% mais parâmetros que `d=896` e leva **2,7× mais tempo**. Por isso o alvo caiu de
20M para 3,5M com teto empírico `d ≤ 896`, declarado como propriedade desta placa e
não da arquitetura.

### 2.4 Configuração congelada

```
d=896, cells=16, r=1, k=15, vocab=2048, mem=96, mixer=dense
BANK + LOCAL-UNGATED-R2

parâmetros treináveis     3.337.865
  embedding               1.837.056   55,0%   (tabela compartilhada, weight tying)
  projeção (mixer d²)       802.816   24,1%
  memória (bank + local)   ~696.000   ~20,9%
  núcleo Chua-Yang            4.480    0,13%  aplicado 15× por token
estado plástico              0,60 MB/sequência
```

O `state_dict` soma 5.174.409 porque inclui `readout.weight` (1.835.008), que é
*weight tying* e não parâmetro independente, mais o buffer `slot_addresses`.

**A dinâmica CelNN é 0,13% do armazenamento mas participa de toda a computação.**
A arquitetura escala compute por reuso temporal, não por empilhar camadas.

---

## 3. D4-A: pré-treino de linguagem

### 3.1 O run contaminado (descartado, preservado como evidência)

Um teste de validação em escala `smoke` deixou `language_raw.jsonl` (3.000 docs) e
`chat_raw.jsonl` (2.000) em cache. A célula de download reusava o arquivo se ele
**existisse**, sem verificar o tamanho. O run de produção reusou o corpus de smoke:

```
tokens únicos no corpus : ~0,7M
orçamento treinado      : 50M tokens  ->  ~75 ÉPOCAS
```

Trajetória (`etapa-9-d4-artefatos/contaminado-linguagem.jsonl`):

| tokens | BPC |
| ---: | ---: |
| 2M | 1,6124 |
| 5M | 1,3795 |
| 10M | 1,2173 |
| 20M | 1,1247 |
| **35M** | **1,1013** ← melhor |
| **50M** | **1,1442** ← regrediu |

14 horas de GPU perdidas. O sinal do overfitting (BPC subindo) foi reportado por mim
como "curva saudável" antes de eu olhar direito — falha de método registrada na
secção 8.

### 3.2 O run limpo

Corpus real: 200.000 documentos TinyStories (~47M tokens únicos) → 1,06 épocas.

| tokens | NLL | PPL | BPC | Δ |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 7,6240 | 2046,83 | 3,1237 | — |
| 2M | 4,2667 | 71,29 | 1,7481 | — |
| 5M | 3,6227 | 37,44 | 1,4843 | −0,264 |
| 10M | 3,2380 | 25,48 | 1,3267 | −0,158 |
| 20M | 2,9614 | 19,32 | 1,2133 | −0,113 |
| 35M | 2,7011 | 14,90 | 1,1067 | −0,107 |
| **50M** | **2,5363** | **12,63** | **1,0392** | −0,068 |

**A curva não saturou.** Retornos decrescentes, mas ainda negativos em 50M.

### 3.3 Contaminado vs limpo: o cruzamento

| tokens | limpo | contaminado | delta |
| ---: | ---: | ---: | ---: |
| 2M | 1,7481 | 1,6124 | +0,1357 |
| 5M | 1,4843 | 1,3795 | +0,1048 |
| 10M | 1,3267 | 1,2173 | +0,1094 |
| 20M | 1,2133 | 1,1247 | +0,0886 |
| **35M** | **1,1067** | 1,1013 | **+0,0054** |
| **50M** | **1,0392** | 1,1442 | **−0,1050** |

O contaminado ia melhor até 35M (decorava mais rápido) e então **regrediu**. O limpo
cruzou em 35M e continuou caindo. Ressalva: held-outs diferentes (um sorteado de 3.000
docs, outro de 200.000), então a comparação é indicativa, não medição controlada.

**Consequência implementada:** gate bloqueante que aborta acima de 4 passes sobre o
corpus, e `reusable_download()` que compara o manifesto do cache com o pedido.

---

## 4. D4-B: bootstrap comportamental e o custo do esquecimento

Mesmo currículo procedural, mesmo orçamento (300k tokens de chat), duas configurações:

| tokens | **com replay 25%, lr 1e-4** | | sem replay, lr 3e-4 | |
| ---: | ---: | ---: | ---: | ---: |
| | inglês (PPL) | comport. | inglês (PPL) | comport. |
| 0 | 12,35 | 0,021 | 18,8 | 0,035 |
| 25k | 15,42 | 0,170 | 2.064 | 0,220 |
| 75k | 14,80 | **0,305** | 43.649 | 0,128 |
| 150k | 13,86 | 0,284 | 941.717 | 0,085 |
| 225k | 13,50 | 0,248 | 68.471 | 0,163 |
| **300k** | **13,02** | 0,262 | **825.301** | 0,156 |

**Degradação de 63.000×** sem replay. Com replay, o inglês partiu de 12,35 e terminou
em 13,02 — praticamente intacto, **e melhorando ao longo do treino**.

O comportamento também foi melhor com replay (pico 0,305 vs 0,220). **Não houve troca
de um eixo pelo outro.** Isso replica o B0.2 numa escala 3× maior e confirma o 2×2 do
B0.3b: o custo de capacidade vinha da taxa de aprendizado reduzida, não do replay.

Nota: o pico comportamental está em 75k tokens, não no fim. O código seleciona por
comportamento (`best_behavior.pt`), não por loss — a NLL de chat continuou caindo
(0,714 → 0,486) enquanto o comportamento oscilava.

---

## 5. D4-C: SFT de chat em escala

Partindo de `best_behavior.pt`, dois segmentos com smol-smoltalk filtrado.

| segmento | tokens | chat NLL | inglês (PPL) | comport. |
| --- | ---: | ---: | ---: | ---: |
| 001 (replay 10%) | 1.000.403 | 3,8473 | 16,38 | **0,064** |
| 002 (replay 25%) | 2.000.336 | 3,4329 | 14,04 | **0,085** |

**Dez vezes mais tokens de conversa piorou o comportamento em 3×** frente ao bootstrap
(0,26 → 0,085). O smol-smoltalk tem perguntas técnicas de adulto; o salto vindo de
historinhas infantis é grande demais para 3,34M de parâmetros.

**Mais dados de tipo errado é pior que menos dados de tipo certo.**

Gerações finais, modo determinístico:

```
"Hello!"                -> "Hello! How can I help you today?"        <- formato ok
"What is your name?"    -> "I'm sorry for the tonight. I'm Don't..."  <- incoerente
"Give me three fruits." -> "Welcome to my friend, I'm Don't forget..." <- incoerente
```

Adquiriu o **tom** de assistente sem a **substância**.

### 5.1 Filtro do smol-smoltalk

Yield medido e limite escolhido por dado, não por palpite:

| limite de chars | mantidos | tokens/diálogo |
| ---: | ---: | ---: |
| 400 | 6,6% | 90 |
| **600** | **12,3%** | **145** |
| 800 | 18,2% | 205 |
| 1200 | 30,6% | 387 |

Acima de ~600 as respostas deixam de ser réplicas curtas demonstráveis. Também:
**16% dos diálogos abrem com turno `system`** (removido, não rejeitado) e a mediana de
resposta é ~1.290 caracteres — daí o truncamento em vez de descarte.

---

## 6. D4-D: chain-of-thought

Currículo de 360 exemplos, 6 tarefas balanceadas, 250k tokens, marcadores em texto
puro (`Thinking:` / `Answer:`) — **sem tokens especiais novos**, para não redimensionar
a tabela de embedding.

| métrica | valor |
| --- | ---: |
| aderência ao formato | **98%** (41/42) |
| coerência trace↔resposta | **86%** (18/21) |
| **resposta correta** | **48%** (20/42) |
| acerto quando trace e resposta concordam | 50% (9/18) |
| acerto quando se contradizem | **0%** (0/3) |
| inglês preservado | 14,37 → 15,07 |

Por tarefa: comparação 5/7, contagem 4/7, patas 4/7, cor 3/7, verdade 3/7,
**soma 1/7**.

### 6.1 O que isso significa

O modelo aprende a escrever passos **e a ser consistente com eles**. Quando trace e
resposta se contradizem, o acerto é zero — a coerência importa. Mas mesmo quando
concordam, metade está errada: **o erro está dentro do raciocínio**, não na cópia.

### 6.2 Fragilidade a fraseado

```
"Which is bigger, a mouse or a cat?"   -> "A cat is bigger."   correto
"What is bigger, mouse or cat?"        -> colapso
"What is bigger, a mouse or a cat?"    -> pensou certo, respondeu errado
"What color is the sky?"               -> correto
"What is the color of the sky?"        -> correto
```

A tarefa de cor generalizou porque tinha **5 formas de pergunta**. As demais, com uma
forma só, não generalizaram apesar de 60 exemplos cada.

**Variei conteúdo mas não fraseado.** Erro de desenho, registrado na secção 8.

Também: 250k tokens sobre ~9k tokens únicos = **~27 épocas**. O gate de 4 épocas que
implementei cobre só a fase de linguagem — inconsistência registrada.

---

## 7. Controle Transformer pareado

### 7.1 Desenho

Tudo idêntico ao D4-A: mesmo tokenizer, corpus de 200.000 documentos, split 98/2,
orçamento de 50M, marcos, otimizador, lr 3e-4 e métrica BPC nos mesmos held-outs.

Duas formas, porque **profundidade é knob livre no Transformer e inexistente na
CellLM** — reportar só uma deixaria em aberto se a conclusão dependeu disso.

### 7.2 Resultado

| tokens | CellLM `n=16` | **TF wide** (d=240, L=4) | TF deep (d=144, L=12) |
| ---: | ---: | ---: | ---: |
| 2M | 1,7481 | **1,3781** | 1,5167 |
| 5M | 1,4843 | **1,1094** | 1,2243 |
| 10M | 1,3267 | **0,9550** | 1,0254 |
| 20M | 1,2133 | **0,8491** | 0,8864 |
| 35M | 1,1067 | **0,7907** | 0,8124 |
| **50M** | **1,0392** | **0,7603** | 0,7752 |

| | params | MFLOPs/token | horas | tok/s | BPC |
| --- | ---: | ---: | ---: | ---: | ---: |
| CellLM | 3.337.865 | 48,4 | **13,55** | 1.025 | 1,0392 |
| TF wide | 3.330.720 | 6,5 | **0,13** | 107.292 | **0,7603** |
| TF deep | 3.340.512 | 6,6 | 0,25 | 54.576 | 0,7752 |

**O Transformer é 36,7% melhor em BPC, com 7,4× menos FLOPs e 105× menos tempo.**
A vantagem existe desde o primeiro marco — não é convergência mais lenta.

### 7.3 O gap de tempo não é explicado pelos FLOPs

7,4× de computação não explica 105× de tempo. O restante (~14×) é **overhead de
lançamento de kernel**: 15 passos sequenciais sobre chunks de 16 células geram muitos
kernels minúsculos. **A CellLM é limitada por latência, não por computação.** Isso é
engenharia (fusão de kernels), não arquitetura.

### 7.4 Confundidor: contexto

O Transformer atende os **256 tokens** diretamente; a CellLM enxerga **16** e alcança o
resto só pela memória associativa. Histórias TinyStories têm ~200 tokens. É a diferença
arquitetural em teste, não um defeito a corrigir — mas precisa estar declarada.

---

## 8. Onde a CellLM ganha: memória em contexto longo

Inferência, batch 1, medido:

| tokens | CellLM | Transformer | vantagem |
| ---: | ---: | ---: | ---: |
| 256 | 0,05 GB | 0,04 GB | — |
| 1024 | 0,04 GB | 0,05 GB | 1,3× |
| 2048 | 0,06 GB | 0,08 GB | 1,3× |
| 4096 | 0,09 GB | 0,18 GB | 2,0× |
| **8192** | **0,16 GB** | **0,59 GB** | **3,7×** |

A memória da CellLM é praticamente **constante** — o estado associativo tem tamanho
fixo (`slots × 96 × 96`). O Transformer guarda chave/valor de cada token.

Tempo, no mesmo teste:

| tokens | CellLM | Transformer | razão |
| ---: | ---: | ---: | ---: |
| 256 | 351 ms | 1 ms | 351× |
| 2048 | 2.809 ms | 10 ms | 281× |
| 8192 | 11.887 ms | 109 ms | **109×** |

A razão **cai** com o contexto (o Transformer piora mais rápido), mas parte de tão
longe que o cruzamento ficaria em centenas de milhares de tokens.

**O trade-off é: a CellLM troca ~100–350× de latência por memória constante.**

Nota adicional: para suportar contexto de 8k, o Transformer precisou de **1,97M de
parâmetros só de tabela de posições** (8192 × 240), enquanto a CellLM precisa de zero.
Ressalva: isso é escolha da implementação (posição aprendida); com RoPE não pagaria.

---

## 9. A janela direta explica parte do gap

`n=32` contra o baseline `n=16`, mesmo orçamento, parâmetros a +0,05%:

| tokens | `n=16` | **`n=32`** | TF wide |
| ---: | ---: | ---: | ---: |
| 1M | 1,7481 | 1,7749 | 1,3781 |
| 2M | — | 1,5793 | 1,3781 |
| 3,5M | — | 1,4517 | — |
| **5M** | **1,4843** | **1,3825** | **1,1094** |

**Dobrar a janela fechou 27% do gap aos 5M, sem custo de memória de inferência**
(0,179 GB nos dois casos). O `n=32` atingiu aos 3,5M a qualidade que o `n=16` levou 5M
para alcançar — 30% menos dados.

Nos marcos intermediários, a fração do gap fechada foi 46% (2M) e 36% (3,5M) por
interpolação logarítmica.

### 9.1 Confundidor não resolvido

A restrição `k·r + 1 ≥ n` obriga a subir o **raio** junto com a janela:

| config | n | r | k |
| --- | ---: | ---: | ---: |
| baseline | 16 | 1 | 15 |
| testado | 32 | **2** | 16 |

São mecanismos diferentes: janela = quanto contexto está no bloco; raio = quantas
células a informação anda **por passo** (alcance por passo, com resolução mais grossa).

O teste de variável única seria `n=32, r=1, k=31` — não coube na GPU (OOM). O teste
complementar `n=16, r=2, k=15` isolaria o raio e **não foi executado**.

### 9.2 Correção registrada

No marco de 1M o `n=32` estava **pior** (1,7749 vs 1,7481) e eu reportei que "alargar
a janela não ajudou". Estava errado — era cedo demais, e a curva descolou a partir de
2M. A conclusão correta é a da tabela acima.

---

## 10. Falhas de método desta etapa

1. **Cache de dados sem validação** — 14h treinadas num corpus de smoke test. Causa
   raiz: a célula checava existência do arquivo, não tamanho. Eu tinha corrigido o
   mesmo padrão no tokenizer no dia anterior e não apliquei aos dados.
2. **Não olhei o sinal de overfitting** — reportei "curva saudável" com o BPC já
   subindo de 1,1013 para 1,1442.
3. **Contrariei minha própria evidência** — desliguei o replay no bootstrap com
   argumento qualitativo, contra um 2×2 (B0.3b) que já media o contrário.
4. **Variei conteúdo, não fraseado**, no currículo de raciocínio.
5. **Substituí em vez de somar** — o treino de raciocínio apagou a conversa geral.
6. **Erro de sintaxe no gerador do notebook** (`\n` literal em f-string) derrubou o run
   após D4-A e D4-B, sem rodar a checagem de AST que eu mesmo havia escrito.
7. **`pgrep` auto-detectando o próprio shell** nos scripts de vigia e lançamento.
8. **Estimativa de batch a partir de teste com batch menor** — o run de janela larga
   morreu com OOM em 5 minutos por não usar o autotune existente.
9. **Arquivo de evidência salvo no diretório errado** (`code/findings/` em vez de
   `findings/`), corrigido durante esta documentação.
10. **Gate de épocas aplicado só à fase de linguagem** — a fase de raciocínio rodou
    ~27 épocas sem aviso.

Correções implementadas: `reusable_download()` com manifesto, `assert` de passes,
`stratified_split` descartando estratos pequenos em vez de abortar, detecção de treino
por arquivo de PID, verificação de AST embutida no gerador do notebook, e checkpoint
genérico para arquiteturas não-CellLM. 45 testes.

---

## 11. Estado dos artefatos

| resultado | local |
| --- | --- |
| D4-A limpo (50M, BPC 1,0392) | `runs/d20_showcase/d20_a_language/` |
| D4-B com replay | `runs/d20_showcase/d20_b_bootstrap/` |
| D4-C chat + `REPORT.md` | `runs/d20_showcase/d20_c_chat/` |
| D4-D raciocínio | `runs/d20_showcase/d20_d_thinking/` |
| TF wide / TF deep | `runs/baseline_{wide,deep}/summary.json` |
| CellLM `n=32` | `runs/cellm_window32/summary.json` |
| Run contaminado (métricas) | `findings/etapa-9-d4-artefatos/` |

**Perdido:** os checkpoints do run contaminado e do bootstrap sem replay estavam em
`/tmp/d4-archive/`, que foi limpo. Só as métricas sobreviveram.

**Incompleto:** `cellm_window64` parou no marco zero (interrompido a pedido); o
`latest.pt` permite retomar.

---

## 12. Perguntas em aberto

1. **Janela ou raio?** Rodar `n=16, r=2, k=15` isola o raio. É o teste mais barato e
   mais informativo pendente.
2. **Transformer com janela 16.** Se ele ainda vencer vendo só 16 tokens, o que importa
   é *selecionar* (atenção), não *quanto se enxerga*. Custa ~10 minutos.
3. **`n=64` completo.** Diria se a tendência da janela satura.
4. **Passos de dinâmica como raciocínio latente.** Variar `k` (15 → 30 → 60) e medir se
   a acurácia sobe sem gastar tokens. Testa uma propriedade que o Transformer não tem;
   este relatório fornece a linha de base — *pensar em palavras, neste tamanho, dá 48%*.
5. **Fusão de kernels.** Os 105× de latência são majoritariamente overhead de
   lançamento, não FLOPs. Uma implementação fundida mudaria a análise de custo.
