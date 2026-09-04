# Etapa 19 — comparação dos braços físicos HAA e sem HAA em 738M tokens

> **STATUS EM 04/09/2026: CONCLUÍDA.** Em checkpoints com exatamente
> **738.197.504 tokens**, o braço físico sem HAA teve NLL menor nas sete fontes,
> venceu 34/35 janelas de 2k e 21/21 janelas de 8k, além de ser 1,34x mais
> rápido no treino real da Arandu. A vantagem ponderada foi **−0,062807
> nat/token** em 2k (IC bootstrap 95% **[−0,069170; −0,056753]**) e
> **−0,079803 nat/token** em 8k (IC **[−0,095131; −0,066174]**), sempre na
> convenção `sem HAA − HAA`, em que negativo favorece remover a HAA. A ablação
> dentro do checkpoint HAA mostrou que seu efeito instantâneo é essencialmente
> zero. A decisão prática é continuar o braço sem HAA como **baseline de
> linguagem**, não adotá-lo como arquitetura final de memória. A diferença entre os
> treinos, porém, não é um estimador causal perfeito: o HAA começou em 5 GPUs e
> terminou com 125 updates a menos, deixando fluxo de dados e batch global
> diferentes nos primeiros 327,68M tokens.

## 1. Pergunta

A Etapa 18 desligou a HAA em um modelo que havia sido treinado com ela. Isso
respondeu se a HAA ajudava no *forward* daquele checkpoint, mas deixou aberta a
hipótese do andaime: talvez a HAA fosse útil durante o treino e dispensável
depois. Esta etapa preenche parte desse quadrado treinando desde zero uma
arquitetura que não contém projeções, parâmetros, estado rápido ou custo de
otimizador da HAA.

As perguntas são:

1. com o mesmo número de tokens, qual braço generaliza melhor?;
2. a HAA passa a ajudar quando o contexto ultrapassa o bloco LOCAL de 64
   tokens?;
3. o checkpoint HAA demonstra memória de entidade mensurável?;
4. a eventual qualidade compensa o custo paramétrico e temporal?;
5. os modelos já geram respostas utilizáveis como chat?

## 2. Checkpoints e proveniência

| item | com HAA | sem HAA |
| --- | ---: | ---: |
| job | 12031 | 12062 |
| arquitetura | `celllm-v3-haa-phylo-600m` | `celllm-v3-local-only-498m` |
| parâmetros | 599.045.122 | 498.357.505 |
| `tokens_seen` | 738.197.504 | 738.197.504 |
| `next_step` | 1.283 | 1.408 |
| world final | 4 | 4 |
| micro-batch por rank | 64 | 64 |
| fingerprint | `a1778e0dfc79a652...` | `d2d1c1ab9605ce00...` |
| SHA-256 | `dcd1a9ff2eca4bdf...` | `1ab7a053f0d3250d...` |

Os dois arquivos passaram no teste estrutural do formato PyTorch e tiveram os
metadados lidos com `mmap=True`. O checkpoint sem HAA foi produzido por uma
retomada de 338 passos e salvo no marco exato
`checkpoint-tokens-738197504.pt`, sem rearmar o cosine: o horizonte continuou
em 5.491.124.663 tokens.

No job 12062, o retorno `-11` aconteceu somente depois das mensagens de
checkpoint permanente e `done: 1408 steps, 738,197,504 tokens`. Como o arquivo
está íntegro e foi carregado e avaliado ponta a ponta, isso não representa
falha do treino nem checkpoint parcial. A posição do erro é compatível com
falha no encerramento do processo distribuído/PyTorch-NCCL; os logs disponíveis
não permitem atribuir o componente exato, portanto essa causa permanece como
diagnóstico provável, não comprovado.

Código da avaliação: commit `d230ea06aee7dcabefd69c9cf37cd3544dfbff80`,
com mudanças não commitadas específicas desta etapa. `libPyCelNN`:
`1892eef2c614248bb5bba8746f12496966bbce4a`, versão 0.3.2. Ambiente local:
Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0, RTX 4060 Laptop 8 GB.

## 3. Protocolo

### 3.1 Qualidade natural em contexto 2k

- cinco janelas por cada uma das sete fontes, 35 unidades no total;
- 2.048 tokens por janela e 2.047 alvos;
- offsets fixados por NumPy PCG64, seed 180017;
- os dois braços recebem exatamente os mesmos IDs e offsets;
- cada modelo roda em processo novo para não contaminar memória CUDA;
- BF16 com autocast, carry nativo entre blocos de 64;
- agregação pelas proporções declaradas do treino: 30/20/15/13/12/5/5%;
- bootstrap estratificado dentro de fonte, 50 mil réplicas.

O desfecho primário é

\[
\Delta = \operatorname{NLL}_{sem\ HAA}-\operatorname{NLL}_{HAA}.
\]

Logo, `Δ < 0` favorece o braço sem HAA.

### 3.2 Extrapolação para contexto 8k

O mesmo protocolo foi repetido com três janelas por fonte, 21 no total, e
8.192 tokens por janela. É um teste deliberadamente fora da distribuição de
comprimento: o treino usou 2.048 tokens. Ele não estima o desempenho normal,
mas oferece à memória global sua melhor oportunidade de demonstrar utilidade.

### 3.3 Ablação causal no checkpoint HAA

Nas mesmas 35 janelas de 2k, o checkpoint HAA foi executado em quatro caminhos:
GLOBAL+LOCAL, apenas LOCAL, apenas GLOBAL e nenhum associativo. Os pesos são os
mesmos; somente os caminhos computacionais mudam. Esse contraste não sofre do
confundidor entre trajetórias de treino.

### 3.4 Memória de entidade e geração

O probe declara `my name is X`, insere filler real do corpus e mede
`log P(X | introduzido como X) − log P(X | introduzido como Y)`. Foram usados
12 pares nas distâncias 0, 128, 512, 2.048 e 4.096, com controle GLOBAL-off e
correção familiar de Bonferroni.

Na geração, seis prompts foram usados nos dois modelos, com top-k 40,
temperatura 0,7, máximo de 64 tokens e seed reiniciada por prompt. Geração é
evidência qualitativa secundária, não teste de qualidade factual.

## 4. Resultado primário: 2.048 tokens

| métrica ponderada | HAA | sem HAA | diferença sem HAA − HAA |
| --- | ---: | ---: | ---: |
| NLL, nat/token | 4,742863 | **4,680056** | **−0,062807** |
| bits/token | 6,842505 | **6,751894** | **−0,090611** |
| perplexidade | 114,7623 | **107,7762** | −6,09% |
| top-1 | 22,0216% | **22,6210%** | **+0,5994 pp** |

- IC 95% da diferença de NLL: **[−0,069170; −0,056753]**;
- IC 95% da diferença de top-1: **[+0,4613; +0,7476] pp**;
- sem HAA venceu 34 das 35 janelas naturais.

### 4.1 Resultado por fonte

| fonte | peso | Δ NLL |
| --- | ---: | ---: |
| FineWeb-Edu | 30% | −0,056053 |
| Tulu-3 | 20% | −0,072747 |
| Cosmopedia | 15% | −0,084155 |
| Smol-SmolTalk | 13% | −0,062582 |
| FineWeb2-PT | 12% | −0,048027 |
| PeS2o | 5% | −0,041234 |
| Wikipedia EN | 5% | −0,057150 |

O sinal não depende de uma fonte ou idioma. As duas fontes de diálogo também
favorecem o braço sem HAA: Tulu-3 em −0,0727 e Smol-SmolTalk em −0,0626 nat.

### 4.2 Resultado por posição

| posição | Δ NLL | IC bootstrap 95% |
| --- | ---: | ---: |
| 0–63 | −0,065065 | [−0,097073; −0,039330] |
| 64–127 | −0,058406 | [−0,080573; −0,033016] |
| 128–255 | −0,071991 | [−0,090371; −0,052215] |
| 256–511 | −0,069575 | [−0,084011; −0,055200] |
| 512–1.023 | −0,060859 | [−0,070580; −0,051330] |
| 1.024–2.047 | −0,061073 | [−0,067908; −0,054186] |

Não aparece uma faixa em que a HAA reverta o resultado.

## 5. Contexto 8k

| métrica ponderada | HAA | sem HAA | diferença |
| --- | ---: | ---: | ---: |
| NLL, nat/token | 4,747500 | **4,667698** | **−0,079803** |
| bits/token | 6,849195 | **6,734064** | **−0,115131** |
| perplexidade | 115,2957 | **106,4524** | −7,67% |
| top-1 | 21,1129% | **21,7542%** | **+0,6413 pp** |

- IC 95% da diferença de NLL: **[−0,095131; −0,066174]**;
- IC 95% da diferença de top-1: **[+0,4871; +0,7800] pp**;
- sem HAA venceu 21/21 janelas;
- entre 2.048–4.095: Δ = −0,058399 nat;
- entre 4.096–8.190: Δ = **−0,098672 nat**, IC
  **[−0,125880; −0,075813]**.

Mesmo onde só a HAA poderia carregar estado entre blocos, ela não recupera o
custo: a desvantagem aumenta na última metade do contexto.

## 6. Ablação dentro do HAA de 738M

| contraste | estimativa | IC bootstrap 95% | leitura |
| --- | ---: | ---: | --- |
| full − GLOBAL-off | +0,000045 | [−0,000009; +0,000098] | HAA indistinguível de zero |
| full − LOCAL-off | −0,645662 | [−0,676603; −0,615696] | LOCAL reduz fortemente a NLL |

O GLOBAL-off preserva praticamente a NLL completa: 4,742818 contra 4,742863.
Remover o LOCAL eleva a NLL para 5,388525 e a perplexidade para 218,88. Assim,
o mecanismo associativo que trabalha é novamente o LOCAL, não a HAA global.

Esse resultado é mais conservador que o checkpoint de 1,148B da Etapa 18,
onde a HAA já piorava a NLL em +0,007603 nat. A leitura conjunta é que ela vai
de inerte em 738M a levemente prejudicial mais tarde.

## 7. Memória de entidade

| distância | efeito causal da HAA | IC 95% | IC familiar de Bonferroni |
| ---: | ---: | ---: | ---: |
| 0 | 0 | [0; 0] | [0; 0] |
| 128 | +0,001351 | [+0,000274; +0,002884] | [+0,000132; +0,003458] |
| 512 | +0,000284 | [−0,000093; +0,000744] | [−0,000178; +0,000905] |
| 2.048 | −0,001845 | [−0,008307; +0,003102] | [−0,010697; +0,004407] |
| 4.096 | +0,000660 | [−0,000111; +0,001574] | [−0,000280; +0,001887] |

Há um efeito estatisticamente detectável em 128 tokens, mas sua magnitude é
aproximadamente 0,00135 em log-probabilidade e desaparece já em 512 tokens.
Não é memória global funcional em escala relevante.

## 8. Curva de treino e custo

Existem 31 registros com contador de tokens coincidente entre 340.787.200 e
734.003.200. Sem HAA teve loss menor em 26/31, diferença média descritiva
−0,127970 nat e mediana −0,083982. O bootstrap IID descritivo dá
[−0,191994; −0,076714], mas não deve ser tratado como inferência confirmatória:
as minibatches de treino não são pareadas.

No intervalo coincidente da Arandu:

| medida | HAA | sem HAA | razão sem HAA/HAA |
| --- | ---: | ---: | ---: |
| throughput mediano | 10.153 tok/s | 13.599 tok/s | **1,339x** |

A ablação temporal local na RTX 4060 mediu speedup GLOBAL-off de 1,441x no
forward de 2k, 1,425x na geração autoregressiva e 1,389x em forward+backward.
Fisicamente, remover a HAA elimina 100.687.617 parâmetros, 16,81% do modelo,
e cerca de 29,88% dos FLOPs matriciais dominantes estimados na Etapa 18.

O `tokens_per_second` de `run-summary.json` não é confiável em retomadas: ele
divide `tokens_seen` cumulativo pelo tempo do segmento atual. Os valores acima
vêm das janelas de `metrics.jsonl`.

## 9. Geração

Uma auditoria posterior descobriu que a primeira rodada de geração desta etapa
era metodologicamente inválida: depois do primeiro token, o decodificador
passava somente o token recém-gerado ao modelo. Isso remove o contexto do
LOCAL e reduz o braço sem HAA a contexto efetivo de um token. O artefato
`stage19-matched-738m-generations.json` foi preservado para auditoria, mas está
**superado e não deve ser usado para julgar qualidade**.

O gerador foi corrigido para recomputar o prefixo nos blocos nativos de 64
tokens, reproduzindo a geometria do treino. Cinco prompts foram repetidos em
greedy e com `temperature=0,7`, `top-k=40`. Nenhum braço respondeu corretamente
às perguntas factuais, aritméticas ou de nome. Exemplos amostrados:

A mesma correção foi aplicada a `chat_celllm.py`: o CLI agora preserva o
histórico visível e reconstrói o estado nos limites de bloco a cada geração.

- HAA, pergunta imediata sobre Mariana: `Descrição a piorúncia que eu fala do meu filho.`;
- sem HAA, mesma pergunta: `Já-me o dia 24 dia. de novembro...`;
- HAA, pergunta distante: começa com `Absolute...` e inventa `Bob` no greedy;
- sem HAA, pergunta distante: repetição de `Absolutely`, sem recuperar Mariana.

Há fragmentos sintáticos mais longos que na rodada defeituosa, mas continuam
incoerentes, repetitivos, misturando idiomas e sem obedecer às instruções. Não
há vantagem qualitativa convincente de nenhum braço.

Portanto, a redução de NLL é real, mas **não significa que o modelo já tenha
qualidade conversacional**. São necessários mais pré-treino e depois SFT/eval
específica; marcadores de papel no corpus não substituem alinhamento.

### 9.1 Confirmação por inferência e memória física dos dois braços

Uma rodada adicional em 04/09/2026 executou cinco prompts idênticos nos dois
checkpoints físicos, incluindo duas versões de “meu nome é Mariana”. As
inferências corrigidas acima confirmam que nenhuma resposta recupera o nome.

Também repetimos o probe contrafactual de nome diretamente nos dois modelos,
agora declarando o nome uma única vez. O ganho é
`log P(A | introduzido A) − log P(A | introduzido B)`:

| filler entre declaração e pergunta | HAA | sem HAA |
| ---: | ---: | ---: |
| 0 | +0,368827 | +0,464316 |
| 32 | −0,000004 | +0,000002 |
| 64 | +0,000591 | **0 exato** |
| 128 | +0,001127 | **0 exato** |
| 512 | +0,000356 | **0 exato** |

No caso imediato, ambos são sensíveis ao nome porque a declaração e a pergunta
estão no mesmo bloco. Isso ainda não vira resposta: a probabilidade média do
nome correto é 0,0072% com HAA e 0,0041% sem HAA, a mediana de rank é 3.003 e
3.447, e a acurácia top-1 é zero nos dois.

Com apenas 32 tokens de filler o sinal já é indistinguível de zero nos dois.
Nos comprimentos 64, 128 e 512, o braço sem HAA dá zero por construção: seu
`memory_state` é `None`, o estado CelNN é reinicializado em cada chamada e o
LOCAL é estateless e restrito ao bloco nativo de 64. O pequeno sinal HAA nas
distâncias maiores tem IC 95% incluindo zero nesta rodada e é pequeno demais
para recuperar o nome. Isso confirma a distinção crítica:

- **sem HAA:** não possui qualquer canal de memória entre blocos;
- **com HAA:** possui o canal, mas não aprendeu memória conversacional útil;
- **LOCAL:** ajuda muito a modelagem de linguagem dentro do bloco, mas não é
  memória de conversa persistente.

Logo, remover a HAA atual é correto para o baseline eficiente, mas **não resolve
o requisito original de memória da CellLM**. Se carregar fatos ditos antes é
requisito central, o braço sem HAA não pode ser o produto final sem receber um
novo mecanismo de estado treinável e testado.

## 10. Confundidor descoberto

O pareamento é exato em tokens, arquitetura compartilhada, seed inicial,
schedule por tokens e janelas de avaliação. Não é exato em trajetória de
otimização:

- o checkpoint HAA de 327.680.000 tokens registra world size 5;
- ele foi retomado com 4 GPUs até 738.197.504;
- o braço sem HAA usou 4 GPUs desde o início;
- HAA termina no update 1.283; sem HAA, no 1.408;
- os quatro estados PCG64 do sampler são diferentes nos checkpoints finais.

Logo, no primeiro trecho o HAA usou batch global de 655.360 tokens e o sem HAA
524.288. O sem HAA recebeu **125 updates adicionais**, 9,74% em relação ao
HAA, e viu uma sequência amostral diferente. O schedule de LR continuou justo,
pois depende de tokens, mas número de updates e ruído do gradiente podem
explicar parte da vantagem física de −0,0628 nat.

Os intervalos bootstrap quantificam somente a variação das janelas fixas,
condicional a estes dois checkpoints. Eles não incluem variância entre seeds
ou entre trajetórias de treino. Há apenas uma seed por braço.

Esse confundidor não afeta a ablação intracheckpoint: nela os pesos, dados e
trajetória são idênticos. Por isso é seguro concluir que a HAA em si está
inerte no forward de 738M. A magnitude exata do benefício de *treinar* sem HAA
ainda exigiria controle com world size e batch global idênticos desde zero.

## 11. Conclusão e decisão

Os três eixos apontam na mesma direção:

1. **utilidade instantânea:** HAA ≈ zero em 738M e piora em 1,148B;
2. **treino físico:** sem HAA generaliza melhor em 2k e 8k, embora a magnitude
   esteja parcialmente confundida por batch/updates;
3. **custo:** HAA adiciona 16,81% dos parâmetros e reduz throughput real em
   aproximadamente 25,3% (`1 − 1/1,339`).

Não há evidência positiva para manter a HAA atual. A hipótese de andaime não é
apoiada pelos dados; no máximo, sua rejeição causal estrita fica limitada pelo
confundidor de trajetória. A decisão de engenharia é **prosseguir sem HAA como
baseline linguístico eficiente** e preservar o LOCAL. Isso não equivale a
declarar resolvida a arquitetura de memória: o braço sem HAA é estruturalmente
incapaz de transportar informação entre blocos.

Próximos passos recomendados:

1. continuar o braço sem HAA até 1.148.190.720 tokens, para o marco planejado
   e comparação com o checkpoint HAA já existente;
2. repetir a avaliação natural e a geração nesse marco;
3. não completar 5,491B nem iniciar SFT antes de verificar melhora substancial
   de geração no checkpoint de 1,148B;
4. se a tese exigir a magnitude causal de treino, repetir em escala menor ou
   com pelo menos três seeds e batch/world idênticos desde o token zero;
5. antes de promover o modelo a arquitetura de chat, redesenhar a memória
   global: gate de escrita/utilidade, controle de colisão e teste de recuperação
   obrigatório; validar primeiro em escala pequena, exigindo recuperação
   causal acima de 64, 512 e 2.048 tokens antes de outro treino caro.

## 12. Artefatos

- `etapa-19-artefatos/stage19-matched-738m-context2048.json` — comparação
  confirmatória 2k;
- `etapa-19-artefatos/stage19-matched-738m-context8192.json` — extrapolação 8k;
- `etapa-19-artefatos/stage19-haa738-within-checkpoint-ablation.json` — ablação
  causal GLOBAL/LOCAL;
- `etapa-19-artefatos/stage19-haa738-entity-binding.json` — memória de entidade;
- `etapa-19-artefatos/stage19-matched-738m-generations.json` e
  `stage19-physical-inferences-greedy.json` — gerações antigas preservadas,
  mas invalidadas pela geometria incremental incorreta;
- `etapa-19-artefatos/stage19-physical-inferences-blockwise-greedy.json` e
  `stage19-physical-inferences-blockwise-sampled.json` — inferências corrigidas
  por recomputação do prefixo em blocos nativos;
- `etapa-19-artefatos/stage19-physical-memory.json` — binding contrafactual
  adicional nos dois checkpoints físicos;
- `etapa-19-artefatos/stage19-matched-training-curve.json` — curva coincidente;
- `etapa-19-artefatos/haa-738m/` e `nohaa-738m/` — checkpoints e telemetria;
- `etapa-19-manifesto-sha256.txt` — hashes dos artefatos principais.
