# CelLM PoC · Pesquisa experimental

O CelLM é uma prova de conceito para investigar se redes neurais celulares aprendem padrões de texto. Os experimentos medem predição de caracteres e tokens, com controles convolucionais e Transformers, e avaliam a contribuição das mensagens locais. Um protocolo separado testa se preservar o campo entre blocos afeta predição, recall e custo de inferência.

## Ambiente e verificação

Linux x86-64, Python 3.11 ou 3.12, GNU Make e uv. O `uv.lock` fixa PyTorch 2.13, [`celnn[torch]==1.0.0`](https://pypi.org/project/celnn/1.0.0/), NumPy e Tokenizers. O padrão instala PyTorch para CPU. Para uma GPU compatível com CUDA 13, use `EXTRA=cuda DEVICE=cuda` nos comandos de execução. Também é possível apontar `PYTHON` para um ambiente existente com essas dependências.

```sh
make setup
make build
make test
make smoke
make -s benchmark CHECKPOINT=runs/smoke/42-carry.pt LENGTH=128
```

O build gera uma distribuição fonte e um wheel em `dist/`. O smoke gera corpus sintético local e percorre treinamento, validação, teste, intervenções e geração em escala reduzida. Use um novo `SMOKE_OUT` a cada execução. As métricas do smoke verificam o funcionamento do pipeline.

## Continuidade do campo celular

Forneça três JSONL de documentos independentes, cada linha com um campo `text` não vazio. Separe treino, validação e teste por documento antes da execução. O pipeline rejeita duplicatas entre partições após normalizar espaços. O teste deve conter pelo menos um documento longo o bastante para o filler de recall de 1.024 tokens. Use corpus em inglês para os prompts incluídos. O BPE é ajustado uma vez sobre o treino e compartilhado por todos os braços. Cada janela usa até 2.048 tokens de entrada, com alvos deslocados em um token e padding excluído da loss. Cada janela é uma sequência independente.

```sh
make continuity EXTRA=cuda DEVICE=cuda OUT=runs/continuity-v1 \
  TRAIN=data/train.jsonl VALIDATION=data/validation.jsonl TEST=data/test.jsonl
make benchmarks EXTRA=cuda DEVICE=cuda OUT=runs/continuity-v1
```

- `carry` transporta todo o campo final para o bloco de 64 posições seguinte. O gradiente é truncado a cada quatro blocos. `reset` treina desde o início com campo zerado a cada bloco. Dentro de cada semente, ambos recebem pesos iniciais idênticos e os três braços consomem os mesmos lotes, na mesma ordem.
- A avaliação final usa o checkpoint `carry` para medir separadamente o efeito de zerar o campo, o efeito de desabilitar LAMP e o efeito de repetir 32 tokens. No overlap, as duas condições pontuam os mesmos primeiros 32 alvos de cada fronteira elegível. O contraste entre os braços treinados separadamente permanece disponível no resumo por semente.
- Os marcos de 5, 20, 50 e 100 milhões contam alvos efetivamente usados na loss. Se um lote cruza um marco, a última atualização usa apenas o prefixo de alvos necessário para alcançá-lo. NLL e perplexidade de validação usam o mesmo conjunto em todos os marcos. O Transformer de controle usa duas camadas e a largura que aproxima a contagem de parâmetros do CelLM.
- A tarefa de recall usa uma associação declarada nos primeiros tokens, filler de um documento de teste e uma consulta posterior. Os valores de um token são sorteados depois de concluir o treino de todos os braços e as mesmas instâncias são usadas em todos os braços e sementes. Cada par muda só o valor inicial. O relatório guarda acurácia, probabilidades e o efeito da troca por distância.
- As amostras de geração usam os mesmos prompts, decodificação gulosa e orçamento. O benchmark usa batch 1, uma repetição de aquecimento e amostras sincronizadas em processos independentes, com comprimentos de 256, 2.048 e 8.192 tokens. Registra latência, throughput, pico de alocação CUDA e pico de RSS do processo. A medição de geração recompõe o prefixo nos dois modelos. A inspeção de coerência é feita sobre as amostras salvas.

`experiments/continuity.json` define a configuração de referência do experimento, com três sementes, largura 128, 16 passos, raio celular 4, raio LAMP 2, taxa AdamW de 0,0003 e os marcos de treino. O núcleo celular usa `DifferentiableCellularNetwork` do [projeto CelNN](https://github.com/delbem-research/celnn) para os templates diagonais, estêncil causal, ativação e integração de Euler. LAMP fornece o `extra_drive` em cada passo, e o CelLM controla a leitura e a continuidade entre blocos. Memória persistente, inicialização linguística e consumo de energia em dispositivos de borda são questões em aberto.

O diretório de saída recebe configuração efetiva, hashes dos arquivos de dados, tokenizer, hashes do código, ambiente, checkpoints finais, curva de treino, amostras e resultados por semente. `manifest.json` marca quando a execução terminou. O treinamento parte de uma nova saída a cada execução. Fixe os mesmos arquivos, lock, semente, dispositivo e threads ao comparar reexecuções.

## Registros e reexecuções

Os registros e insumos disponíveis estão em [`artefacts/`](artefacts/README.md). Os executores históricos ficam em [`historical/`](historical/README.md). Os alvos usam caminhos locais do checkout.

```sh
make verify-artefacts
make history
make d4-check
make lamp EXTRA=cuda
make text8 EXTRA=cuda DEVICE=cuda
```

Em um checkout novo, use Git LFS para hidratar `artefacts/text8/text8` e `artefacts/lamp/weights.pt`. Em um checkout de trabalho que já contém esses binários, configure Git LFS antes de adicioná-los ao índice.

| Avaliação | Insumos e verificação |
| --- | --- |
| Text8 | Corpus, notebook executado e relatório da Etapa 1. O alvo confere o MD5 oficial e executa sete variantes celulares e um controle convolucional, com três sementes e 20.000 passos. O job histórico usou A100 e uma revisão anterior da biblioteca CelNN. O replay local usa a versão publicada 1.0.0. [Origem do Text8](https://mattmahoney.net/dc/textdata.html) |
| TinyStories D4 | Curvas, sumários, manifesto e relatório da Etapa 9. `make d4-check` verifica as contagens de parâmetros dos três modelos. `make tinystories-d4 EXTRA=cuda DEVICE=cuda` requer os arquivos originais `language_raw.jsonl` e `tokenizer.json` em `artefacts/tinystories-d4/` e verifica seus hashes. O cache original ficou indisponível, e o snapshot recuperado do treinador e do Transformer tem hashes diferentes dos registrados no manifesto histórico. O BPC dos registros usa bytes UTF-8 no denominador. |
| LAMP | Pesos de inferência, 35 janelas de tokens e JSON primário da Etapa 19. O replay usa as janelas da avaliação original e ponderação por fonte. O checkpoint tem **599.045.122 parâmetros** e foi treinado por **738.197.504 tokens**. |

`make history` recupera 2,2802 e 1,8312 BPC no Text8, recalcula 1,0392 / 0,7603 / 0,7752 BPC no D4 e agrega as medições de LAMP. A reexecução de LAMP retornou **4,742863 → 5,388525 NLL**.

## Licença

O código novo está sob Apache-2.0. Os executores históricos incluem a licença Apache-2.0 da origem. Dados, notebooks, resultados e checkpoints mantêm a proveniência documentada em `artefacts/`.
