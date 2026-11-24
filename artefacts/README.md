# Artefatos dos experimentos

Os alvos do Makefile resolvem os insumos a partir deste diretório. Os originais do workspace histórico foram preservados.

| Diretório | Origem e conteúdo |
| --- | --- |
| `text8/` | FINDINGS Etapa 1, notebook executado do job 11831 e corpus oficial Text8 com MD5 `3bea1919949baf155f99411df5fada7e`. |
| `tinystories-d4/` | FINDINGS Etapa 9 D4, manifesto histórico, curvas e sumários originais. Os hashes do corpus ordenado e do tokenizer constam do manifesto. Os dois arquivos não estavam disponíveis no volume. |
| `lamp/` | FINDINGS Etapa 19 e JSON primário com offsets e métricas. `weights.pt` contém os tensores de modelo e metadados do checkpoint original com SHA-256 `dcd1a9ff2eca4bdf7e2fc6615bb0b1dca4f14e625d4bd472e1fb8dd4e43026ea`. `windows.npz` contém as 35 janelas uint16 de 2.048 IDs, extraídas aos offsets registrados do pack original; o sorteio PCG64 foi conferido com seed 180017. |

`make verify-artefacts` verifica o SHA-256 dos arquivos preservados. `weights.pt` guarda os pesos de inferência e pode ser carregado com `weights_only=True`. Corpus Text8 e pesos de LAMP ficam no checkout via Git LFS. A distribuição Python inclui código e os registros leves na distribuição fonte. Os dois arquivos grandes permanecem no checkout de pesquisa.

## Referência bibliográfica

Matheus Tomieiro de Oliveira. *CelLM: Cellular Neural Dynamics for Text Processing*. 2026.
