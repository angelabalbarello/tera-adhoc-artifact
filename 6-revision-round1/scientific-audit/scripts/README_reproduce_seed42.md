# Pacote de reprodução — seed 42 (ADHOC-D-26-02099, revisão round 1)

Reproduz com um comando, após descompactar, em qualquer máquina com Python >= 3.9:

```
pip install numpy pandas
python3 reproduce_seed42.py
```

Saída esperada: os quatro resultados de semântica de inferência
(fixed_prefix FR=0 TTDef=0.015625; fixed_incremental 0.009549;
mheg_prefix_recompute 0.023438 / 44.6855% — idêntico ao publicado;
mheg_incremental_skip 0.170139 / 44.3803%), a verificação
FR >= frac(p_max < theta) e a reconciliação da Tabela 13
(aftkd_fixed: p̄_max 0.9400±0.0826, 100% >= 0.10, FR=0 = publicado).

Caminhos podem ser passados por argumento (`--data`, `--models`,
`--published`, `--calib`, `--out`); por padrão tudo é lido do diretório do
próprio script. Nenhuma dependência de torch: o checkpoint é lido por um
unpickler mínimo e o LSTM roda em numpy (as fórmulas são as do pipeline
publicado; ver Seção 3.4 e 4.1 do artigo).

Os dados e checkpoints abaixo são cópias byte-idênticas dos artefatos do
repositório público tera-adhoc-artifact
(`1-pipeline-tera/results/exp_seed12_round1/`), campanha congelada
exp_seed12_round1 (manifest.json no repositório).

| arquivo | tamanho | sha256 |
|---|---|---|
| `X_te_seed42.npy` | 1,869,056 B | `2086883c922d994ca1d37f1a4dabf3ee5fa542dce09ece46e4bbb913953483c5` |
| `baseline_seed42.pt` | 235,988 B | `4d365467500748686eeed6f181d7cb2859ccf6530860b430e260a680f5c275b8` |
| `calibration_summary.csv` | 569 B | `15af2ab8ed3e2ae042aed2b19780743aefebbc689b2c852985383c8b97d92331` |
| `expanded_per_seed.csv` | 3,387 B | `a55e37c13aac6124d2d479721321c890368b6816059faf334455cb2140cd2a7a` |
| `expected_tab13_reconciliation.csv` | 37,895 B | `939c19d4ea7aa48647efcadb16c4592ccd67804a0f3cf00ee5740086778a8105` |
| `reproduce_seed42.py` | 8,724 B | `b49a8d91e66b223463a444fb018ef846051e0a045699c9eb12e4f04aa1baa7e7` |
| `student_seed42.pt` | 235,979 B | `9f58c7611ec33d97889ec1f8b454c437f7a57a39625275b653ef5d20a038f428` |
| `y_ep_te_seed42.npy` | 1,384 B | `5096dd0761f38801cf48fc06a7065d0913674922c19da7193d04bb709f25a713` |
| `y_fr_te_seed42.npy` | 120,704 B | `5c5a34cc6bb06e5e9248665534f43a8fc27b5346e4383499b2e088cae60ada76` |
