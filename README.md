# MBA-DATA-SCIENCE

Trabalho de Conclusão de Curso apresentado para obtenção do título de especialista em Data Science e Analytics – 2025

Organização do Projeto Segundo a Metodologia CRISP-DM

Este documento descreve a estrutura do projeto de processamento e limpeza de dados para previsão do índice DST, organizada segundo a metodologia CRISP-DM (Cross Industry Standard Process for Data Mining).

1. Entendimento do Negócio

Objetivo: Prever tempestades geomagnéticas por meio do índice DST utilizando dados solares, solares-interplanetários e manchas solares.

Previsão do índice DST com base em dados históricos.

Suporte à tomada de decisão sobre eventos espaciais extremos.

2. Entendimento dos Dados

Fontes de dados:

labels.csv: contém o índice DST (variável alvo).

solar_wind.csv: contém dados interplanetários (ex: densidade, velocidade, campo magnético).

sunspots.csv: contém o número de manchas solares suavizadas.

Características:

Campos temporais com período comum (period).

Presença de valores ausentes e possíveis outliers.

3. Preparação dos Dados

Etapas implementadas no pipeline (app/pipeline.py):

Carregamento:

Leitura dos arquivos .csv com pandas.read_csv().

Limpeza:

Remoção de valores nulos com dropna().

Detecção e remoção de outliers via método IQR por coluna.

Integração:

Junção dos datasets por period com merge().

Exportação:

Salvamento do dataset final em processed_data.csv.

4. Modelagem (futura)

O dataset limpo será utilizado para treinamento de modelos como LSTM ou Temporal Fusion Transformer (TFT).

5. Avaliação (futura)

Métricas: MAE, RMSE, R² para avaliar a performance da previsão do DST.

6. Deploy (futura)

Automatização do pipeline com GitHub Actions.

Publicação de previsões em dashboards ou alertas automatizados.

Testes Automatizados

Objetivo: Garantir a robustez do pipeline com dados reais.

Arquivo: tests/test_pipeline_real_data.py

Verificações:

Arquivos são carregados corretamente.

Dados são limpos e combinados com sucesso.

Arquivo final é gerado e contém as colunas esperadas.

Estrutura de Pastas

my_tcc_project/
├── app/
│ ├── pipeline.py
│ ├── processing.py
│ └── data.py
├── data/
│ ├── labels.csv
│ ├── solar_wind.csv
│ └── sunspots.csv
├── tests/
│ └── test_pipeline_real_data.py
└── README.md

Autor: LinoÚltima atualização: Maio/2025
