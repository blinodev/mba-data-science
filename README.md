# Modelo de previsão das tempestades geomagneticas.

Organização do Projeto Segundo a Metodologia CRISP-DM

Este documento descreve a estrutura do projeto de processamento e limpeza de dados para previsão do índice DST, organizada segundo a metodologia CRISP-DM (Cross Industry Standard Process for Data Mining).

1. Entendimento do Negócio 🚀

Objetivo: Prever tempestades geomagnéticas por meio do índice DST utilizando dados solares, interplanetários e de manchas solares.

Antecipar eventos espaciais extremos que afetam redes elétricas e sistemas de comunicação.

Utilizar o índice DST como métrica chave para indicar distúrbios na magnetosfera.

2. Entendimento dos Dados 🧠

Fontes de dados:

labels.csv: contém o índice DST (variável alvo).

solar_wind.csv: contém dados do vento solar e campo magnético interplanetário.

sunspots.csv: contém o número de manchas solares suavizado (SSN).

Características principais:

Coluna temporal comum: period.

Presença de valores ausentes e outliers.

Necessidade de padronização e limpeza.

3. Preparação dos Dados 🔧

Pipeline implementado em app/pipeline.py:

Carregamento:

Leitura dos arquivos .csv com pandas.read_csv().

Limpeza:

Remoção de valores nulos com dropna().

Detecção e remoção de outliers com base no método IQR por coluna.

Integração:

Junção dos três datasets por period usando merge().

Exportação:

Salvamento do dataset processado como processed_data.csv.

4. Modelagem (Etapa Futura) 🤖

O dataset limpo será base para modelos de aprendizado de máquina.

Possíveis modelos: LSTM, Temporal Fusion Transformer (TFT), Random Forest.

5. Avaliação (Etapa Futura) 📊

Avaliação do desempenho preditivo com métricas como:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

R² (Coeficiente de Determinação)

6. Deploy (Etapa Futura) 🚢

Automatização do pipeline com GitHub Actions.

Integração com dashboards (por exemplo, Power BI ou Streamlit).

Geração de alertas baseados em predições de eventos críticos.



Testes Automatizados ✅

Objetivo: Validar o funcionamento do pipeline com dados reais.

Arquivo de testes: tests/test_pipeline_real_data.py

Cobertura dos testes:

Verifica se os arquivos .csv são carregados corretamente.

Confirma remoção de nulos e outliers.

Checa integridade do dataset final gerado.

Assegura que a coluna period esteja presente e bem formatada.

Estrutura de Pastas do Projeto 📁

## Estrutura de Pastas do Projeto

```bash
my_tcc_project/
├── app/
│   ├── data.py
│   └── model_rgline.py
│   
├── src/
│   ├── linear_regression.py
│   ├── transformer_model_ACE.py
│   ├── transformer_model_DSCOVR.py
│   └── transformer_model_DSCOVR.py

├── tests/
│   ├── test_data.py
│   └── test_rgline.py
└── README.md
```

Autor: Lino 👨‍🚀Última atualização: Maio/2025 🗓️
