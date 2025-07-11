
# main.py

import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# === Imports dos seus módulos ===
from src.log import configurar_logging, registrar_modelo

from src.processamento import (
    carregar_dados,
    separar_dados_treino_teste,
    preparar_dados_transformer
)
from src.arvore import (
    executar_modelos_arvores,
    treinar_decision_tree,
    treinar_random_forest,
    treinar_gradient_boosting,
    tunar_decision_tree,
    tunar_random_forest,
    tunar_gradient_boosting,
)   
from src.model import (
    preparar_dados,
    treinar_regressao,
    treinar_e_registrar,
    treinar_transformer,
)
from src.analysis import (
    estimar_modelo,
    diagnosticar_residuos,
    analisar_multicolinearidade,
    analise_splits
)
from src.plot import ( 
    plotar_comparacao_modelos, 
    exibir_model,
    plotar_distribuicao_treino_teste
    
)

from src.avaliacao import ( 
    prever_transformer, 
    avaliar_modelo,
    avaliar_transformer
)

def main():
    # Configuração inicial
    configurar_logging()
    os.makedirs("output", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    try:
        # ========= Regressão Linear =========
        logging.info("\n" + "="*50)
        logging.info("🚀 INICIANDO PROCESSO DE MODELAGEM")
        logging.info("="*50 + "\n")
        
        logging.info("📊 Carregando dados...")
        df = carregar_dados("data/clean/df_final.pkl")
        X_train, X_test, y_train, y_test = separar_dados_treino_teste(df, target="dst")
        
        logging.info("\n🔵 Treinando Regressão Linear...")
        modelo_lr, metrics_lr = treinar_e_registrar(
            "Regressão Linear", 
            treinar_regressao, 
            X_train, y_train, 
            X_test, y_test
        )

        # Análise OLS
        df_train = X_train.copy()
        df_train['target'] = y_train
        modelo_ols = estimar_modelo(df_train, "target ~ " + " + ".join(X_train.columns))
        logging.info("\n📝 Resumo OLS:\n" + modelo_ols.summary().as_text())

        diagnosticar_residuos(modelo_ols)
        logging.info("\n🔍 Multicolinearidade:\n" + analisar_multicolinearidade(X_train).to_string())

        # ========= Modelos baseados em árvore =========
        logging.info("\n" + "="*50)
        logging.info("🌳 INICIANDO MODELOS BASEADOS EM ÁRVORE")
        logging.info("="*50 + "\n")
        
        logging.info("🌳 Treinando Decision Tree...")
        modelo_dt, params_dt, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='decision_tree',
            usar_tuning=False,
            max_depth=5,
            random_state=42
)
        
        logging.info("🔵 Treinando Random Forest...")
        modelo_rf, params_rf, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='random_forest',
            usar_tuning=False,
            n_estimators=100,
            random_state=42
        )

        logging.info("\n🟢 Treinando Gradient Boosting...")
        modelo_gb, params_gb, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='gradient_boosting',
            usar_tuning=True,
            random_state=42
        )

        # Análise de splits
        if hasattr(modelo_rf, 'estimators_'):
            df_splits = analise_splits(modelo_rf.estimators_[0], X_train, y_train)
            

        # ========= Transformer =========
        logging.info("\n" + "="*50)
        logging.info("🤖 INICIANDO MODELO TRANSFORMER")
        logging.info("="*50 + "\n")
        
        caminho_transformer = 'data/clean/df_prox_h.pkl'
        if not os.path.exists(caminho_transformer):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_transformer}")

        dados_prox_h = pd.read_pickle(caminho_transformer)
        X_tr_train, X_tr_test, y_tr_train, y_tr_test = preparar_dados_transformer(
            dados_prox_h 
            
        )

        modelo_transformer, metrics_transformer = treinar_e_registrar(
            "Transformer",
            treinar_transformer,
            X_tr_train, y_tr_train,
            X_tr_test, y_tr_test
        )

        # ========= Avaliação e comparação final =========
        logging.info("\n" + "="*50)
        logging.info("📈 AVALIAÇÃO FINAL DOS MODELOS")
        logging.info("="*50 + "\n")
        
        # ========= Avaliação e comparação final =========
        modelos = [
            ("Regressão Linear", modelo_lr, X_test, y_test),
            ("OLS", modelo_ols, X_test, y_test),
            ("Decision Tree", modelo_dt, X_test, y_test),  # Agora definido
            ("Random Forest", modelo_rf, X_test, y_test),
            ("Gradient Boosting", modelo_gb, X_test, y_test),
            ("Transformer (prox_h)", modelo_transformer, X_tr_test, y_tr_test)
        ]

        resultados_r2 = []
        resultados_rmse = []
        nomes = []

        for nome, modelo, X_t, y_t in modelos:
            nome_arquivo = f"output/desemp_model_{nome.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
            exibir_model(modelo, X_t, y_t, nome_modelo=nome, salvar_em=nome_arquivo)

            aval = avaliar_modelo(modelo, X_t, y_t)
            nomes.append(nome)
            resultados_r2.append(aval["R2"])
            resultados_rmse.append(aval["RMSE"])

        plotar_comparacao_modelos(nomes, resultados_r2, resultados_rmse)
        
    except Exception as e:
        logging.error(f"\n❌ Erro durante a execução: {str(e)}")
        raise
        
if __name__ == "__main__":
        os.makedirs("output", exist_ok=True)
        main()


