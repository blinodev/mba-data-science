
# main.py


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = remove INFO, 2 = remove WARNING, 3 = remove all (exceto erro fatal)
import sys
import logging
import pandas as pd
from utils.log import configurar_logging
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils.plot import (relatorio_missing, 
                   carregar_dados, 
                   separar_dados_treino_teste, 
                   plotar_distribuicao_treino_teste,
                   exibir_model,
                   plotar_arvore_decisao,
                   exibir_importancia_variaveis,
                   plotar_fitted_values,
                   preparar_dados_transformer,
                   avaliar_modelo,
                   avaliar_transformer,
                   prever_transformer,
                   plotar_comparacao_modelos,
                   plotar_resultados_deep_learning

)
from utils.log import (configurar_logging,
                       registrar_modelo,

                        
)
from modelos.model import (treinar_regressao, 
                     treinar_e_registrar, 
                     estimar_modelo, 
                     diagnosticar_residuos,
                     verificar_tamanhos,
                     analise_splits,
                     tunar_decision_tree,
                     tunar_random_forest,
                     executar_modelos_arvores,
                     treinar_random_forest,
                     treinar_transformer,
                     treinar_e_registrar_transformer
                     
)

def main():
    # Configuração inicial
    log_path = configurar_logging()
    logging.info(f"📂 Logs serão gravados em: {log_path}")
    
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
        plotar_distribuicao_treino_teste(X_train, X_test)

        exibir_model(modelo_lr, X_test, y_test, nome_modelo="Regressão Linear")

        # Log de sucesso
        logging.info("✅ Regressão Linear treinado com sucesso.")

        # Análise OLS
        df_train = X_train.copy()
        df_train['target'] = y_train
        modelo_ols = estimar_modelo(df_train, "target ~ " + " + ".join(X_train.columns))
        logging.info("\n📝 Resumo OLS:\n" + modelo_ols.summary().as_text())
        diagnosticar_residuos(modelo_ols)
        
        # ========= Modelos baseados em árvore =========
        logging.info("\n" + "="*50)
        logging.info("🌳 INICIANDO MODELOS BASEADOS EM ÁRVORE")
        logging.info("="*50 + "\n")
        
        logging.info("🌳 Treinando Decision Tree...")
        modelo_dt, params_dt, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='decision_tree',
            usar_tuning=True,
            random_state=42
)
        # Analise Split
        tree_split_df = analise_splits(modelo_dt, X_train, y_train)
        print(tree_split_df)

        # Obtendo os valores avaliação do modelo de árvore de decisão
        exibir_model(modelo_dt, X_train, y_train, nome_modelo="Árvore de Decisão - Treino")
        exibir_model(modelo_dt, X_test, y_test, nome_modelo="Árvore de Decisão - Teste")

        logging.info("🌳 Tuning Decision Tree...")

        # Importância das variáveis
        logging.info("🌳 Importância das Variáveis Decision Tree...")
        exibir_importancia_variaveis(modelo_dt, X_train, salvar_em="output/importancia_tree_best.png")

        # Previsões no conjunto de teste
        tree_best_pred_test = modelo_dt.predict(X_test)

        # Gráfico de valores ajustados (fitted values)
        plotar_fitted_values(y_test, tree_best_pred_test, nome_modelo="Decision Tree Tunada")

        # Log final
        logging.info("✅ Decision Tree Tuning com sucesso.")

        logging.info("🔵 Treinando Random Forest...")
        modelo_rf, params_rf, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='random_forest',
            usar_tuning=False            
        )
        # Obtendo os valores avaliação do modelo de árvore de decisão
        exibir_model(modelo_rf, X_train, y_train, nome_modelo="Random Forest - Treino")
        exibir_model(modelo_rf, X_test, y_test, nome_modelo="Random Forest - Teste")

        # Importância das variáveis
        logging.info("🌳 Importância das Variáveis Random Forest...")
        exibir_importancia_variaveis(modelo_rf, X_train, salvar_em="output/importancia_modelo_rf.png")

        # Previsões no conjunto de teste
        rf_grid_pred_test = modelo_rf.predict(X_test)


        # Gráfico de valores ajustados (fitted values)
        plotar_fitted_values(y_test, rf_grid_pred_test, salvar_em="output/rf_fitted_values.png")


        logging.info("\n🟢 Treinando Gradient Boosting...")
        modelo_gb, params_gb, _ = executar_modelos_arvores(
            X_train, y_train,
            modelo_tipo='gradient_boosting',
            usar_tuning=True            
        )
        # Obtendo os valores avaliação do modelo de árvore de decisão
        exibir_model(modelo_gb, X_train, y_train, nome_modelo="Gradient Boosting - Treino")
        exibir_model(modelo_gb, X_test, y_test, nome_modelo="Gradient Boosting - Teste")

        # Importância das variáveis
        logging.info("🌳 Importância das Variáveis Gradient Boosting...")
        exibir_importancia_variaveis(modelo_gb, X_train, salvar_em="output/importancia_modelo_gb.png")

        # Previsões no conjunto de teste
        gb_grid_pred_test = modelo_gb.predict(X_test)

        # Gráfico de valores ajustados (fitted values)
        plotar_fitted_values(y_test, gb_grid_pred_test, salvar_em="output/gb_fitted_values.png")

        # ========= Transformer =========
        logging.info("\n" + "="*50)
        logging.info("🤖 INICIANDO MODELO TRANSFORMER")
        logging.info("="*50 + "\n")
        logging.info("📊 Carregando dados...")

        caminho_transformer = 'data/clean/df_prox_h.pkl'
        if not os.path.exists(caminho_transformer):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho_transformer}")

        dados_prox_h = pd.read_pickle(caminho_transformer)
        X_tr_train, X_tr_test, y_tr_train, y_tr_test = preparar_dados_transformer(
            dados_prox_h             
        )
        # Treinamento e registro do modelo Transformer com histórico
        modelo_transformer, history_transformer, metrics_transformer = treinar_e_registrar_transformer(
            "Transformer",
            treinar_transformer,
            X_tr_train, y_tr_train,
            X_tr_test, y_tr_test,
            epochs=20
)

        plotar_resultados_deep_learning(
            modelo_transformer,
            X_tr_test,
            y_tr_test,
            prefixo_nome="modelo_dst"
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


