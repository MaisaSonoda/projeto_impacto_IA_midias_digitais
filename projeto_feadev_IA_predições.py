# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
import numpy as np

# abrir database
df = pd.read_csv("Global_AI_Content_Impact_Dataset.csv")

# criar dicionário com os modelos de predição
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor()
}

# definir grids de hiperparâmetros por modelo
param_distributions = {
    "LinearRegression": {},
    "Ridge": {
        "regressor__alpha": np.logspace(-3, 3, 10)
    },
    "Lasso": {
        "regressor__alpha": np.logspace(-3, 3, 10)
    },
    "DecisionTree": {
        "regressor__max_depth": [None, 5, 10, 20],
        "regressor__min_samples_split": [2, 5, 10]
    },
    "RandomForest": {
        "regressor__n_estimators": [50, 100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5]
    }
}

# definir colunas alvo
target_columns = ["Job Loss Due to AI (%)", "Revenue Increase Due to AI (%)", "Consumer Trust in AI (%)"]
all_results = {}

for target in target_columns:
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # identificar colunas categóricas e numéricas
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # criar preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="passthrough"
    )

    rmse_results = {}

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])

        if param_distributions.get(name):
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=param_distributions[name],
                n_iter=10,
                cv=3,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
        else:
            best_model = pipeline.fit(X_train, y_train)

        preds = best_model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        rmse = sqrt(mse)
        rmse_results[name] = rmse

    all_results[target] = rmse_results

# transformar resultados em DataFrame e exibir
all_results_df = pd.DataFrame(all_results)
print(all_results_df)

#montagem de gráficos:

# Reorganizar o DataFrame para formato "long" (necessário para seaborn)
df_melted = all_results_df.reset_index().melt(id_vars="index", var_name="Target", value_name="RMSE")
df_melted.rename(columns={"index": "Model"}, inplace=True)

# Gráfico para mostrar desempenho dos modelos
sns.barplot(data=df_melted, x="Target", y="RMSE", hue="Model")
plt.title("RMSE por Modelo e Target")
plt.ylabel("RMSE (quanto menor, melhor)")
plt.xticks(rotation=30)
plt.legend(title="Modelo")
plt.tight_layout()
plt.show()

"""
para quem estiver revisando isso, acho legal tentar implementar um cross validation e talvez um feature importance
mas aí fica do seu critério mesmo
e tmb, fique à vontade para mudar as coisas ou tentar otimizar o código
pq não faço a minima ideia de como diminuir esse tempo que demora para o código rodar
"""