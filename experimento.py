# Corre un experimento del grupo targets: Optuna, modelos finales con todas las semillas
# y curva de ganancia en 201910. Todos los experimentos usan este mismo script:
# lo único que cambia entre ellos está en EXPERIMENTOS.
#   python experimento.py --exp base
import argparse
import json
import os
import shutil
import time

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True)
parser.add_argument("--datos", default="competencia_02_2019.parquet")
parser.add_argument("--salida", default="resultados")
parser.add_argument("--trials", type=int, default=50)
args = parser.parse_args()

GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500
MES_VALIDACION = 201908
MES_TEST = 201910
CORTES = list(range(8_000, 15_001, 500))

# Vector fijo de semillas (de ambos, Fernando y Laura). El mismo para todos los experimentos.
SEMILLAS = [
    130399, 220391, 417577, 575753, 616261, 975935, 98956, 619348, 491242, 983565,
    777781, 500041, 600011, 800077, 200023, 388890, 1555565, 790126, 250020, 1000085,
]

MESES_BASE = {"busqueda": [201903, 201904, 201905, 201906],
              "final": [201903, 201904, 201905, 201906, 201907, 201908]}
# BAJA+3 mira tres meses adelante: pierde un mes en cada etapa
MESES_BAJA3 = {"busqueda": [201903, 201904, 201905],
               "final": [201903, 201904, 201905, 201906, 201907]}

EXPERIMENTOS = {
    "base": {"positivos": ["BAJA+1", "BAJA+2"], "meses": MESES_BASE},
    "A": {"positivos": ["BAJA+2"], "meses": MESES_BASE},
    "B": {"positivos": ["BAJA+1"], "meses": MESES_BASE},
    "C1": {"positivos": ["BAJA+3"], "meses": MESES_BAJA3},
    "C2": {"positivos": ["BAJA+2", "BAJA+3"], "meses": MESES_BAJA3},
    "D": {"positivos": ["BAJA+1", "BAJA+2"], "meses": MESES_BASE, "peso_baja1": True},
    "E": {"positivos": ["BAJA+2"], "meses": MESES_BASE, "regresion": True},
}

COLUMNAS_NO_FEATURES = ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase_extendida"]


def target(clase, conf):
    """Etiqueta de entrenamiento según el experimento. E usa la ganancia de cada cliente."""
    if conf.get("regresion"):
        return np.where(clase == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO).astype(float)
    return clase.isin(conf["positivos"]).astype(int).to_numpy()


def pesos(clase, peso_baja1):
    return np.where(clase == "BAJA+1", peso_baja1, 1.0)


def ganancia_por_corte(prob, clase):
    """Ganancia en cada corte de CORTES, ordenando de mayor a menor probabilidad."""
    orden = np.argsort(-prob, kind="stable")
    ganancia = np.where(clase.to_numpy()[orden] == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    acumulada = np.cumsum(ganancia)
    return {corte: float(acumulada[corte - 1]) for corte in CORTES}


conf = EXPERIMENTOS[args.exp]
assert len(SEMILLAS) >= 10, "Completar SEMILLAS con el vector fijo del grupo"
carpeta = os.path.join(args.salida, args.exp)
os.makedirs(carpeta, exist_ok=True)

# ---------------------------------------------------------------- datos
datos = pd.read_parquet(args.datos)
meses_usados = set(conf["meses"]["final"]) | {MES_VALIDACION, MES_TEST}
datos = datos[datos["foto_mes"].isin(meses_usados)].copy()
features = [c for c in datos.columns
            if c not in COLUMNAS_NO_FEATURES and pd.api.types.is_numeric_dtype(datos[c])]
datos[features] = datos[features].astype("float32")
print(f"Experimento {args.exp}: {len(features)} variables, {len(datos):,} filas")

busqueda = datos[datos["foto_mes"].isin(conf["meses"]["busqueda"])]
validacion = datos[datos["foto_mes"] == MES_VALIDACION]
final = datos[datos["foto_mes"].isin(conf["meses"]["final"])]
test = datos[datos["foto_mes"] == MES_TEST]

objetivo = "regression" if conf.get("regresion") else "binary"
params_fijos = {"objective": objetivo, "metric": "auc", "bagging_freq": 1,
                "feature_pre_filter": False, "verbose": -1}

dtrain = lgb.Dataset(busqueda[features], label=target(busqueda["clase_extendida"], conf),
                     free_raw_data=False)
# Validación siempre contra BAJA+2, para todos los experimentos
dvalid = lgb.Dataset(validacion[features],
                     label=(validacion["clase_ternaria"] == "BAJA+2").astype(int),
                     reference=dtrain)


# ---------------------------------------------------------------- búsqueda bayesiana
def objective(trial):
    params = {
        **params_fijos,
        "num_leaves": trial.suggest_int("num_leaves", 8, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 1000),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.1, 1.0),
        "seed": SEMILLAS[0],
    }
    if conf.get("peso_baja1"):
        peso = trial.suggest_float("peso_baja1", 0.0, 1.0)
        dtrain.set_weight(pesos(busqueda["clase_extendida"], peso))
    modelo = lgb.train(params, dtrain, num_boost_round=3000, valid_sets=[dvalid],
                       callbacks=[lgb.early_stopping(int(50 + 5 / params["learning_rate"]), verbose=False)])
    trial.set_user_attr("best_iter", modelo.best_iteration)
    return modelo.best_score["valid_0"]["auc"]


# La base de Optuna se guarda en el disco de la máquina y se copia a la carpeta de salida (el bucket)
# después de cada trial. Si Google apaga la máquina, la búsqueda retoma desde la copia.
# El nombre incluye la carpeta de salida, para que dos corridas distintas nunca compartan búsqueda
db_local = os.path.join(os.path.expanduser("~"), "optuna_" + carpeta.strip("/~").replace("/", "_") + ".db")
db_copia = os.path.join(carpeta, "optuna.db")
if not os.path.exists(db_local) and os.path.exists(db_copia):
    shutil.copy(db_copia, db_local)

estudio = optuna.create_study(direction="maximize", study_name=args.exp,
                              storage=f"sqlite:///{db_local}", load_if_exists=True,
                              sampler=optuna.samplers.TPESampler(seed=SEMILLAS[0]))
hechos = len([t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE])
faltan = max(0, args.trials - hechos)
print(f"Trials completos: {hechos}. Faltan: {faltan}")
inicio = time.time()
estudio.optimize(objective, n_trials=faltan,
                 callbacks=[lambda estudio, trial: shutil.copy(db_local, db_copia)])
mejor = estudio.best_trial
print(f"Búsqueda: {(time.time() - inicio) / 60:.1f} min. AUC agosto {mejor.value:.5f}, {mejor.params}")


def params_modelo(semilla):
    params = {**params_fijos, **mejor.params, "seed": semilla}
    params.pop("peso_baja1", None)
    return params


def entrenar(tabla, semilla):
    peso = pesos(tabla["clase_extendida"], mejor.params["peso_baja1"]) if conf.get("peso_baja1") else None
    dataset = lgb.Dataset(tabla[features], label=target(tabla["clase_extendida"], conf), weight=peso)
    return lgb.train(params_modelo(semilla), dataset, num_boost_round=mejor.user_attrs["best_iter"])


# ---------------------------------------------------------------- predicción de agosto para los pesos del ensamble
# El mejor trial reentrenado con los meses de búsqueda (no vio agosto) y la primera semilla
modelo_agosto = entrenar(busqueda, SEMILLAS[0])
pd.DataFrame({"numero_de_cliente": validacion["numero_de_cliente"],
              "prediccion": modelo_agosto.predict(validacion[features]),
              "clase_ternaria": validacion["clase_ternaria"]}
             ).to_csv(os.path.join(carpeta, "prediccion_agosto.csv"), index=False)

# ---------------------------------------------------------------- modelos finales y ganancia en octubre
predicciones = {"numero_de_cliente": test["numero_de_cliente"].to_numpy()}
filas = []
for semilla in SEMILLAS:
    modelo = entrenar(final, semilla)
    prob = modelo.predict(test[features])
    predicciones[f"semilla_{semilla}"] = prob
    for corte, ganancia in ganancia_por_corte(prob, test["clase_ternaria"]).items():
        filas.append({"experimento": args.exp, "semilla": semilla, "corte": corte, "ganancia": ganancia})
    print(f"Semilla {semilla} lista")

pd.DataFrame(predicciones).to_csv(os.path.join(carpeta, "prediccion_octubre.csv"), index=False)
curvas = pd.DataFrame(filas)
curvas.to_csv(os.path.join(carpeta, "ganancia_octubre.csv"), index=False)
resumen = curvas.groupby("semilla")["ganancia"].mean()
print(f"Promedio de la curva 8.000-15.000: media {resumen.mean():,.0f}, desvío {resumen.std():,.0f}")

with open(os.path.join(carpeta, "mejor_trial.json"), "w") as f:
    json.dump({"auc_agosto": mejor.value, "params": mejor.params,
               "best_iter": mejor.user_attrs["best_iter"], "trials": len(estudio.trials)}, f, indent=2)
