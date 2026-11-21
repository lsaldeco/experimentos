# Dos pruebas agregadas después del diseño original. No hacen búsqueda nueva:
# cada experimento usa los hiperparámetros de su mejor trial (mejor_trial.json).
#
#   --modo arboles     Elige la cantidad de árboles del modelo final con early stopping
#                      sobre un 10% de clientes separado del mismo período del modelo final,
#                      en vez de usar la del mejor trial (elegida entrenando con un período más corto).
#                      Test en 201910, como el experimento original.
#   --modo septiembre  Segundo mes de test: 201909. El modelo final entrena con un mes menos
#                      que el original, para dejar el mismo hueco de dos meses. Árboles del mejor trial.
#
#   python variantes.py --exp base --modo arboles
#
# Las definiciones de experimentos, target y pesos son las mismas que en experimento.py.
import argparse
import gc
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--exp", required=True)
parser.add_argument("--modo", required=True, choices=["arboles", "septiembre"])
parser.add_argument("--datos", default="competencia_02_2019.parquet")
parser.add_argument("--salida", default="resultados")
args = parser.parse_args()
sys.stdout.reconfigure(line_buffering=True)

GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500
CORTES = list(range(2_500, 6_501, 250))
FRACCION_HOLDOUT = 0.10
SEMILLAS = [
    130399, 220391, 417577, 575753, 616261, 975935, 98956, 619348, 491242, 983565,
    777781, 500041, 600011, 800077, 200023, 388890, 1555565, 790126, 250020, 1000085,
]

MESES_FINAL_BASE = [201903, 201904, 201905, 201906, 201907, 201908]
MESES_FINAL_BAJA3 = [201903, 201904, 201905, 201906, 201907]
EXPERIMENTOS = {
    "base": {"positivos": ["BAJA+1", "BAJA+2"], "meses": MESES_FINAL_BASE},
    "A": {"positivos": ["BAJA+2"], "meses": MESES_FINAL_BASE},
    "B": {"positivos": ["BAJA+1"], "meses": MESES_FINAL_BASE},
    "C1": {"positivos": ["BAJA+3"], "meses": MESES_FINAL_BAJA3},
    "C2": {"positivos": ["BAJA+2", "BAJA+3"], "meses": MESES_FINAL_BAJA3},
    "D": {"positivos": ["BAJA+1", "BAJA+2"], "meses": MESES_FINAL_BASE},
    "E": {"positivos": ["BAJA+2"], "meses": MESES_FINAL_BASE, "regresion": True},
    "F": {"positivos": ["BAJA+1", "BAJA+2", "BAJA+3"], "meses": MESES_FINAL_BAJA3},
}
COLUMNAS_NO_FEATURES = ["numero_de_cliente", "foto_mes", "clase_ternaria", "clase_extendida"]

conf = EXPERIMENTOS[args.exp]
if args.modo == "arboles":
    meses_train, mes_test = conf["meses"], 201910
else:
    # Un mes menos que el modelo final original: entre el último mes de entrenamiento
    # y el de test quedan los mismos dos meses que entre 201908 y 201910
    meses_train, mes_test = conf["meses"][:-1], 201909

carpeta = os.path.join(args.salida, args.exp, args.modo)
os.makedirs(carpeta, exist_ok=True)
if os.path.exists(os.path.join(carpeta, "resumen.json")):
    print(f"{args.exp} {args.modo}: ya estaba terminado")
    sys.exit(0)


def leer_mejor_trial():
    """El del experimento en la carpeta de salida. Si no está (corridas de Laura), el copiado al repo."""
    rutas = [os.path.join(args.salida, args.exp, "mejor_trial.json"),
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "mejor_trials", f"{args.exp}.json")]
    for ruta in rutas:
        if os.path.exists(ruta):
            print(f"Mejor trial: {ruta}")
            with open(ruta) as f:
                return json.load(f)
    raise FileNotFoundError(f"No hay mejor_trial.json para {args.exp}")


mejor = leer_mejor_trial()


def target(clase):
    if conf.get("regresion"):
        return np.where(clase == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO).astype(float)
    return clase.isin(conf["positivos"]).astype(int).to_numpy()


def pesos(clase):
    """Pesos de BAJA+1 y BAJA+3 del mejor trial (D y F). None si el experimento no los usa."""
    p1 = mejor["params"].get("peso_baja1")
    p3 = mejor["params"].get("peso_baja3")
    if p1 is None and p3 is None:
        return None
    return np.select([clase == "BAJA+1", clase == "BAJA+3"],
                     [1.0 if p1 is None else p1, 1.0 if p3 is None else p3], 1.0)


def params_modelo(semilla):
    params = {"objective": "regression" if conf.get("regresion") else "binary", "metric": "auc",
              "bagging_freq": 1, "max_bin": 31, "feature_pre_filter": False, "verbose": -1,
              **mejor["params"], "seed": semilla}
    params.pop("peso_baja1", None)
    params.pop("peso_baja3", None)
    return params


def ganancia_por_corte(prob, clase):
    orden = np.argsort(-prob, kind="stable")
    acumulada = np.cumsum(np.where(clase.to_numpy()[orden] == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO))
    return {corte: float(acumulada[corte - 1]) for corte in CORTES}


def dataset(tabla, **kwargs):
    return lgb.Dataset(tabla[features], label=target(tabla["clase_extendida"]),
                       weight=pesos(tabla["clase_extendida"]), **kwargs)


# ---------------------------------------------------------------- datos
datos = pd.read_parquet(args.datos)
datos = datos[datos["foto_mes"].isin(set(meses_train) | {mes_test})].copy()
features = [c for c in datos.columns
            if c not in COLUMNAS_NO_FEATURES and pd.api.types.is_numeric_dtype(datos[c])]
datos[features] = datos[features].astype("float32")
train = datos[datos["foto_mes"].isin(meses_train)]
test = datos[datos["foto_mes"] == mes_test]
del datos
gc.collect()
print(f"{args.exp} {args.modo}: {len(features)} variables, train {sorted(train['foto_mes'].unique())}, "
      f"test {mes_test}")

# ---------------------------------------------------------------- cantidad de árboles
arboles = mejor["best_iter"]
resumen = {"experimento": args.exp, "modo": args.modo, "meses_train": meses_train, "mes_test": mes_test,
           "arboles_mejor_trial": mejor["best_iter"]}
if args.modo == "arboles":
    # 10% de clientes afuera, con todos sus meses: ningún cliente queda de los dos lados
    clientes = train["numero_de_cliente"].drop_duplicates()
    afuera = clientes.sample(frac=FRACCION_HOLDOUT, random_state=SEMILLAS[0])
    es_holdout = train["numero_de_cliente"].isin(afuera)
    dtrain = dataset(train[~es_holdout], free_raw_data=False)
    # Igual que en la búsqueda: se valida con AUC contra BAJA+2
    dvalid = lgb.Dataset(train.loc[es_holdout, features],
                         label=(train.loc[es_holdout, "clase_ternaria"] == "BAJA+2").astype(int),
                         reference=dtrain)
    lr = mejor["params"]["learning_rate"]
    modelo = lgb.train(params_modelo(SEMILLAS[0]), dtrain, num_boost_round=3000, valid_sets=[dvalid],
                       callbacks=[lgb.early_stopping(int(50 + 5 / lr), verbose=False)])
    arboles = modelo.best_iteration
    resumen.update({"arboles_holdout": arboles, "auc_holdout": modelo.best_score["valid_0"]["auc"]})
    print(f"Árboles: {mejor['best_iter']} en el mejor trial, {arboles} con el 10% de clientes "
          f"(AUC {resumen['auc_holdout']:.4f})")
    del dtrain, dvalid, modelo
    gc.collect()

# ---------------------------------------------------------------- modelos finales y ganancia
predicciones = {"numero_de_cliente": test["numero_de_cliente"].to_numpy()}
filas = []
for semilla in SEMILLAS:
    modelo = lgb.train(params_modelo(semilla), dataset(train), num_boost_round=arboles)
    prob = modelo.predict(test[features])
    predicciones[f"semilla_{semilla}"] = prob
    for corte, ganancia in ganancia_por_corte(prob, test["clase_ternaria"]).items():
        filas.append({"experimento": args.exp, "semilla": semilla, "corte": corte, "ganancia": ganancia})
    print(f"Semilla {semilla} lista")

pd.DataFrame(predicciones).to_csv(os.path.join(carpeta, f"prediccion_{mes_test}.csv"), index=False)
curvas = pd.DataFrame(filas)
curvas.to_csv(os.path.join(carpeta, f"ganancia_{mes_test}.csv"), index=False)
medias = curvas.groupby("semilla")["ganancia"].mean()
resumen.update({"curva_media": medias.mean(), "curva_desvio": medias.std()})
print(f"Promedio de la curva {CORTES[0]:,}-{CORTES[-1]:,}: media {medias.mean():,.0f}, desvío {medias.std():,.0f}")
# Se escribe al final: si existe, la variante está completa
with open(os.path.join(carpeta, "resumen.json"), "w") as f:
    json.dump(resumen, f, indent=2)
