# Experimento colaborativo: manipulación del target

Grupo targets (comisión lunes): Fernando Steeb y Laura Aldeco.

Probamos distintas definiciones del target para predecir BAJA+2 y las comparamos con el baseline (BAJA+1 y BAJA+2 como positivos). También armamos un ensamble ponderado de todos los experimentos.

## Experimentos

| Id | Positivos | Qué cambia respecto del baseline |
|---|---|---|
| base | BAJA+1 y BAJA+2 | Nada |
| A | BAJA+2 | BAJA+1 pasa a ser negativo |
| B | BAJA+1 | Solo los que se van el mes siguiente |
| C1 | BAJA+3 | Solo los que se van dentro de tres meses |
| C2 | BAJA+2 y BAJA+3 | El positivo se amplía hacia el futuro |
| D | BAJA+1 y BAJA+2 | BAJA+1 entra con un peso entre 0 y 1 que busca Optuna |
| E | Target continuo | Regresión: BAJA+2 vale 1.072.500 y el resto −27.500 |

Todo lo demás es igual en todos los experimentos: datos, variables, búsqueda de hiperparámetros, semillas y evaluación.

## Meses

| Experimentos | Búsqueda de hiperparámetros | Modelo final |
|---|---|---|
| base, A, B, D, E | Entrena con 201903–201906 | Entrena con 201903–201908 |
| C1 y C2 | Entrena con 201903–201905 | Entrena con 201903–201907 |

Todos validan con AUC en 201908 contra BAJA+2 y se evalúan con la ganancia en 201910. C1 y C2 pierden un mes porque BAJA+3 necesita saber si el cliente sigue tres meses después.

## Scripts

**`preparar_datos.py`** se corre una sola vez. Lee del crudo solo 2019, calcula `clase_ternaria` y `clase_extendida` (que agrega BAJA+3) y los lags y deltas de 1 y 2 meses de todas las variables numéricas, sin rellenar nulos. Guarda todo en un parquet. También genera `control_meses.csv`, con la proporción de ceros y de nulos de cada variable en cada mes, para detectar meses con datos rotos.

**`experimento.py`** se corre una vez por experimento (`--exp base`, `A`, `B`, `C1`, `C2`, `D` o `E`).
- **Búsqueda:** optimización bayesiana con Optuna sobre `num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction` y `bagging_fraction`, con los rangos del notebook `z601_clases.ipynb`. `bagging_freq` está fijo en 1 y `max_bin` en 31, como en ese notebook. El resto de los hiperparámetros queda en el default de LightGBM.
- **Modelo final:** se entrena con los hiperparámetros del mejor trial y su cantidad de árboles, una vez por cada semilla.
- **Retomar:** si la VM se apaga, la búsqueda sigue desde el último trial guardado.

## Cómo correrlo en Google Cloud

Con el crudo en el bucket (`~/buckets/b1/datasets/competencia_02_crudo.csv.gz`), desde la carpeta del repo:

```
python preparar_datos.py --crudo ~/buckets/b1/datasets/competencia_02_crudo.csv.gz --salida ~/buckets/b1/datasets/competencia_02_2019.parquet
```

Antes de la primera corrida larga, una prueba corta con los datos reales (10% de los clientes, 2 trials y 2 semillas). Guarda todo en `<salida>/base_prueba/`, aparte de la corrida real:

```
python experimento.py --exp base --prueba --datos ~/buckets/b1/datasets/competencia_02_2019.parquet --salida ~/buckets/b1/exp/targets
```

Después, cada experimento en segundo plano:

```
nohup python experimento.py --exp base --datos ~/buckets/b1/datasets/competencia_02_2019.parquet --salida ~/buckets/b1/exp/targets > base.log 2>&1 &
```

`--trials` fija la cantidad de trials de la búsqueda (50 por defecto). Usa duckdb, lightgbm, optuna, pandas y pyarrow.

## Resultados de cada experimento

Quedan en `<salida>/<experimento>/`:

| Archivo | Contenido |
|---|---|
| `mejor_trial.json` | AUC en agosto, hiperparámetros y cantidad de árboles del mejor trial |
| `prediccion_julio.csv` | Predicción de julio del mejor trial. Ningún modelo vio julio: ahí se eligen los pesos del ensamble |
| `prediccion_agosto.csv` | Predicción de agosto del mejor trial, para confirmar los pesos del ensamble |
| `prediccion_octubre.csv` | Predicción de octubre de cada semilla |
| `ganancia_octubre.csv` | Ganancia de cada semilla en los cortes de 2.500 a 6.500 envíos, de a 250 |
| `optuna.db` | La búsqueda completa |

## Semillas

Usamos 20 semillas fijas, las mismas para todos los experimentos (lista `SEMILLAS` en `experimento.py`). Las 10 primeras son de Fernando y las 10 siguientes de Laura. La primera también es la semilla de Optuna.
