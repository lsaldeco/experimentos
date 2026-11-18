# Arma el dataset de los experimentos: solo 2019, con clase_ternaria, clase_extendida
# (agrega BAJA+3) y lags y deltas de 1 y 2 meses. Se corre una sola vez.
import argparse

import duckdb

parser = argparse.ArgumentParser()
parser.add_argument("--crudo", default="competencia_02_crudo.csv.gz")
parser.add_argument("--salida", default="competencia_02_2019.parquet")
args = parser.parse_args()

ID = ["numero_de_cliente", "foto_mes"]
TIPOS_NUMERICOS = ("BIGINT", "INTEGER", "SMALLINT", "TINYINT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT")

con = duckdb.connect()

# 1. Solo 2019. Los lags de marzo necesitan enero y febrero, y las clases de octubre necesitan diciembre.
#    Las filas que no se pueden leer quedan registradas en reject_errors en lugar de perderse sin aviso.
con.execute(f"""
CREATE TABLE crudo AS
SELECT *
FROM read_csv('{args.crudo}', auto_detect = true, null_padding = true,
              ignore_errors = true, store_rejects = true)
WHERE foto_mes BETWEEN 201901 AND 201912
""")
rechazadas = con.execute("SELECT count(*) FROM reject_errors").fetchone()[0]
print(f"Filas que no se pudieron leer (en todo el archivo): {rechazadas}")
print(con.execute("SELECT foto_mes, count(*) AS clientes FROM crudo GROUP BY 1 ORDER BY 1").df())

# 2. Clases. mes_2 se evalúa primero: un cliente que falta un mes y vuelve (1 -> 0 -> 1) es CONTINUA.
#    BAJA+3 sale de los CONTINUA que siguen en el mes 2 y no están en el mes 3.
con.execute("""
CREATE TABLE con_clase AS
WITH clientes AS (
    SELECT DISTINCT numero_de_cliente FROM crudo
),
meses AS (
    SELECT DISTINCT foto_mes FROM crudo
),
presencia AS (
    SELECT c.numero_de_cliente, m.foto_mes,
           CASE WHEN k.numero_de_cliente IS NULL THEN 0 ELSE 1 END AS mes_0
    FROM clientes c
    CROSS JOIN meses m
    LEFT JOIN (SELECT numero_de_cliente, foto_mes FROM crudo) k
           ON k.numero_de_cliente = c.numero_de_cliente AND k.foto_mes = m.foto_mes
),
futuro AS (
    SELECT numero_de_cliente, foto_mes, mes_0,
           LEAD(mes_0, 1) OVER w AS mes_1,
           LEAD(mes_0, 2) OVER w AS mes_2,
           LEAD(mes_0, 3) OVER w AS mes_3
    FROM presencia
    WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
),
clases AS (
    SELECT numero_de_cliente, foto_mes, mes_1, mes_2, mes_3,
           CASE
               WHEN mes_2 = 1 THEN 'CONTINUA'
               WHEN mes_1 = 0 THEN 'BAJA+1'
               WHEN mes_1 = 1 AND mes_2 = 0 THEN 'BAJA+2'
           END AS clase_ternaria
    FROM futuro
    WHERE mes_0 = 1
)
SELECT k.*,
       c.clase_ternaria,
       CASE
           WHEN c.mes_1 = 1 AND c.mes_2 = 1 AND c.mes_3 = 0 THEN 'BAJA+3'
           ELSE c.clase_ternaria
       END AS clase_extendida
FROM crudo k
JOIN clases c USING (numero_de_cliente, foto_mes)
""")
print(con.execute("""
SELECT foto_mes, clase_extendida, count(*) AS n
FROM con_clase GROUP BY ALL ORDER BY foto_mes, clase_extendida
""").df().pivot(index="foto_mes", columns="clase_extendida", values="n"))

# 3. Lags y deltas de 1 y 2 meses de todas las variables numéricas. Sin rellenar nulos:
#    si no hay mes anterior, el lag y el delta quedan en NULL.
columnas = con.execute("DESCRIBE crudo").df()
numericas = [
    fila.column_name for fila in columnas.itertuples()
    if fila.column_name not in ID and fila.column_type.startswith(TIPOS_NUMERICOS)
]
historicas = []
for c in numericas:
    historicas += [
        f'LAG("{c}", 1) OVER w AS "{c}_lag1"',
        f'LAG("{c}", 2) OVER w AS "{c}_lag2"',
        f'"{c}" - LAG("{c}", 1) OVER w AS "{c}_delta1"',
        f'"{c}" - LAG("{c}", 2) OVER w AS "{c}_delta2"',
    ]
print(f"Variables con lags y deltas: {len(numericas)}")

# 4. Control de meses rotos: proporción de ceros y de nulos de cada variable en cada mes.
#    Se marcan los meses que se alejan más de 0,3 de la mediana de esa variable en el año.
UMBRAL_MES_ROTO = 0.3
agregados = []
for c in numericas:
    agregados += [f'avg(CASE WHEN "{c}" = 0 THEN 1 ELSE 0 END) AS "{c}__ceros"',
                  f'avg(CASE WHEN "{c}" IS NULL THEN 1 ELSE 0 END) AS "{c}__nulos"']
control = con.execute(f"SELECT foto_mes, {', '.join(agregados)} FROM crudo GROUP BY 1 ORDER BY 1").df()
control = control.melt(id_vars="foto_mes", var_name="medida", value_name="proporcion")
control[["variable", "tipo"]] = control["medida"].str.split("__", expand=True)
control["desvio"] = (control["proporcion"]
                     - control.groupby("medida")["proporcion"].transform("median")).abs()
control.drop(columns="medida").to_csv("control_meses.csv", index=False)
sospechosos = control[control["desvio"] > UMBRAL_MES_ROTO]
print(f"Variable-mes sospechosos (detalle en control_meses.csv): {len(sospechosos)}")
print(sospechosos[["foto_mes", "variable", "tipo", "proporcion", "desvio"]].to_string(index=False))

con.execute(f"""
COPY (
    SELECT *, {", ".join(historicas)}
    FROM con_clase
    WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
) TO '{args.salida}' (FORMAT PARQUET)
""")
print(f"Guardado en {args.salida}")
