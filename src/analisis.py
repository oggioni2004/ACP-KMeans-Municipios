"""
Funciones del análisis: preparación, ACP (PCA) y K-Means.

La app de Streamlit importa este módulo, así que todo lo que se ve en la app
se calcula aquí, en vivo, a partir de data/municipios_2024.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "municipios_2024.csv"
SEMILLA = 42

# Variables que entran al modelo: nombre de columna -> (etiqueta, descripción, unidad)
VARIABLES = {
    "dependencia_transferencias": (
        "Dependencia de transferencias",
        "Parte de los ingresos totales que llega de la Nación (SGP y otras transferencias).",
        "% de ingresos",
    ),
    "inversion_fbkf": (
        "Inversión en capital fijo",
        "Parte del gasto que se va a formación bruta de capital fijo (obras, infraestructura).",
        "% del gasto",
    ),
    "endeudamiento_lp": (
        "Endeudamiento de largo plazo",
        "Saldo de deuda de largo plazo frente a los ingresos corrientes.",
        "% de ingresos corrientes",
    ),
    "ahorro_corriente": (
        "Ahorro corriente",
        "Lo que sobra de los ingresos corrientes después de pagar el funcionamiento.",
        "% de ingresos corrientes",
    ),
    "balance_primario": (
        "Balance fiscal primario",
        "Ingresos menos gastos, sin contar intereses de la deuda.",
        "% de ingresos",
    ),
    "ejecucion_ingresos": (
        "Ejecución de ingresos",
        "Ingresos recaudados frente a los que se presupuestaron (más de 100 = recaudó más de lo planeado).",
        "% del presupuesto",
    ),
    "ejecucion_inversion": (
        "Ejecución de inversión",
        "Inversión ejecutada frente a la presupuestada.",
        "% del presupuesto",
    ),
    "holgura": (
        "Holgura (Ley 617)",
        "Margen que le queda al municipio frente al tope legal de gastos de funcionamiento.",
        "puntos porcentuales",
    ),
    "predial_pc": (
        "Predial per cápita",
        "Recaudo del impuesto predial por habitante.",
        "pesos por habitante",
    ),
    "ica_pc": (
        "ICA per cápita",
        "Recaudo del impuesto de industria y comercio por habitante.",
        "pesos por habitante",
    ),
    "regalias_pc": (
        "Regalías per cápita",
        "Regalías efectivamente giradas al municipio por habitante.",
        "pesos por habitante",
    ),
}
VARS_PESOS = ["predial_pc", "ica_pc", "regalias_pc"]  # muy asimétricas -> log
ETIQUETA = {k: v[0] for k, v in VARIABLES.items()}

# Regiones del DNP (Plan Nacional de Desarrollo) por código de departamento
REGIONES = {
    "Caribe": ["08", "13", "20", "23", "44", "47", "70", "88"],
    "Pacífico": ["19", "27", "52", "76"],
    "Eje Cafetero y Antioquia": ["05", "17", "63", "66"],
    "Centro Oriente": ["11", "15", "25", "54", "68"],
    "Centro Sur y Amazonía": ["18", "41", "73", "86", "91", "94", "97"],
    "Llanos": ["50", "81", "85", "95", "99"],
}
DEPTO_A_REGION = {d: r for r, ds in REGIONES.items() for d in ds}


def cargar_datos():
    df = pd.read_csv(DATOS, dtype={"codigo": str, "cod_depto": str})
    df["region"] = df["cod_depto"].map(DEPTO_A_REGION)
    return df


def preparar(df, variables, usar_log=True, recorte=0.01):
    """Devuelve (X_transformada sin estandarizar, X_estandarizada, scaler).

    - usar_log: aplica log(1 + x) a las variables en pesos per cápita, que están
      muy sesgadas (pocos municipios con valores enormes).
    - recorte: winsorización. Lleva los valores por debajo del percentil `recorte`
      y por encima del percentil 1-`recorte` a esos límites, para que unos pocos
      datos extremos no dominen las distancias de K-Means.
    """
    X = df[variables].copy().astype(float)
    if usar_log:
        for c in variables:
            if c in VARS_PESOS:
                X[c] = np.log1p(X[c])
    if recorte and recorte > 0:
        X = X.clip(X.quantile(recorte), X.quantile(1 - recorte), axis=1)
    scaler = StandardScaler()
    Z = pd.DataFrame(scaler.fit_transform(X), columns=variables, index=df.index)
    return X, Z, scaler


def ajustar_pca(Z):
    pca = PCA(random_state=SEMILLA).fit(Z)
    var = pd.DataFrame(
        {
            "Componente": [f"CP{i+1}" for i in range(pca.n_components_)],
            "Valor propio": pca.explained_variance_,
            "Varianza explicada": pca.explained_variance_ratio_,
            "Varianza acumulada": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    cargas = pd.DataFrame(
        pca.components_.T,
        index=Z.columns,
        columns=[f"CP{i+1}" for i in range(pca.n_components_)],
    )
    puntajes = pd.DataFrame(
        pca.transform(Z),
        index=Z.index,
        columns=[f"CP{i+1}" for i in range(pca.n_components_)],
    )
    return pca, var, cargas, puntajes


def sugerir_componentes(var, umbral=0.70):
    """Dos criterios clásicos: Kaiser (valor propio > 1) y varianza acumulada >= umbral."""
    kaiser = int((var["Valor propio"] > 1).sum())
    acumulado = int((var["Varianza acumulada"] < umbral).sum() + 1)
    return kaiser, acumulado


def evaluar_k(datos, k_min=2, k_max=10):
    filas = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(datos)
        filas.append(
            {
                "K": k,
                "Inercia": km.inertia_,
                "Silueta": silhouette_score(datos, km.labels_),
            }
        )
    return pd.DataFrame(filas)


def ajustar_kmeans(datos, k):
    km = KMeans(n_clusters=k, n_init=20, random_state=SEMILLA).fit(datos)
    # Se renumeran los grupos por tamaño (A = el más grande) para que la
    # etiqueta no cambie entre corridas; la letra NO implica orden ni ranking.
    tam = pd.Series(km.labels_).value_counts()
    mapa = {g: chr(65 + i) for i, g in enumerate(tam.index)}
    etiquetas = pd.Series(km.labels_).map(mapa).values
    sil = silhouette_samples(datos, km.labels_)
    return km, etiquetas, sil


def perfil_grupos(df, etiquetas, variables):
    """Mediana de las variables ORIGINALES por grupo (la mediana resiste extremos)."""
    d = df.copy()
    d["Grupo"] = etiquetas
    perfil = d.groupby("Grupo")[variables + ["puntaje_idf"]].median()
    perfil.insert(0, "Municipios", d.groupby("Grupo").size())
    return perfil
