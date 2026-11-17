"""
Prepara la base de datos del proyecto a partir de TerriData (DNP).

Qué hace:
1. Descarga el archivo plano de la dimensión "Finanzas públicas" de TerriData
   (si no está ya en data/raw/).
2. Se queda con el año 2024 y con los municipios (código DIVIPOLA de 5 dígitos).
3. Pasa los datos de formato largo (una fila por indicador) a formato ancho
   (una fila por municipio, una columna por indicador).
4. Guarda data/municipios_2024.csv, que es la base que usa la app.

Uso (desde la raíz del repositorio):
    python src/preparar_datos.py
"""

from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "data" / "raw"
SALIDA = RAIZ / "data" / "municipios_2024.csv"

URL = "https://terridata.dnp.gov.co/assets/docs/txt/dimensiones/TerriData_Dim7.txt.zip"
ANIO = "2024"

# Indicador original de TerriData -> nombre corto de columna
INDICADORES = {
    # Nuevo Índice de Desempeño Fiscal (IDF) - variables del modelo
    "Dependencia de las transferencias": "dependencia_transferencias",
    "Relevancia Formación Bruta de Capital fijo": "inversion_fbkf",
    "Endeudamiento de Largo Plazo": "endeudamiento_lp",
    "Ahorro Corriente": "ahorro_corriente",
    "Balance Fiscal Primario": "balance_primario",
    "Capacidad de Ejecución de Ingresos": "ejecucion_ingresos",
    "Capacidad de Ejecución de Inversión": "ejecucion_inversion",
    "Nivel de Holgura": "holgura",
    # Recaudo y regalías per cápita (pesos corrientes) - variables del modelo
    "Ingresos per cápita por impuesto predial": "predial_pc",
    "Ingresos per cápita por impuesto a la Industria y al comercio": "ica_pc",
    "Regalías per cápita (Valor efectivamente girado al municipio)": "regalias_pc",
    # Resultados del IDF - NO entran al modelo, se usan para describir los grupos
    "Puntaje nuevo Índice de Desempeño Fiscal": "puntaje_idf",
    "Componente de resultados fiscales": "comp_resultados",
    "Componente de gestión financiera": "comp_gestion",
}


def descargar():
    RAW.mkdir(parents=True, exist_ok=True)
    txt = RAW / "TerriData_Dim7.txt"
    if txt.exists():
        return txt
    zip_path = RAW / "TerriData_Dim7.txt.zip"
    print("Descargando TerriData (unos 30 MB)...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW)
    return txt


def a_numero(serie):
    # TerriData usa formato colombiano: miles con "." y decimales con ","
    return pd.to_numeric(
        serie.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def main():
    txt = descargar()
    print("Leyendo archivo plano...")
    df = pd.read_csv(txt, sep="|", dtype=str)

    df = df[(df["Año"] == ANIO) & (df["Indicador"].isin(INDICADORES))]
    # Solo municipios: código DIVIPOLA de 5 dígitos. Se excluyen el total
    # nacional (01001) y los departamentos (códigos que terminan en 000).
    cod = df["Código Entidad"]
    df = df[(cod.str.len() == 5) & (~cod.str.endswith("000")) & (cod != "01001")]
    df["valor"] = a_numero(df["Dato Numérico"])
    df["variable"] = df["Indicador"].map(INDICADORES)

    ancho = df.pivot_table(
        index=["Código Entidad", "Entidad", "Código Departamento", "Departamento"],
        columns="variable",
        values="valor",
        aggfunc="first",
    ).reset_index()
    ancho.columns.name = None
    ancho = ancho.rename(
        columns={
            "Código Entidad": "codigo",
            "Entidad": "municipio",
            "Código Departamento": "cod_depto",
            "Departamento": "departamento",
        }
    )
    ancho = ancho[["codigo", "municipio", "cod_depto", "departamento"] + list(INDICADORES.values())]
    ancho.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"Guardado {SALIDA.relative_to(RAIZ)}: {ancho.shape[0]} municipios x {ancho.shape[1]} columnas")


if __name__ == "__main__":
    main()
