"""
Perfiles fiscales de los municipios colombianos (2024)
ACP + K-Means · Machine Learning aplicado a Economía y Finanzas · PUJ Cali

Ejecutar:  streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from analisis import (  # noqa: E402
    ETIQUETA,
    VARIABLES,
    VARS_PESOS,
    ajustar_kmeans,
    ajustar_pca,
    cargar_datos,
    evaluar_k,
    perfil_grupos,
    preparar,
    robustez,
    sugerir_componentes,
)

st.set_page_config(
    page_title="Perfiles fiscales municipales · ACP y K-Means",
    page_icon="🏛️",
    layout="wide",
)

PALETA = ["#1F6F78", "#E07A5F", "#6A4C93", "#F2B134", "#3D9970", "#C0392B", "#2E86AB", "#8D6E63", "#7F8C8D", "#D35400"]
VALLE = "76"
CFG_BASE = dict(usar_log=True, recorte=0.01, n_comp=4, k=4)

# Nombres descriptivos de los grupos de la configuración recomendada (4 CP, K=4).
# Las letras se asignan por tamaño del grupo; NO son un ranking.
NOMBRES_BASE = {
    "A": "Caja apretada y poca inversión",
    "B": "Dependientes pero ordenados",
    "C": "Municipios de regalías e inversión",
    "D": "Autónomos con recaudo propio",
}


# ----------------------------------------------------------------------------
# Datos y cálculos (con caché para que la app sea rápida)
# ----------------------------------------------------------------------------
@st.cache_data
def datos():
    return cargar_datos()


@st.cache_data
def pipeline(variables, usar_log, recorte):
    df = datos()
    X, Z, _ = preparar(df, list(variables), usar_log=usar_log, recorte=recorte)
    _, var, cargas, puntajes = ajustar_pca(Z)
    return X, Z, var, cargas, puntajes


@st.cache_data
def curva_k(variables, usar_log, recorte, n_comp, k_max=10):
    _, _, _, _, puntajes = pipeline(variables, usar_log, recorte)
    return evaluar_k(puntajes.iloc[:, :n_comp], 2, k_max)


@st.cache_data(show_spinner=False)
def pruebas_robustez(variables, usar_log, recorte, n_comp, k):
    df = datos()
    return robustez(df, list(variables), usar_log=usar_log, recorte=recorte, n_comp=n_comp, k=k,
                    es_valle=(df["cod_depto"] == VALLE).values)


@st.cache_data
def segmentar(variables, usar_log, recorte, n_comp, k):
    _, _, _, _, puntajes = pipeline(variables, usar_log, recorte)
    km, etiquetas, sil = ajustar_kmeans(puntajes.iloc[:, :n_comp], k)
    return etiquetas, sil


def fmt_num(x, dec=1):
    """Formato colombiano: miles con punto y decimales con coma."""
    s = f"{x:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def es_config_base(cfg, variables):
    return (
        tuple(variables) == tuple(VARIABLES)
        and cfg["usar_log"] == CFG_BASE["usar_log"]
        and cfg["recorte"] == CFG_BASE["recorte"]
        and cfg["n_comp"] == CFG_BASE["n_comp"]
        and cfg["k"] == CFG_BASE["k"]
    )


def nombre_grupo(g, base):
    return f"{g} · {NOMBRES_BASE[g]}" if base else f"Grupo {g}"


# ----------------------------------------------------------------------------
# Barra lateral: navegación y parámetros del modelo
# ----------------------------------------------------------------------------
df = datos()

st.sidebar.title("🏛️ Perfiles fiscales municipales")
st.sidebar.caption("Colombia · Nuevo IDF 2024 · ACP + K-Means")

pagina = st.sidebar.radio(
    "Secciones",
    [
        "1. Contexto",
        "2. Exploración de datos",
        "3. Preparación",
        "4. Componentes principales (ACP)",
        "5. Segmentación (K-Means)",
        "6. Lupa al Valle del Cauca",
        "7. Conclusiones",
    ],
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Parámetros del modelo")
st.sidebar.caption(
    "Todo se recalcula en vivo. Los valores por defecto son la configuración "
    "recomendada que se justifica en cada sección."
)
variables = st.sidebar.multiselect(
    "Variables del modelo",
    options=list(VARIABLES),
    default=list(VARIABLES),
    format_func=lambda c: ETIQUETA[c],
)
if len(variables) < 4:
    st.sidebar.error("Selecciona al menos 4 variables.")
    st.stop()
variables = tuple(variables)

usar_log = st.sidebar.toggle("Logaritmo en variables en pesos per cápita", value=True)
recorte = st.sidebar.select_slider(
    "Recorte de extremos (winsorización)",
    options=[0.0, 0.005, 0.01, 0.025, 0.05],
    value=0.01,
    format_func=lambda x: "Sin recorte" if x == 0 else f"Percentiles {x*100:g}% y {100-x*100:g}%",
)
X, Z, var, cargas, puntajes = pipeline(variables, usar_log, recorte)
kaiser, acum70 = sugerir_componentes(var)
n_comp = st.sidebar.slider(
    "Componentes para K-Means",
    2,
    len(variables),
    min(CFG_BASE["n_comp"], len(variables)),
    help=f"Kaiser sugiere {kaiser}; para llegar al 70% de varianza se necesitan {acum70}.",
)
k = st.sidebar.slider("Número de grupos (K)", 2, 10, CFG_BASE["k"])
cfg = dict(usar_log=usar_log, recorte=recorte, n_comp=n_comp, k=k)
base = es_config_base(cfg, variables)

etiquetas, sil = segmentar(variables, usar_log, recorte, n_comp, k)
res = df.copy()
res["Grupo"] = etiquetas
res["Silueta"] = sil
res = pd.concat([res, puntajes.iloc[:, : max(n_comp, 2)]], axis=1)
res["Nombre grupo"] = [nombre_grupo(g, base) for g in etiquetas]
orden_grupos = sorted(res["Grupo"].unique())
color_grupo = {nombre_grupo(g, base): PALETA[i % len(PALETA)] for i, g in enumerate(orden_grupos)}

if not base:
    st.sidebar.info(
        "Estás explorando una configuración distinta a la recomendada: los grupos "
        "se nombran solo con letras y las conclusiones escritas (sección 7) "
        "corresponden a la configuración por defecto."
    )
st.sidebar.divider()
st.sidebar.caption(
    "Fuente: DNP – TerriData, dimensión Finanzas públicas (Nuevo IDF y "
    "Operaciones Efectivas de Caja), vigencia 2024."
)


# ----------------------------------------------------------------------------
# 1. Contexto
# ----------------------------------------------------------------------------
if pagina.startswith("1"):
    st.title("¿Qué perfiles financieros tienen los municipios colombianos?")
    st.markdown("#### …y ¿dónde queda el Valle del Cauca?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Municipios", fmt_num(len(df), 0))
    c2.metric("Variables del modelo", len(VARIABLES))
    c3.metric("Vigencia", "2024")
    c4.metric("Municipios del Valle", int((df["cod_depto"] == VALLE).sum()))

    st.markdown(
        """
**El problema.** Los 1.102 municipios de esta base manejan presupuestos propios, pero sus finanzas son muy
distintas: unos viven casi solo de lo que les gira la Nación, otros recaudan bastante predial e ICA, otros
reciben regalías por minería o petróleo y otros están cerca del tope legal de gastos. El Departamento
Nacional de Planeación (DNP) resume todo esto en un solo número, el **Índice de Desempeño Fiscal (IDF)**,
y con él arma un ranking. El problema es que **un mismo puntaje puede esconder situaciones muy distintas**:
un municipio puede sacar 55 puntos por depender de transferencias aunque esté ordenado, y otro puede sacar
55 por tener la caja apretada aunque recaude bien.

**La pregunta.** En lugar de ordenar a los municipios en una sola fila, queremos ver **qué tipos de
municipio existen según su estructura financiera**, en cuántas dimensiones se pueden resumir sus
diferencias y **en qué tipo caen los municipios del Valle del Cauca**.

**Qué representa cada observación.** Un municipio colombiano en la vigencia fiscal 2024.
        """
    )

    st.subheader("¿Por qué ACP y K-Means?")
    a, b = st.columns(2)
    a.markdown(
        """
**ACP (Análisis de Componentes Principales).** Las 11 variables financieras están correlacionadas
(por ejemplo, quien recauda más predial suele recaudar más ICA y depender menos de las transferencias).
El ACP junta esa información redundante en pocas **dimensiones de fondo**, como "autonomía fiscal" o
"regalías e inversión". Así podemos ver la estructura en vez de mirar 11 columnas a la vez.
        """
    )
    b.markdown(
        """
**K-Means.** Una vez resumidas las dimensiones, K-Means busca **grupos de municipios con perfiles
parecidos**. Eso sirve para la política pública: un municipio que depende de las transferencias necesita
cosas distintas de uno que vive de regalías, aunque los dos tengan el mismo puntaje en el IDF.
        """
    )

    st.subheader("Fuente, cobertura y limitaciones")
    st.markdown(
        """
- **Fuente:** [TerriData – DNP](https://terridata.dnp.gov.co/), descarga masiva de la dimensión *Finanzas
  públicas*. Los indicadores vienen del **Nuevo Índice de Desempeño Fiscal 2024** (metodología del DNP
  desde 2020) y de las **Operaciones Efectivas de Caja** (recaudo per cápita), que el DNP construye a partir
  de los reportes CUIPO y FUT de los municipios.
- **Cobertura:** 1.102 municipios con información completa para 2024 (el país tiene 1.103 contando Bogotá).
  Se excluyen los totales nacionales y departamentales.
- **Limitaciones:** (1) son datos **reportados por los propios municipios**, que pueden tener errores o
  rezagos; (2) es **un solo año**, una foto y no una tendencia; (3) 2024 fue el primer año de los nuevos
  alcaldes, así que la ejecución puede verse afectada por el empalme; (4) los valores per cápita dependen de
  las proyecciones de población del DANE; (5) las regalías son volátiles porque dependen de precios del
  petróleo y de minerales.
        """
    )


# ----------------------------------------------------------------------------
# 2. Exploración de datos
# ----------------------------------------------------------------------------
elif pagina.startswith("2"):
    st.title("Exploración de los datos")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas (municipios)", fmt_num(len(df), 0))
    c2.metric("Columnas", df.shape[1])
    c3.metric("Valores faltantes", int(df[list(VARIABLES)].isna().sum().sum()))
    c4.metric("Códigos duplicados", int(df["codigo"].duplicated().sum()))

    st.subheader("Diccionario de variables")
    dicc = pd.DataFrame(
        [(ETIQUETA[c], c, VARIABLES[c][1], VARIABLES[c][2]) for c in VARIABLES],
        columns=["Variable", "Columna", "Qué mide", "Unidad"],
    )
    st.dataframe(dicc, hide_index=True, width="stretch")
    st.caption(
        "No entran al modelo: el código DIVIPOLA y el nombre (son identificadores), el departamento y la región "
        "(se usan después para leer los grupos), y el puntaje del IDF con sus componentes (son un resumen de "
        "estas mismas variables; meterlos repetiría información). El puntaje se usa al final para contrastar "
        "los grupos."
    )

    st.subheader("Estadísticas descriptivas (valores originales)")
    desc = df[list(VARIABLES)].describe().T
    desc.index = [ETIQUETA[c] for c in desc.index]
    desc["asimetría"] = df[list(VARIABLES)].skew().values
    st.dataframe(desc.round(2), width="stretch")
    st.caption(
        "Fíjate en la **asimetría**: las variables en pesos per cápita (predial, ICA, regalías) tienen asimetrías "
        "muy altas, porque pocos municipios tienen valores enormes. Además hay extremos como una ejecución de "
        "ingresos de 822% o un balance primario de −203%. Estos dos problemas se tratan en *Preparación*."
    )

    st.subheader("Distribución de cada variable")
    v_sel = st.selectbox("Variable", list(VARIABLES), format_func=lambda c: ETIQUETA[c])
    fig = px.histogram(
        df, x=v_sel, nbins=60, color_discrete_sequence=[PALETA[0]],
        labels={v_sel: f"{ETIQUETA[v_sel]} ({VARIABLES[v_sel][2]})"},
        title=f"Distribución de {ETIQUETA[v_sel].lower()} en 1.102 municipios",
    )
    fig.update_layout(yaxis_title="Número de municipios", bargap=0.02)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Matriz de correlaciones")
    corr = X.corr()
    corr.index = corr.columns = [ETIQUETA[c] for c in corr.columns]
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Correlación de Pearson entre variables (después de la transformación logarítmica)",
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        """
**Lectura.** El bloque más fuerte es el de la **autonomía fiscal**: la dependencia de transferencias se
correlaciona negativamente con el ICA (≈ −0,78) y con el predial (≈ −0,69), y esos dos impuestos se
mueven juntos (≈ +0,61). El segundo bloque es el de la **solvencia del funcionamiento**: ahorro corriente y
holgura se mueven juntos (≈ +0,64). Las regalías van con la inversión en capital fijo (≈ +0,44). Que haya
bloques de variables correlacionadas es justamente lo que hace útil al ACP.
        """
    )


# ----------------------------------------------------------------------------
# 3. Preparación
# ----------------------------------------------------------------------------
elif pagina.startswith("3"):
    st.title("Preparación de los datos")
    st.markdown(
        """
| Problema encontrado | Decisión | Por qué |
|---|---|---|
| La base viene en formato largo (una fila por indicador) y mezcla municipios, departamentos y el total nacional | Se pasa a formato ancho y se dejan solo los códigos municipales de 5 dígitos | Cada fila debe ser un municipio |
| Valores faltantes | No quedan (0 en las 11 variables) | Los 33 registros sin recaudo per cápita eran totales departamentales y el nacional |
| Duplicados | No hay códigos repetidos | Hay nombres repetidos (por ejemplo, *Bolívar* existe en varios departamentos), por eso el identificador es el código DIVIPOLA |
| Predial, ICA y regalías per cápita muy sesgados | Se aplica **log(1 + x)** | Comprime los valores enormes y deja ver diferencias entre municipios pequeños; el +1 evita problemas con los ceros (muchos municipios no reciben regalías) |
| Valores extremos (por ejemplo, ejecución de ingresos de 822%) | **Winsorización** al percentil 1% y 99% | K-Means usa distancias y la media; unos pocos extremos pueden "jalar" un grupo entero. Recortar no borra municipios, solo limita su peso |
| Variables en escalas distintas (porcentajes y pesos) | **Estandarización** (media 0 y desviación 1) | ACP y K-Means dependen de la escala: sin estandarizar, la variable con números más grandes dominaría |
        """
    )
    st.info(
        "En la barra lateral puedes desactivar el logaritmo o cambiar el recorte y ver cómo cambian el ACP y los "
        "grupos. Ese análisis de sensibilidad se comenta en las conclusiones."
    )

    st.subheader("Antes y después")
    v_sel = st.selectbox("Variable", list(VARIABLES), index=list(VARIABLES).index("regalias_pc"),
                         format_func=lambda c: ETIQUETA[c])
    if v_sel in variables:
        a, b = st.columns(2)
        f1 = px.histogram(df, x=v_sel, nbins=50, color_discrete_sequence=[PALETA[1]],
                          title="Original", labels={v_sel: f"{ETIQUETA[v_sel]} ({VARIABLES[v_sel][2]})"})
        f1.update_layout(yaxis_title="Municipios")
        f2 = px.histogram(Z, x=v_sel, nbins=50, color_discrete_sequence=[PALETA[0]],
                          title="Transformada y estandarizada (la que entra al modelo)",
                          labels={v_sel: f"{ETIQUETA[v_sel]} (desviaciones estándar)"})
        f2.update_layout(yaxis_title="Municipios")
        a.plotly_chart(f1, width="stretch")
        b.plotly_chart(f2, width="stretch")
    else:
        st.warning("Esa variable no está seleccionada en el modelo.")

    st.subheader("Datos estandarizados (primeras filas)")
    vista = pd.concat([df[["municipio", "departamento"]], Z], axis=1).head(15)
    vista.columns = ["Municipio", "Departamento"] + [ETIQUETA[c] for c in Z.columns]
    st.dataframe(vista.round(2), hide_index=True, width="stretch")
    st.caption(f"Comprobación: medias ≈ {Z.mean().abs().max():.1e} y desviaciones ≈ {Z.std(ddof=0).mean():.2f}.")


# ----------------------------------------------------------------------------
# 4. ACP
# ----------------------------------------------------------------------------
elif pagina.startswith("4"):
    st.title("Análisis de Componentes Principales (ACP)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Criterio de Kaiser (valor propio > 1)", f"{kaiser} componentes")
    c2.metric("Para llegar al 70% de varianza", f"{acum70} componentes")
    c3.metric(f"Varianza conservada con {n_comp} CP",
              f"{fmt_num(var['Varianza acumulada'].iloc[n_comp-1]*100)}%")

    st.subheader("Varianza explicada y gráfico de sedimentación")
    fig = go.Figure()
    fig.add_bar(x=var["Componente"], y=var["Varianza explicada"] * 100, name="Varianza de cada componente",
                marker_color=PALETA[0], text=(var["Varianza explicada"] * 100).round(1), textposition="outside")
    fig.add_scatter(x=var["Componente"], y=var["Varianza acumulada"] * 100, name="Varianza acumulada",
                    mode="lines+markers", line=dict(color=PALETA[1], width=3))
    fig.add_hline(y=70, line_dash="dot", annotation_text="70%")
    fig.add_vline(x=n_comp - 0.5, line_dash="dash", line_color="gray",
                  annotation_text=f"Se conservan {n_comp}")
    fig.update_layout(title="Gráfico de sedimentación (scree plot)", xaxis_title="Componente principal",
                      yaxis_title="% de la varianza total", legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    tabla = var.copy()
    tabla["Varianza explicada"] = (tabla["Varianza explicada"] * 100).round(1)
    tabla["Varianza acumulada"] = (tabla["Varianza acumulada"] * 100).round(1)
    st.dataframe(tabla.round(3), hide_index=True, width="stretch")

    st.markdown(
        f"""
**¿Cuántos componentes conservar?** La varianza **no** se concentra en uno o dos componentes: el primero
explica cerca del {fmt_num(var['Varianza explicada'].iloc[0]*100, 0)}% y el segundo cerca del
{fmt_num(var['Varianza explicada'].iloc[1]*100, 0)}%. Eso ya dice algo: las finanzas municipales tienen
**varias dimensiones** que no se reducen a una sola. Se conservan **4 componentes** por tres razones:
(1) son los que cumplen el criterio de Kaiser (valor propio mayor que 1, es decir, cada uno resume más
información que una variable original); (2) juntos conservan cerca del 66% de la varianza; (3) el quinto
componente casi no tiene una lectura económica clara y K-Means con 5 componentes da grupos prácticamente
iguales (en las pruebas de robustez, el índice ARI fue 0,93).
        """
    )

    st.subheader("Cargas: ¿qué significa cada componente?")
    nc_show = min(max(n_comp, 4), len(variables))
    car = cargas.iloc[:, :nc_show].copy()
    car.index = [ETIQUETA[c] for c in car.index]
    fig = px.imshow(car, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-0.8, zmax=0.8, aspect="auto",
                    title="Cargas de cada variable en los componentes (rojo = positivo, azul = negativo)")
    fig.update_layout(height=520)
    st.plotly_chart(fig, width="stretch")
    if tuple(variables) == tuple(VARIABLES):
        st.markdown(
            """
| Componente | Variables que más pesan | Interpretación |
|---|---|---|
| **CP1 – Dependencia vs. autonomía fiscal** | Dependencia de transferencias (+), ICA y predial per cápita (−), ahorro corriente (−) | Valores altos indican un municipio que vive de la Nación y recauda poco. Valores bajos indican un municipio con base tributaria propia |
| **CP2 – Regalías y holgura vs. recaudo urbano** | Inversión en capital fijo, regalías, holgura y ahorro (+); predial e ICA (−) | Separa a los municipios que tienen recursos para invertir (regalías) y margen de funcionamiento de los que se sostienen con impuestos urbanos |
| **CP3 – Regalías e inversión vs. solvencia del funcionamiento** | Regalías e inversión (+); holgura, ahorro y ejecución de inversión (−) | Muestra municipios que invierten mucho gracias a las regalías aunque tengan poco margen en su gasto corriente |
| **CP4 – Endeudamiento** | Endeudamiento de largo plazo (+), balance primario (−) | Aísla a los municipios endeudados y con déficit |

*Nota:* el signo de un componente es arbitrario (multiplicarlo por −1 no cambia nada). Lo que importa es
qué variables van en la misma dirección y cuáles en direcciones opuestas.
            """
        )

    st.subheader("Municipios en el plano de componentes")
    cx, cy = st.columns(2)
    eje_x = cx.selectbox("Eje horizontal", list(puntajes.columns), index=0)
    eje_y = cy.selectbox("Eje vertical", list(puntajes.columns), index=1)
    ix, iy = list(puntajes.columns).index(eje_x), list(puntajes.columns).index(eje_y)
    pct = (var["Varianza explicada"].iloc[ix] + var["Varianza explicada"].iloc[iy]) * 100
    plano = pd.concat([df[["municipio", "departamento"]], puntajes[[eje_x, eje_y]]], axis=1)
    plano["Valle del Cauca"] = np.where(df["cod_depto"] == VALLE, "Valle del Cauca", "Resto del país")
    fig = px.scatter(
        plano, x=eje_x, y=eje_y, color="Valle del Cauca", hover_name="municipio",
        hover_data={"departamento": True, eje_x: ":.2f", eje_y: ":.2f", "Valle del Cauca": False},
        color_discrete_map={"Valle del Cauca": PALETA[1], "Resto del país": "#B0BEC5"}, opacity=0.8,
        title=f"Biplot {eje_x} vs {eje_y}: resume el {fmt_num(pct)}% de la información total",
        labels={eje_x: f"{eje_x} ({fmt_num(var['Varianza explicada'].iloc[ix]*100)}%)",
                eje_y: f"{eje_y} ({fmt_num(var['Varianza explicada'].iloc[iy]*100)}%)"},
    )
    escala = np.abs(puntajes[[eje_x, eje_y]]).max().max() * 0.8
    for v in cargas.index:
        x1, y1 = cargas.loc[v, eje_x] * escala, cargas.loc[v, eje_y] * escala
        fig.add_annotation(x=x1, y=y1, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor="#37474F", arrowwidth=1.5)
        fig.add_annotation(x=x1 * 1.12, y=y1 * 1.12, text=ETIQUETA[v], showarrow=False,
                           font=dict(size=11, color="#263238"))
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Cada punto es un municipio. Las flechas muestran hacia dónde crece cada variable original en este plano. "
        f"**Ojo:** este gráfico solo muestra el {fmt_num(pct)}% de la información; dos municipios que se ven "
        "cerca aquí pueden estar lejos en los componentes que no se dibujan."
    )


# ----------------------------------------------------------------------------
# 5. K-Means
# ----------------------------------------------------------------------------
elif pagina.startswith("5"):
    st.title("Segmentación con K-Means")
    st.markdown(
        f"K-Means se aplica sobre los **{n_comp} primeros componentes principales** (no sobre las 11 variables). "
        "Así cada dimensión de fondo pesa lo que debe y el ruido de los componentes menores no distorsiona las "
        "distancias."
    )

    curva = curva_k(variables, usar_log, recorte, n_comp)
    a, b = st.columns(2)
    f1 = px.line(curva, x="K", y="Inercia", markers=True, title="Método del codo",
                 labels={"Inercia": "Inercia (suma de distancias² dentro de los grupos)", "K": "Número de grupos K"})
    f1.update_traces(line_color=PALETA[0])
    f1.add_vline(x=k, line_dash="dash", line_color=PALETA[1])
    f2 = px.line(curva, x="K", y="Silueta", markers=True, title="Coeficiente de silueta promedio",
                 labels={"Silueta": "Silueta promedio (−1 a 1, más alto = mejor separado)", "K": "Número de grupos K"})
    f2.update_traces(line_color=PALETA[2])
    f2.add_vline(x=k, line_dash="dash", line_color=PALETA[1])
    a.plotly_chart(f1, width="stretch")
    b.plotly_chart(f2, width="stretch")
    tabla_k = curva.copy()
    tabla_k["Inercia"] = tabla_k["Inercia"].round(0)
    tabla_k["Silueta"] = tabla_k["Silueta"].round(3)
    st.dataframe(tabla_k, hide_index=True, width="stretch")

    st.markdown(
        """
**¿Cómo se eligió K = 4?** El codo no es muy marcado: la inercia cae fuerte hasta K = 3 o 4 y después
baja más despacio. La silueta es más alta en K = 2 (≈ 0,25), pero esa partición solo separa a los municipios
"autónomos" de los "dependientes", algo que el CP1 ya muestra. **K = 3 y K = 4 tienen casi la misma
silueta (0,202 y 0,196), y K = 3 es más estable** cuando se remuestrean los municipios (ARI promedio ≈ 0,82
frente a ≈ 0,51; mueve K a 3 en la barra lateral y mira la tabla de robustez al final de esta sección).
Elegí **K = 4 por interpretación, no por estadística**: separa a los **municipios de regalías**, que tienen una
lógica fiscal distinta, y distingue a los municipios **con caja apretada** de los dependientes **ordenados**.
Con K ≥ 5 los grupos nuevos se parten en pedazos sin una lectura distinta. La conclusión sobre el Valle no
depende de esta elección: con K = 3 también queda sobrerrepresentado en el grupo de caja apretada (74%
frente a 37% del resto del país).

**Hay que decirlo claro:** una silueta de 0,20 indica una estructura **débil**. Los municipios no forman islas
separadas, sino un **continuo** con zonas más densas. Los grupos sirven como **perfiles típicos**, no como
categorías tajantes, y sus fronteras se mueven si cambian los municipios de la muestra.
        """
    )

    st.subheader("Los grupos en el plano de componentes")
    pct12 = var["Varianza explicada"].iloc[:2].sum() * 100
    fig = px.scatter(
        res, x="CP1", y="CP2", color="Nombre grupo", hover_name="municipio",
        hover_data={"departamento": True, "puntaje_idf": ":.1f", "Silueta": ":.2f", "Nombre grupo": False,
                    "CP1": ":.2f", "CP2": ":.2f"},
        color_discrete_map=color_grupo, category_orders={"Nombre grupo": list(color_grupo)}, opacity=0.75,
        title=f"Grupos de K-Means vistos en CP1 vs CP2 ({fmt_num(pct12)}% de la información)",
        labels={"CP1": "CP1 · dependencia (+) vs autonomía fiscal (−)", "CP2": "CP2 · regalías/holgura (+) vs recaudo urbano (−)",
                "Nombre grupo": "Grupo", "puntaje_idf": "Puntaje IDF"},
    )
    fig.update_layout(height=600, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    if n_comp > 2:
        fig3 = px.scatter_3d(res, x="CP1", y="CP2", z="CP3", color="Nombre grupo", hover_name="municipio",
                             color_discrete_map=color_grupo, opacity=0.7,
                             title="Vista 3D (CP1, CP2, CP3): gíralo con el mouse")
        fig3.update_traces(marker=dict(size=3))
        fig3.update_layout(height=600, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig3, width="stretch")

    st.subheader("Perfil de cada grupo con las variables originales")
    perfil = perfil_grupos(df, etiquetas, list(variables))
    nac = df[list(variables)].median()
    z_perfil = (perfil[list(variables)] - nac) / df[list(variables)].std()
    z_perfil.index = [nombre_grupo(g, base) for g in z_perfil.index]
    z_perfil.columns = [ETIQUETA[c] for c in z_perfil.columns]
    fig = px.imshow(z_perfil, text_auto=".1f", color_continuous_scale="RdBu_r", zmin=-1.5, zmax=1.5, aspect="auto",
                    title="¿En qué se diferencia cada grupo? Mediana del grupo frente a la mediana nacional (en desviaciones estándar)")
    fig.update_layout(height=380)
    st.plotly_chart(fig, width="stretch")

    tabla = perfil.copy()
    tabla.index = [nombre_grupo(g, base) for g in tabla.index]
    tabla.loc["Mediana nacional"] = [len(df)] + list(nac.values) + [df["puntaje_idf"].median()]
    tabla.columns = ["Municipios"] + [ETIQUETA[c] for c in variables] + ["Puntaje IDF (no entra al modelo)"]
    st.dataframe(tabla.round(1), width="stretch")
    st.caption("Medianas en unidades originales (porcentajes o pesos por habitante). Se usa la mediana porque resiste los extremos.")

    if base:
        st.markdown(
            """
| Grupo | Rasgos (frente a la mediana nacional) | Ejemplos |
|---|---|---|
| **A · Caja apretada y poca inversión** (≈ 30% del país) | Tienen el **ahorro corriente más bajo** (≈ 38%), la **menor holgura** frente a la Ley 617, la **menor inversión en capital fijo** (≈ 8% del gasto) y el indicador de endeudamiento más alto. Su dependencia de transferencias (≈ 80%) es casi la mediana nacional (79%): no es lo que los define. Puntaje IDF ≈ 51,5 | Buenaventura, Buga, Jamundí, Sevilla |
| **B · Dependientes pero ordenados** | Son los más dependientes (≈ 85%) y recaudan poco, pero tienen **buen ahorro y la mayor holgura**, y recaudan más de lo presupuestado. Puntaje ≈ 55,6 | Municipios pequeños y rurales, sobre todo de Centro Oriente y el Pacífico |
| **C · Municipios de regalías e inversión** | Regalías ≈ 450.000 pesos por habitante (mediana) y **≈ 32% del gasto en capital fijo**. Tienen la ejecución de inversión más baja: les llega plata que les cuesta ejecutar. Puntaje ≈ 58,1 | Municipios petroleros y mineros de los Llanos, Caribe y Centro Oriente |
| **D · Autónomos con recaudo propio** | Dependen solo ≈ 52% de transferencias, con **predial e ICA 3 a 4 veces la mediana**, el mejor ahorro y el mejor balance. Puntaje ≈ 61,1 | Bogotá, Cali, Palmira, Yumbo, Candelaria |

Las letras se asignan por tamaño del grupo (A es el más grande); **no son un ranking**. Que el puntaje IDF
cambie entre grupos **no valida** los perfiles: el puntaje se calcula con 8 de estas 11 variables, así que es
lógico que se mueva con ellas. La prueba útil es la contraria, y está justo abajo.
            """
        )

    st.subheader("¿El puntaje IDF basta para saber qué tipo de municipio es?")
    # Cuartiles del puntaje: Q1 = el 25% de municipios con puntaje más bajo; Q4 = el 25% más alto
    cuartil = pd.qcut(res["puntaje_idf"], 4, labels=False)
    rangos = res.groupby(cuartil)["puntaje_idf"].agg(["min", "max"])
    nombres_q = {q: f"Q{q + 1} · {fmt_num(r['min'])} a {fmt_num(r['max'])} puntos" for q, r in rangos.iterrows()}
    composicion = pd.crosstab(cuartil.map(nombres_q), res["Nombre grupo"], normalize="index") * 100
    composicion = composicion[[g for g in color_grupo if g in composicion.columns]]
    fig = px.imshow(
        composicion, text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto",
        labels={"x": "Grupo", "y": "Cuartil del puntaje IDF", "color": "% del cuartil"},
        title="De cada cuartil del puntaje IDF, ¿qué % de municipios cae en cada grupo?",
    )
    fig.update_layout(height=330)
    st.plotly_chart(fig, width="stretch")
    # eta²: parte de la variación del puntaje IDF que se explica por el grupo (como un R² con el grupo)
    media = res["puntaje_idf"].mean()
    por_grupo = res.groupby("Grupo")["puntaje_idf"]
    eta2 = (por_grupo.size() * (por_grupo.mean() - media) ** 2).sum() / ((res["puntaje_idf"] - media) ** 2).sum()
    q2 = composicion.iloc[1]
    st.markdown(
        f"""
**Lectura.** Si el puntaje bastara para describir a un municipio, cada cuartil tendría un solo tipo. No pasa:
en el segundo cuartil ({q2.name.split(" · ")[1]}) hay
{", ".join(f"{fmt_num(v, 0)}% en {g}" for g, v in q2.items())}. Los grupos explican solo el
**{fmt_num(eta2 * 100, 0)}%** de la variación del puntaje IDF (*eta²*: la parte de la variación que se explica
por pertenecer a un grupo). Es la evidencia central del trabajo: **un mismo puntaje esconde situaciones
distintas**, y dos municipios con el mismo número pueden necesitar cosas muy diferentes.
        """
    )

    st.subheader("Silueta por municipio")
    s = res.sort_values(["Grupo", "Silueta"], ascending=[True, False]).reset_index(drop=True)
    fig = px.bar(s, x=s.index, y="Silueta", color="Nombre grupo", color_discrete_map=color_grupo,
                 hover_name="municipio", title="Coeficiente de silueta de cada municipio, ordenado dentro de su grupo",
                 labels={"x": "Municipios (ordenados por grupo)", "Silueta": "Silueta", "Nombre grupo": "Grupo"})
    fig.update_traces(marker_line_width=0)
    fig.add_hline(y=s["Silueta"].mean(), line_dash="dash", annotation_text=f"Promedio {s['Silueta'].mean():.2f}")
    fig.update_layout(bargap=0, height=420, legend=dict(orientation="h", y=-0.25), xaxis_showticklabels=False)
    st.plotly_chart(fig, width="stretch")
    neg = (res["Silueta"] < 0).mean() * 100
    st.markdown(
        f"El **{fmt_num(neg)}%** de los municipios tiene silueta negativa: están más cerca de otro grupo que del suyo. "
        "Son casos frontera. Abajo están los más ambiguos:"
    )
    amb = res.nsmallest(10, "Silueta")[["municipio", "departamento", "Nombre grupo", "Silueta", "puntaje_idf"]]
    amb.columns = ["Municipio", "Departamento", "Grupo asignado", "Silueta", "Puntaje IDF"]
    st.dataframe(amb.round(2), hide_index=True, width="stretch")

    st.subheader("¿Qué tan estables son los grupos? Pruebas de robustez")
    st.markdown(
        "Cada fila cambia **una** decisión del análisis y compara los grupos nuevos con los de arriba usando el "
        "**ARI** (*Adjusted Rand Index*): 1 = mismos grupos; 0 = se parecen lo que se parecerían por azar. "
        "Todo se calcula en vivo con la configuración de la barra lateral."
    )
    with st.spinner("Calculando las pruebas de robustez (unos segundos la primera vez)…"):
        rob = pruebas_robustez(variables, usar_log, recorte, n_comp, k)
    st.dataframe(
        rob, hide_index=True, width="stretch",
        column_config={
            "ARI mínimo": st.column_config.NumberColumn(format="%.2f"),
            "ARI promedio": st.column_config.NumberColumn(format="%.2f"),
            "ARI máximo": st.column_config.NumberColumn(format="%.2f"),
            "% del Valle en el grupo A": st.column_config.NumberColumn(format="%.0f%%"),
            "% del resto en el grupo A": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.caption(
        "**Cómo leerla.** *Semillas*: si el ARI es cercano a 1, K-Means llega siempre a la misma solución. "
        "*Remuestreo*: se agrupan 50 muestras de municipios tomadas al azar (con reemplazo) y se compara; un ARI "
        "bajo indica que la frontera entre grupos depende de qué municipios entran, algo esperable con una silueta "
        "de ≈ 0,20. Las dos últimas columnas muestran qué parte del Valle y del resto del país cae en el grupo que "
        "más se parece al A: si el Valle sigue muy por encima del resto, el hallazgo del Valle se sostiene."
    )

    st.subheader("Busca un municipio")
    mpio = st.selectbox("Municipio", res.sort_values("municipio").apply(
        lambda r: f"{r['municipio']} ({r['departamento']})", axis=1), index=None,
        placeholder="Escribe el nombre…")
    if mpio:
        fila = res[res.apply(lambda r: f"{r['municipio']} ({r['departamento']})", axis=1) == mpio].iloc[0]
        st.success(f"**{fila['municipio']}** está en **{fila['Nombre grupo']}** (silueta {fila['Silueta']:.2f}, "
                   f"puntaje IDF {fmt_num(fila['puntaje_idf'])}).")
        comp = pd.DataFrame({
            "Variable": [ETIQUETA[c] for c in variables],
            "Municipio": [fila[c] for c in variables],
            "Mediana de su grupo": [res.loc[res["Grupo"] == fila["Grupo"], c].median() for c in variables],
            "Mediana nacional": [df[c].median() for c in variables],
        })
        st.dataframe(comp.round(1), hide_index=True, width="stretch")

    st.download_button(
        "⬇️ Descargar resultados (CSV con formato colombiano)",
        res[["codigo", "municipio", "departamento", "region", "Grupo", "Nombre grupo", "Silueta", "puntaje_idf"]
            + list(variables)].to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
        file_name="grupos_municipios_2024.csv", mime="text/csv",
    )


# ----------------------------------------------------------------------------
# 6. Valle del Cauca
# ----------------------------------------------------------------------------
elif pagina.startswith("6"):
    st.title("Lupa al Valle del Cauca")
    res["Ámbito"] = np.where(res["cod_depto"] == VALLE, "Valle del Cauca", "Resto del país")

    dist = (res.groupby(["Ámbito", "Nombre grupo"]).size() / res.groupby("Ámbito").size()).mul(100).rename("%").reset_index()
    fig = px.bar(dist, x="%", y="Ámbito", color="Nombre grupo", orientation="h", color_discrete_map=color_grupo,
                 category_orders={"Nombre grupo": list(color_grupo)}, text=dist["%"].round(0),
                 title="¿Cómo se reparten los municipios entre los grupos? Valle vs. resto del país",
                 labels={"%": "% de los municipios", "Ámbito": "", "Nombre grupo": "Grupo"})
    fig.update_layout(height=320, legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig, width="stretch")

    if base:
        # Todas las cifras del texto se calculan aquí, en vivo, con la configuración recomendada.
        es_valle = res["cod_depto"] == VALLE
        en_a = res["Grupo"] == "A"
        valle_a = res[es_valle & en_a]
        pct_valle_a = 100 * valle_a.shape[0] / es_valle.sum()
        pct_resto_a = 100 * (en_a & ~es_valle).sum() / (~es_valle).sum()
        nac = df[list(VARIABLES) + ["puntaje_idf"]].median()

        def sobre_mediana(col):
            return int((valle_a[col] > nac[col]).sum())

        def mediana(filtro, col, dec=0):
            return fmt_num(res.loc[filtro, col].median(), dec)

        # Puesto del Valle entre los departamentos con 10 o más municipios
        por_depto = res.groupby("departamento").agg(n=("Grupo", "size"), pct_a=("Grupo", lambda g: (g == "A").mean()))
        por_depto = por_depto[por_depto["n"] >= 10]
        puesto = int((por_depto["pct_a"] > por_depto.loc["Valle del Cauca", "pct_a"]).sum()) + 1
        autonomos = sorted(res.loc[es_valle & (res["Grupo"] == "D"), "municipio"])
        n_regalias = int((es_valle & (res["Grupo"] == "C")).sum())
        texto_regalias = (
            "Ningún municipio vallecaucano está en el grupo de regalías." if n_regalias == 0
            else f"{n_regalias} municipios vallecaucanos están en el grupo de regalías."
        )

        st.markdown(
            f"""
**El hallazgo.** El **{fmt_num(pct_valle_a, 0)}% de los municipios del Valle ({valle_a.shape[0]} de
{int(es_valle.sum())})** cae en el grupo **"Caja apretada y poca inversión"**, frente al
{fmt_num(pct_resto_a, 1)}% del resto del país. Entre los departamentos con 10 o más municipios, el Valle es el
**número {puesto}** en esa proporción. {texto_regalias}

**No es un problema de recaudo ni de dependencia.** De esos {valle_a.shape[0]} municipios,
**{sobre_mediana("predial_pc")} recaudan predial por encima de la mediana nacional** y {sobre_mediana("ica_pc")}
también ICA; solo {sobre_mediana("dependencia_transferencias")} dependen de las transferencias más que la
mediana del país. El Valle en conjunto depende **menos** de la Nación que el resto (mediana
{mediana(es_valle, "dependencia_transferencias")}% frente a {mediana(~es_valle, "dependencia_transferencias")}%).

**Lo que los distingue es cuánto les queda después de funcionar.** Frente al resto del país, el Valle tiene
menos holgura frente al tope de la Ley 617 ({mediana(es_valle, "holgura")} frente a
{mediana(~es_valle, "holgura")} puntos), menos ahorro corriente ({mediana(es_valle, "ahorro_corriente")}%
frente a {mediana(~es_valle, "ahorro_corriente")}%) y destina a capital fijo apenas
{mediana(es_valle, "inversion_fbkf")}% del gasto, frente a {mediana(~es_valle, "inversion_fbkf")}%.

**El puntaje IDF no lo ve.** El puntaje mediano del Valle ({mediana(es_valle, "puntaje_idf", 1)}) es casi
igual al del resto del país ({mediana(~es_valle, "puntaje_idf", 1)}). Un ranking diría que el Valle es
"promedio"; su estructura dice que su margen es estrecho.

**El corredor industrial es la excepción.** {len(autonomos)} municipios están en el grupo de **autónomos con
recaudo propio**, casi todos sobre el corredor industrial y agroindustrial del centro del departamento:
{", ".join(autonomos)}.

**Lectura.** El Valle tiene un núcleo con base tributaria fuerte y un anillo de ciudades intermedias
(Buenaventura, Buga, Jamundí, Sevilla) que recaudan, pero les queda poco margen e invierten poco. Para la
gobernación o para quien les preste, el riesgo no está en cuánto recaudan sino en **cuánto les queda después
de pagar el funcionamiento**. Esto describe el patrón, no su causa: puede venir de gastos de funcionamiento
rígidos, de la categoría de Ley 617 de cada municipio o de no tener regalías para invertir.

*Robustez:* con los datos de TerriData de 2020 a 2024 (consulta hecha fuera de la app) el patrón se repite
los cinco años: el Valle tiene menos holgura, menos ahorro y cerca de la mitad de inversión en capital fijo
que el país, mientras su puntaje IDF queda unas veces arriba y otras abajo. La proporción del Valle en este
grupo se mantiene cerca del 67% sin recorte de extremos o con 5 componentes; se debilita si se hace K-Means
sin ACP o si se quita la holgura del modelo.
            """
        )

    st.subheader("Los 42 municipios del Valle")
    valle = res[res["cod_depto"] == VALLE]
    resto = res[res["cod_depto"] != VALLE]
    fig = go.Figure()
    fig.add_scatter(x=resto["CP1"], y=resto["CP2"], mode="markers", name="Resto del país",
                    marker=dict(color="#CFD8DC", size=5), text=resto["municipio"], hoverinfo="text")
    for g, color in color_grupo.items():
        sub = valle[valle["Nombre grupo"] == g]
        if len(sub):
            fig.add_scatter(x=sub["CP1"], y=sub["CP2"], mode="markers+text", name=f"Valle · {g}",
                            marker=dict(color=color, size=10, line=dict(color="white", width=1)),
                            text=sub["municipio"], textposition="top center", textfont=dict(size=9),
                            hovertemplate="%{text}<extra>" + g + "</extra>")
    fig.update_layout(title="Municipios del Valle (con color según su grupo) en el plano CP1 vs CP2",
                      xaxis_title="CP1 · dependencia (+) vs autonomía fiscal (−)",
                      yaxis_title="CP2 · regalías/holgura (+) vs recaudo urbano (−)",
                      legend=dict(orientation="h", y=-0.15))
    fig.update_layout(height=650)
    st.plotly_chart(fig, width="stretch")

    t = valle[["municipio", "Nombre grupo", "puntaje_idf", "Silueta", "dependencia_transferencias",
               "ahorro_corriente", "holgura", "endeudamiento_lp", "predial_pc", "ica_pc"]].sort_values(
        ["Nombre grupo", "puntaje_idf"], ascending=[True, False])
    t.columns = ["Municipio", "Grupo", "Puntaje IDF", "Silueta", "Dependencia (%)", "Ahorro corriente (%)",
                 "Holgura", "Endeudamiento LP (%)", "Predial per cápita", "ICA per cápita"]
    st.dataframe(t.round(1), hide_index=True, width="stretch", height=560)


# ----------------------------------------------------------------------------
# 7. Conclusiones
# ----------------------------------------------------------------------------
else:
    st.title("Discusión de resultados y conclusiones")
    st.caption("Las cifras corresponden a la configuración recomendada: 11 variables, logaritmo, recorte 1%/99%, 4 componentes y K = 4.")
    st.markdown(
        """
### 1. ¿Qué variables definen los componentes principales?
- **CP1 (≈ 26%)** es el eje de **dependencia frente a autonomía fiscal**: dependencia de transferencias
  contra recaudo de ICA y predial. Es la diferencia más importante entre municipios colombianos.
- **CP2 (≈ 17%)** separa a los municipios con **regalías, inversión y margen de funcionamiento** de los que
  se sostienen con impuestos urbanos.
- **CP3 (≈ 13%)** contrasta la **inversión financiada con regalías** con la **solvencia del funcionamiento**
  (ahorro y holgura).
- **CP4 (≈ 10%)** aísla el **endeudamiento** con déficit primario.

### 2. ¿Cuánta información se conservó?
Los 4 componentes conservan **≈ 66%** de la varianza de las 11 variables. No es una reducción espectacular,
y eso también es un resultado: **las finanzas municipales no se pueden resumir en un solo número**. Eso pone
en duda leer el puntaje del IDF como si fuera unidimensional. El plano CP1-CP2 que se dibuja solo muestra
≈ 43%, así que hay que leerlo con cuidado.

### 3. ¿Qué diferencias reales aparecen entre los grupos?
Hay cuatro perfiles con lecturas económicas distintas: **caja apretada y poca inversión**, **dependientes
pero ordenados**, **municipios de regalías e inversión** y **autónomos con recaudo propio**. La diferencia
más útil es que **la dependencia de transferencias no implica desorden**: el grupo B es el más dependiente
(≈ 85%), pero tiene mucho mejor ahorro y holgura que el A, cuya dependencia está en la mediana nacional. Al revés, el grupo C tiene recursos (regalías) pero es
el que menos ejecuta su inversión. El puntaje IDF va de ≈ 51,5 (A) a ≈ 61,1 (D), pero eso no valida los
grupos, porque el puntaje se calcula con 8 de las mismas variables. Lo importante es lo contrario: **dentro de
cada cuartil del IDF conviven los cuatro perfiles**, y los grupos explican solo cerca del 22% de la variación
del puntaje. Es decir, **municipios con puntajes parecidos pueden tener problemas muy distintos**.

### 4. ¿Son grupos claramente distinguibles?
**No del todo.** La silueta promedio es ≈ 0,20 y cerca de 3% de los municipios tiene silueta negativa
(están en la frontera entre dos grupos). Los municipios forman un **continuo** con zonas de mayor densidad,
no islas separadas. Los grupos sirven como **arquetipos** para orientar el análisis, no como etiquetas
definitivas.

**Qué es estable y qué no.** El algoritmo es consistente: con otras semillas aleatorias el ARI va de 0,97 a
1,00, y sin recorte de extremos o con 3 o 5 componentes queda entre 0,90 y 0,93. Pero los grupos exactos
**no** son estables a la muestra: si se remuestrean los municipios, el ARI promedio cae a ≈ 0,51, porque las
fronteras de un continuo se mueven. K = 3 sería más estable (≈ 0,82); se eligió K = 4 por su lectura
económica. Lo que sí resiste todas las pruebas es el hallazgo central: el Valle queda en el grupo de caja
apretada con el doble o más de frecuencia que el resto del país (tabla al final de la sección 5).

### 5. Relación con la pregunta: el Valle del Cauca
El Valle **no se parece al promedio nacional, aunque su puntaje IDF sí**: el 69% de sus municipios está en
*caja apretada y poca inversión* (frente al 28,5% del resto del país), y su núcleo industrial (Cali, Yumbo,
Palmira, Candelaria) es *autónomo*. Sus municipios recaudan más y dependen menos de la Nación que el
promedio; su reto no es recaudar más sino **ganar margen después del funcionamiento para poder invertir**.
El patrón (menos holgura, menos ahorro y menos inversión que el país) se repite de 2020 a 2024.

### 6. Limitaciones
1. **Un solo año (2024)**, que además es de empalme entre administraciones. Un municipio puede cambiar de
   grupo de un año a otro; lo ideal sería repetir el análisis con 2020 a 2024 y ver quién se mueve.
2. **Sensibilidad a la transformación:** sin logaritmo en las variables en pesos, los grupos cambian
   bastante (ARI ≈ 0,45), porque quedan dominados por los valores extremos. La decisión de usar logaritmo
   está justificada, pero condiciona el resultado.
3. **Datos autorreportados** por los municipios (CUIPO y FUT) y población proyectada para los per cápita.
4. **K-Means supone grupos más o menos esféricos y de tamaño parecido**. Con una estructura débil
   (silueta ≈ 0,2) la frontera entre grupos es en parte arbitraria. Métodos como el clustering jerárquico o
   los modelos de mezcla gaussiana podrían compararse en un trabajo posterior.
5. Los grupos **describen**, no explican: no dicen *por qué* un municipio tiene la caja apretada (tamaño de
   la nómina, pasivos pensionales, historia de endeudamiento, etc.).
        """
    )
