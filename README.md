# Perfiles fiscales de los municipios colombianos (2024)
### ACP + K-Means · ¿dónde queda el Valle del Cauca?

Taller individual de **Machine Learning aplicado a Economía y Finanzas**, Pontificia Universidad Javeriana Cali.

- **App desplegada:** https://acp-kmeans-municipios-kuxjasvhetsygpyd7sdzxc.streamlit.app/
- **Video de sustentación:** _(pegar aquí el enlace)_

---

## 1. Contexto y pregunta

Los municipios de Colombia tienen estructuras financieras muy distintas: unos viven casi solo de las
transferencias de la Nación (SGP), otros recaudan bastante predial e ICA, otros reciben regalías y otros
están cerca del tope legal de gastos de funcionamiento (Ley 617 de 2000). El DNP resume todo esto en un
solo número, el **Índice de Desempeño Fiscal (IDF)**, y con él arma un ranking. El problema es que un mismo
puntaje puede esconder situaciones muy distintas.

**Pregunta:** ¿qué *tipos* de municipio existen según su estructura financiera, en cuántas dimensiones se
pueden resumir sus diferencias y en qué tipo caen los municipios del **Valle del Cauca**?

**¿Por qué ACP y K-Means?**
- Las 11 variables financieras están correlacionadas en bloques (autonomía fiscal, solvencia del
  funcionamiento, regalías e inversión). El **ACP** resume esa redundancia en pocas dimensiones que se
  pueden interpretar.
- **K-Means** busca perfiles típicos de municipio. Esto sirve para la política pública: un municipio
  dependiente pero ordenado necesita cosas distintas de uno con la caja apretada, aunque los dos tengan
  el mismo puntaje en el IDF.

## 2. Datos

| | |
|---|---|
| **Fuente** | [TerriData – DNP](https://terridata.dnp.gov.co/), descarga masiva de la dimensión 7 *Finanzas públicas* |
| **Observación** | Un municipio, vigencia 2024 |
| **Cobertura** | 1.102 municipios con información completa (se excluyen los totales nacional y departamentales) |
| **Variables del modelo (11)** | 8 indicadores del Nuevo IDF + 3 de recaudo y regalías per cápita (ver tabla) |
| **Variables de contraste (no entran al modelo)** | Puntaje del IDF y sus dos componentes, departamento y región |

| Variable | Qué mide |
|---|---|
| Dependencia de transferencias | % de los ingresos que llega de la Nación |
| Inversión en capital fijo | % del gasto que se va a formación bruta de capital fijo |
| Endeudamiento de largo plazo | Deuda de largo plazo frente a los ingresos corrientes |
| Ahorro corriente | Lo que sobra de los ingresos corrientes después del funcionamiento |
| Balance fiscal primario | Ingresos menos gastos, sin contar intereses |
| Ejecución de ingresos | Recaudo frente a lo presupuestado |
| Ejecución de inversión | Inversión ejecutada frente a la presupuestada |
| Holgura | Margen frente al tope de gastos de funcionamiento (Ley 617) |
| Predial per cápita | Recaudo de predial por habitante |
| ICA per cápita | Recaudo de industria y comercio por habitante |
| Regalías per cápita | Regalías giradas por habitante |

**Limitaciones de la fuente:** datos que reportan los mismos municipios (CUIPO y FUT); un solo año, que
además es de empalme entre administraciones; los per cápita dependen de las proyecciones de población del
DANE; las regalías son volátiles.

## 3. Metodología

1. **Preparación** (`src/preparar_datos.py`): descarga del archivo plano, filtro a 2024 y a códigos
   municipales, paso de formato largo a ancho. No quedan faltantes ni códigos duplicados.
2. **Transformación** (`src/analisis.py`):
   - `log(1 + x)` en las tres variables en pesos per cápita, que son muy asimétricas.
   - Winsorización al percentil 1% y 99%, para limitar extremos como una ejecución de ingresos de 822%.
   - Estandarización (media 0 y desviación 1).
3. **ACP:** varianza explicada, gráfico de sedimentación, cargas y biplot. Se conservan **4 componentes**
   (criterio de Kaiser, ≈ 66% de la varianza).
4. **K-Means** sobre los 4 componentes: se compara K = 2…10 con el método del codo y la silueta, y se
   elige **K = 4**. Luego se perfilan los grupos con las medianas de las variables originales.
5. **Robustez:** estabilidad frente a la semilla aleatoria, al recorte, al número de componentes, al
   logaritmo y al remuestreo de municipios, medida con el ARI (*Adjusted Rand Index*). La función
   `robustez()` de `src/analisis.py` la calcula en vivo y la app la muestra al final de la sección 5.

Toda la app calcula en vivo. Desde la barra lateral se pueden cambiar variables, transformación, recorte,
número de componentes y K.

## 4. Resultados principales

- **Componentes:** CP1 = dependencia frente a autonomía fiscal (26%); CP2 = regalías y holgura frente a
  recaudo urbano (17%); CP3 = inversión con regalías frente a solvencia del funcionamiento (13%);
  CP4 = endeudamiento (10%). Las finanzas municipales **no son unidimensionales**.
- **Cuatro perfiles:**

| Grupo | Municipios | Rasgo principal | IDF mediano* |
|---|---|---|---|
| A · Caja apretada y poca inversión | 331 | Bajo ahorro corriente, poca holgura y poca inversión en capital fijo; dependencia en la mediana nacional | 51,5 |
| B · Dependientes pero ordenados | 286 | Muy dependientes, pero con buen ahorro y holgura; recaudan mucho más de lo presupuestado (el DNP lo penaliza) | 55,6 |
| C · Municipios de regalías e inversión | 255 | Regalías altas y mucha inversión en capital fijo | 58,1 |
| D · Autónomos con recaudo propio | 230 | Predial e ICA altos, baja dependencia | 61,1 |

\*El puntaje del IDF no entró al modelo, pero no sirve para validar los grupos: se calcula con 8 de las
mismas variables. Las letras no son un ranking.

- **Un mismo puntaje esconde perfiles distintos:** en cada cuartil del puntaje IDF conviven los cuatro grupos,
  y los grupos explican solo ≈ 22% de la variación del puntaje (*eta²*). La app lo muestra en vivo en la
  sección 5.

- **Separación:** la silueta es ≈ 0,20, una estructura débil. Los municipios forman un continuo y los
  grupos funcionan como arquetipos. El algoritmo es consistente (ARI de 0,97 a 1,00 con otras semillas),
  pero los grupos exactos cambian si se remuestrean los municipios (ARI ≈ 0,51). K = 3 es más estable
  (≈ 0,82) y tiene casi la misma silueta; K = 4 se eligió por interpretación, porque separa a los
  municipios de regalías. El hallazgo del Valle se mantiene en todas las pruebas.
- **Valle del Cauca:** el **69% de sus municipios (29 de 42)** está en *caja apretada y poca inversión*,
  frente al 28,5% del resto del país (el primero entre los departamentos con 10 o más municipios). No es un
  problema de recaudo ni de dependencia: los 29 recaudan predial por encima de la mediana nacional y el Valle
  depende menos de la Nación que el resto (71% frente a 79%). Lo que lo distingue es el margen después de
  funcionar: holgura de 21 frente a 28, ahorro corriente de 41% frente a 49% y 6% del gasto en capital fijo
  frente a 14%. Su puntaje IDF (54,9) es casi igual al del resto (56,0): **el ranking no ve el patrón**.
  El corredor industrial (Cali, Yumbo, Palmira, Candelaria, El Cerrito…) es *autónomo*.
- **Robustez en el tiempo:** con TerriData 2020–2024 (consulta hecha fuera de la app) el Valle tiene menos
  holgura, menos ahorro y cerca de la mitad de inversión en capital fijo que el país en los cinco años.
- **Limitación clave:** sin el logaritmo, los grupos cambian bastante (ARI ≈ 0,45).

## 5. Cómo ejecutar

```bash
git clone <url-del-repositorio>
cd ACP-KMeans-Municipios
pip install -r requirements.txt
streamlit run app.py
```

La base procesada `data/municipios_2024.csv` ya está en el repositorio. Para regenerarla desde TerriData
(descarga ~30 MB y descomprime ~370 MB en `data/raw/`):

```bash
python src/preparar_datos.py
```

## 6. Estructura

```
├── app.py                    # Aplicación Streamlit (7 secciones)
├── src/
│   ├── preparar_datos.py     # Descarga y construcción de la base
│   └── analisis.py           # Preparación, ACP, K-Means y perfiles
├── data/
│   └── municipios_2024.csv   # Base procesada (1.102 municipios)
├── .streamlit/config.toml    # Tema visual
├── requirements.txt
└── README.md
```
