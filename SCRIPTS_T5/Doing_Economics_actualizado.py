# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Limpiar todas las figuras previas acumuladas en Spyder
plt.close('all')

# ==========================================
# 1. Cargar datos de la NASA
# ==========================================
url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv"
df = pd.read_csv(url, skiprows=1, na_values="***")

# Limpieza de filas y conversión de tipos
df = df[pd.to_numeric(df["Year"], errors="coerce").notnull()].copy()
df["Year"] = df["Year"].astype(int)

cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", 
        "DJF", "MAM", "JJA", "SON", "J-D"]
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# PARTE 1.1: Anomalías de Temperatura
# ==========================================

# (i) Gráfico 1: Un mes específico (Enero)
plt.figure(figsize=(10, 4.5))
plt.plot(df["Year"], df["Jan"], color="#1f77b4", linewidth=1.2, label="Anomalía Enero")
plt.axhline(0, color="red", linestyle="--", linewidth=1.2)
plt.title("Anomalías de Temperatura en el Hemisferio Norte (Enero, 1880-Presente)", fontweight='bold')
plt.xlabel("Año")
plt.ylabel("Anomalía de Temperatura (°C)")
plt.text(1885, 0.08, "promedio de 1951 a 1980", color="red", fontsize=9, fontweight='bold')
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# (ii) Gráfico 2: Estaciones (DJF, MAM, JJA, SON)

fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, sharey=True)

estaciones = {
    "DJF": "Invierno",
    "MAM": "Primavera",
    "JJA": "Verano",
    "SON": "Otoño"
}

for ax, (codigo, nombre) in zip(axes, estaciones.items()):
    ax.plot(
        df["Year"],
        df[codigo],
        linewidth=1.2
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_ylabel("°C")
    ax.set_title(nombre, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)

# Misma escala para las cuatro estaciones
axes[0].set_ylim(-1.1, 2.0)

axes[-1].set_xlabel("Año")

fig.suptitle(
    "Anomalías de Temperatura por Estación (1880-Presente)",
    fontweight="bold",
    fontsize=14
)

plt.tight_layout()
plt.show()

# (iii) Gráfico 3: Promedio Anual (J-D)
plt.figure(figsize=(10, 4.5))
plt.plot(df["Year"], df["J-D"], color="#d62728", linewidth=1.5, label="Promedio Anual (J-D)")
plt.axhline(0, color="blue", linestyle="--", linewidth=1.2)
plt.title("Anomalías de Temperatura Promedio Anual (1880-Presente)", fontweight='bold')
plt.xlabel("Año")
plt.ylabel("Anomalía de Temperatura (°C)")
plt.text(1885, 0.08, "promedio de 1951 a 1980", color="blue", fontsize=9, fontweight='bold')
plt.legend(loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# PARTE 1.2: Distribución y Eventos Extremos
# ==========================================

p1 = df[(df["Year"] >= 1951) & (df["Year"] <= 1980)]
p2 = df[(df["Year"] >= 1981) & (df["Year"] <= 2010)]

meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
obs_p1 = p1[meses].values.flatten()
obs_p2 = p2[meses].values.flatten()

obs_p1 = obs_p1[~np.isnan(obs_p1)]
obs_p2 = obs_p2[~np.isnan(obs_p2)]

d3_p1 = np.quantile(obs_p1, 0.30)
d7_p1 = np.quantile(obs_p1, 0.70)

print("\n--- RESULTADOS PARTE 1.2 ---")
print(f"Decil 3 (Umbral 'Frío' 1951-1980): {d3_p1:.3f} °C")
print(f"Decil 7 (Umbral 'Caliente' 1951-1980): {d7_p1:.3f} °C")

meses_calientes_p2 = np.sum(obs_p2 > d7_p1)
pct_calientes_p2 = (meses_calientes_p2 / len(obs_p2)) * 100
print(f"Porcentaje de meses 'calientes' en 1981-2010: {pct_calientes_p2:.2f}%")

periodos = {
    "1921-1950": df[(df["Year"] >= 1921) & (df["Year"] <= 1950)],
    "1951-1980": p1,
    "1981-2010": p2
}

estaciones = ["DJF", "MAM", "JJA", "SON"]
res_stats = []

for nombre_p, datos_p in periodos.items():
    for est in estaciones:
        res_stats.append({
            "Periodo": nombre_p,
            "Estacion": est,
            "Media": datos_p[est].mean(),
            "Varianza": datos_p[est].var()
        })

tabla_stats = pd.DataFrame(res_stats)
print("\n--- Media y Varianza por Estación ---")
print(tabla_stats)

# Gráfico 4: Histogramas Comparativos
plt.figure(figsize=(10, 5))
plt.hist(obs_p1, bins=20, alpha=0.5, label="1951–1980 (Línea base)", color="blue", density=True)
plt.hist(obs_p2, bins=20, alpha=0.5, label="1981–2010", color="red", density=True)
plt.axvline(d7_p1, color="black", linestyle="--", linewidth=1.5, label=f"Umbral Caliente ({d7_p1:.2f}°C)")
plt.title("Distribución de Anomalías de Temperatura Mensuales (Comparación de Periodos)", fontweight='bold')
plt.xlabel("Anomalía de Temperatura (°C)")
plt.ylabel("Densidad de Frecuencia")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ==========================================
# PARTE 1.3: CO2 y Relación con la Temperatura
# ==========================================

co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
co2_df = pd.read_csv(
    co2_url, 
    comment='#', 
    sep=r'\s+',
    header=None, 
    names=["year", "month", "decimal_date", "interpolated", "trend", "ndays", "sd", "unc"]
)

# Gráfico 5: CO2 vs Tiempo
plt.figure(figsize=(10, 4.5))
plt.plot(co2_df["decimal_date"], co2_df["interpolated"], color="lightgrey", label="Interpolado (Estacional)")
plt.plot(co2_df["decimal_date"], co2_df["trend"], color="darkgreen", linewidth=2, label="Tendencia Filtrada")
plt.title("Concentración Atmosférica de CO₂ en Mauna Loa (1960-Presente)", fontweight='bold')
plt.xlabel("Año")
plt.ylabel("CO₂ (Partes por Millón - ppm)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Gráfico 6: Dispersión CO2 vs Temp (Mayo)
co2_mayo = co2_df[co2_df["month"] == 5][["year", "trend"]].rename(columns={"year": "Year", "trend": "CO2_trend"})
merged_data = pd.merge(df[["Year", "May"]], co2_mayo, on="Year").dropna()

r_coef, p_val = pearsonr(merged_data["CO2_trend"], merged_data["May"])

print("\n--- RESULTADOS PARTE 1.3 ---")
print(f"Coeficiente de Correlación de Pearson (CO2 vs Temp Mayo): {r_coef:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(merged_data["May"], merged_data["CO2_trend"], color="teal", alpha=0.7, edgecolors="k")
plt.title(f"Relación entre Anomalía de Temp. (Mayo) y CO₂ (r = {r_coef:.2f})", fontweight='bold')
plt.xlabel("Anomalía de Temperatura en Mayo (°C)")
plt.ylabel("Concentración de CO₂ (ppm)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================
# BLOQUE DE CÓDIGO: GENERACIÓN DE TABLAS DE FRECUENCIAS (1.2.1)
# =============================================================

# 1. Cargar los datos directamente desde la URL oficial de la NASA
url_datos = "https://data.giss.nasa.gov/gistemp/tabledata_v4/NH.Ts+dSST.csv"
df = pd.read_csv(url_datos, skiprows=1)

# Lista de meses y conversión a numérico
meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
for col in meses:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 2. Filtrar y extraer datos por periodo
p1_data = df[(df['Year'] >= 1951) & (df['Year'] <= 1980)][meses].to_numpy().flatten()
p2_data = df[(df['Year'] >= 1981) & (df['Year'] <= 2010)][meses].to_numpy().flatten()

p1_clean = p1_data[~np.isnan(p1_data)]
p2_clean = p2_data[~np.isnan(p2_data)]

# 3. Agrupar en rangos de 0.05 °C
bins = np.arange(-0.50, 1.55, 0.05)
freq_p1 = pd.cut(p1_clean, bins=bins).value_counts().sort_index()
freq_p2 = pd.cut(p2_clean, bins=bins).value_counts().sort_index()

etiquetas = [f"{b:.2f}" for b in bins[1:]]

# DataFrames para cada periodo
df_t1 = pd.DataFrame({'Rango de anomalía de temp. (T) [1951–1980]': etiquetas, 'Frecuencia': freq_p1.values})
df_t1 = df_t1[df_t1['Frecuencia'] > 0]

df_t2 = pd.DataFrame({'Rango de anomalía de temp. (T) [1981–2010]': etiquetas, 'Frecuencia': freq_p2.values})
df_t2 = df_t2[df_t2['Frecuencia'] > 0]

# 4. Función para renderizar e imprimir la tabla como figura visual
def crear_tabla_grafica(df_tabla, titulo, color_header, alto_figura=8):
    fig, ax = plt.subplots(figsize=(6.5, alto_figura))
    ax.axis('off')
    
    tabla = ax.table(
        cellText=df_tabla.values,
        colLabels=df_tabla.columns,
        cellLoc='center',
        loc='center'
    )
    
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(9)
    tabla.scale(1.1, 1.2)
    
    for (row, col), cell in tabla.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor('#d3d3d3')
        if row == 0:
            cell.set_facecolor(color_header)
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#f9f9f9' if row % 2 == 0 else '#ffffff')
                
    plt.title(titulo, fontsize=12, fontweight='bold', pad=25)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# Generar Tabla 1.2.1a (1951–1980)
# -------------------------------------------------------------
crear_tabla_grafica(
    df_t1, 
    'Tabla 1.2.1a: Frecuencia de anomalías de temperatura (1951–1980)', 
    '#2b5c8f',
    alto_figura=8
)

# -------------------------------------------------------------
# Generar Tabla 1.2.1b (1981–2010)
# -------------------------------------------------------------
crear_tabla_grafica(
    df_t2, 
    'Tabla 1.2.1b: Frecuencia de anomalías de temperatura (1981–2010)', 
    '#d95f02',
    alto_figura=12
)