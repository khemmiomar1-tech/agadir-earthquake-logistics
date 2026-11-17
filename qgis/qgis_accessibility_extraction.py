from qgis.core import QgsProject, QgsVectorLayer, QgsField, QgsFeature
from PyQt5.QtCore import QVariant
import pandas as pd
import numpy as np

# =====================
# 1. CHEMINS DES CSV
# =====================
csv_adp = r"C:\Users\lenovo\Downloads\adp_cvrp_demande_positive_clean.csv"
csv_matrix = r"C:\Users\lenovo\Downloads\distance_matrix_cvrp_T6_final.csv"

# =====================
# 2. LIRE LES CSV
# =====================
df_adp = pd.read_csv(csv_adp)
df_matrix = pd.read_csv(csv_matrix)

# Assurer que les colonnes sont correctes
# df_adp: node_id, demand_final, x, y
# df_matrix: InputID (dépot), TargetID (ADP), time_min (temps/access)

DEPOT = "DEPOT_18"  # ou le nom de ton dépôt si différent

# =====================
# 3. EXTRACTION ACCESSIBILITE
# =====================
access_dict = {}
subset_matrix = df_matrix[df_matrix['InputID'] == DEPOT]

for _, row in subset_matrix.iterrows():
    target = row['TargetID']
    time = row['time_min']
    access_dict[target] = time

# Ajouter access à df_adp
df_adp['access'] = df_adp['node_id'].map(access_dict)

# Remplacer NaN par 0 si nécessaire
df_adp['access'] = df_adp['access'].fillna(0)

# =====================
# 4. NORMALISATION
# =====================
def normalize(series):
    arr = series.to_numpy(dtype=float)
    if arr.max() == arr.min():
        return np.zeros(len(arr))
    return (arr - arr.min()) / (arr.max() - arr.min())

df_adp['demand_n'] = normalize(df_adp['demand_final'])
df_adp['access_n'] = normalize(df_adp['access'])

# =====================
# 5. CALCUL SOVI
# =====================
df_adp['SoVI'] = 0.6 * df_adp['demand_n'] + 0.4 * df_adp['access_n']

# =====================
# 6. CREER UNE COUCHE MEMOIRE
# =====================
crs = "EPSG:26192"  # adapter au projet QGIS

layer = QgsVectorLayer(f"Point?crs={crs}", "SoVI_T6", "memory")
provider = layer.dataProvider()

# Champs
provider.addAttributes([
    QgsField("ADP_id", QVariant.String),
    QgsField("demand", QVariant.Double),
    QgsField("access", QVariant.Double),
    QgsField("SoVI", QVariant.Double)
])
layer.updateFields()

# Ajouter les features
for idx, row in df_adp.iterrows():
    feat = QgsFeature(layer.fields())
    feat.setGeometry(QgsPointXY(row['x'], row['y']))
    feat['ADP_id'] = row['node_id']
    feat['demand'] = float(row['demand_final'])
    feat['access'] = float(row['access'])
    feat['SoVI'] = float(row['SoVI'])
    provider.addFeature(feat)

layer.updateExtents()
QgsProject.instance().addMapLayer(layer)

print("✔ Couche SoVI créée avec succès !")