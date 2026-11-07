import json
import math

def calcular_distancia_centroides(file_path):
    with open(file_path, 'r') as f:
        cells = json.load(f)
    
    print(f"{'ID A':<10} | {'ID B':<10} | {'Distância (km)':<15}")
    print("-" * 40)
    
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            c1 = cells[i]['centroid']
            c2 = cells[j]['centroid']
            
            # Cálculo de distância simples (Haversine aproximado)
            dlat = c1[0] - c2[0]
            dlon = c1[1] - c2[1]
            dist = math.sqrt(dlat**2 + dlon**2) * 111.32
            
            if dist < 200:  # Filtrar apenas os próximos para análise
                print(f"{cells[i]['id']:<10} | {cells[j]['id']:<10} | {dist:>12.2f} km")

def analyze_cluster_proximity(cells: list):
    """
    Lista a distância entre todos os pares de clusters para identificar sobreposições.
    """
    print(f"\n{'ID A':<10} | {'ID B':<10} | {'Distância (km)':<15} | {'Status'}")
    print("-" * 60)
    
    for i, c1 in enumerate(cells):
        for j, c2 in enumerate(cells[i+1:], i+1):
            # Haversine simples entre os centroides (lat, lng)
            dlat = math.radians(c2.centroid[0] - c1.centroid[0])
            dlon = math.radians(c2.centroid[1] - c1.centroid[1])
            a = (math.sin(dlat/2)**2 + math.cos(math.radians(c1.centroid[0])) * math.cos(math.radians(c2.centroid[0])) * math.sin(dlon/2)**2)
            dist = 2 * 6371.0 * math.asin(math.sqrt(a))
            
            status = "SOBREPOSTO?" if dist < 20 else "OK" # 20km é um limiar comum
            if dist < 100: # Foca apenas nos que estão dentro do seu EPS_KM
                print(f"{c1.id:<10} | {c2.id:<10} | {dist:>12.2f} km | {status}")


# Uso:
calcular_distancia_centroides('output.json')
