import math

from lm_motion import (
    haversine_km,
)

# ------------------------------------------------------------
# Flash rate
# ------------------------------------------------------------

def compute_flash_rate(cluster, dt_minutes):

    if dt_minutes <= 0:
        return 0.0

    return len(cluster) / dt_minutes


# ------------------------------------------------------------
# 
# ------------------------------------------------------------
def compute_density(cluster, centroid):

    # raio máximo do cluster
    max_radius_km = 0.0

    for p in cluster:

        d = haversine_km(p.lat, p.lng,
                         centroid[0], centroid[1])

        if d > max_radius_km:
            max_radius_km = d

    # evita divisão por zero
    max_radius_km = max(max_radius_km, 1.0)

    # área aproximada
    area_km2 = math.pi * (max_radius_km ** 2)

    # densidade
    density = len(cluster) / area_km2

    return density


# ------------------------------------------------------------
# 
# ------------------------------------------------------------
def compute_severity(flash_rate, density):

    # ok
    severity = (flash_rate * 0.5 + density * 0.5)
    
    return severity


# ---------------------------------------------------------------------------
# 
# ---------------------------------------------------------------------------
def detect_jump(prev_rate: float, curr_rate: float,
                threshold: float = 2.0) -> bool:
    """Retorna True se a taxa atual cresceu >= threshold vezes."""
    if prev_rate == 0:
        return False
    return (curr_rate / prev_rate) >= threshold

    