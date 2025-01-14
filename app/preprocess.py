import numpy as np



hounsfield_min = -1000
hounsfield_max = 2000
hounsfield_range = hounsfield_max - hounsfield_min

def normalize_vol(vol):
    vol[vol < hounsfield_min] = hounsfield_min
    vol[vol > hounsfield_max] = hounsfield_max
    return (vol - hounsfield_min) / hounsfield_range
