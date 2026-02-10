import numpy as np 

def interpolation(x, method="linear"):
    if len(x) >= 3:
        return x.bfill().ffill().interpolate(method=method, limit_direction="both")
    else:
        return x
    
def fourier_features(x, periods):
    t = np.asarray(x).flatten()
    feats = []
    for p in periods:
        w = 2 * np.pi / p
        feats.append(np.sin(w * t))
        feats.append(np.cos(w * t))
    return np.vstack(feats).T