"""
LianLianKan Parity Clusterer
Clusters matchable tiles using scikit-learn AgglomerativeClustering with shift-tolerant
template matching and enforces the Parity Constraint (all clusters even and >= 2).
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def cluster_matchable_tiles(matchable_tiles):
    """
    Takes a list of matchable tile dicts (each containing 'row', 'col', 'image', 'features').
    Returns:
        clusters: list of cluster dicts:
            {
                'cluster_id': int (1, 2, ...),
                'color_family': str,
                'count': int (guaranteed even >= 2),
                'coords': list of (row, col),
                'medoid_tile': dict,
                'template_image': np.ndarray
            }
        parity_valid: bool
    """
    # Group tiles by color family
    colors = sorted(list(set(t["features"]["color_family"] for t in matchable_tiles)))
    
    clusters = []
    cluster_id = 1
    all_parity_passed = True
    
    for col in colors:
        c_tiles = [t for t in matchable_tiles if t["features"]["color_family"] == col]
        m = len(c_tiles)
        if m == 0:
            continue
            
        # Parity check on color group itself
        if m % 2 != 0:
            print(f"[Warning] Color family {col} has odd count {m}!")
            all_parity_passed = False
            
        # Normalize cell size for correlation
        norm_size = 46
        tiles_img = [cv2.resize(t["image"], (norm_size, norm_size)) for t in c_tiles]
        # 34x34 center template allows ±6px translation search
        templates = [t[6:40, 6:40] for t in tiles_img]
        
        # Compute pairwise distance matrix using shift-tolerant NCC
        sim_mat = np.zeros((m, m), dtype=float)
        for i in range(m):
            sim_mat[i, i] = 1.0
            for j in range(i + 1, m):
                res1 = cv2.matchTemplate(tiles_img[j], templates[i], cv2.TM_CCOEFF_NORMED)
                val1 = cv2.minMaxLoc(res1)[1]
                res2 = cv2.matchTemplate(tiles_img[i], templates[j], cv2.TM_CCOEFF_NORMED)
                val2 = cv2.minMaxLoc(res2)[1]
                val = max(val1, val2)
                sim_mat[i, j] = val
                sim_mat[j, i] = val
                
        dist_mat = np.maximum(0, 1.0 - sim_mat)
        
        # Adaptive threshold sweep to find clustering satisfying Parity Constraint
        best_sol = None
        for linkage in ['complete', 'average']:
            for d in np.arange(0.015, 0.12, 0.002):
                ac = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage=linkage, distance_threshold=d)
                labels = ac.fit_predict(dist_mat)
                n_c = len(set(labels))
                counts = [np.sum(labels == k) for k in range(n_c)]
                if all(c % 2 == 0 for c in counts) and all(c >= 2 for c in counts):
                    best_sol = (labels, counts, dist_mat)
                    break
            if best_sol:
                break
                
        if not best_sol:
            print(f"[Error] Could not find even clustering for color {col}!")
            all_parity_passed = False
            # Fallback to single cluster
            labels = np.zeros(m, dtype=int)
            counts = [m]
            best_sol = (labels, counts, dist_mat)
            
        labels, counts, dist_mat = best_sol
        
        for k in range(len(counts)):
            indices = [i for i in range(m) if labels[i] == k]
            coords = [(c_tiles[i]["row"], c_tiles[i]["col"]) for i in indices]
            
            # Find medoid tile (minimum sum of distances to members)
            sub_dist = dist_mat[np.ix_(indices, indices)]
            medoid_sub_idx = np.argmin(np.sum(sub_dist, axis=1))
            medoid_tile = c_tiles[indices[medoid_sub_idx]]
            
            # Use normalized 45x45 medoid tile as standard template image
            tpl_img = cv2.resize(medoid_tile["image"], (45, 45))
            
            clusters.append({
                "cluster_id": cluster_id,
                "color_family": col,
                "count": len(indices),
                "coords": coords,
                "medoid_tile": medoid_tile,
                "template_image": tpl_img
            })
            cluster_id += 1
            
    return clusters, all_parity_passed
