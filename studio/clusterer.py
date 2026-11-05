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
        clusters: list of cluster dicts
        parity_valid: bool
    """
    m = len(matchable_tiles)
    if m == 0:
        return [], True

    # Normalize cell size for correlation
    norm_size = 36
    tiles_img = [cv2.resize(t["image"], (norm_size, norm_size)) for t in matchable_tiles]
    # Center template allows translation search
    pad = 5
    templates = [t[pad:norm_size-pad, pad:norm_size-pad] for t in tiles_img]

    # Compute pairwise distance matrix using shift-tolerant NCC
    sim_mat = np.zeros((m, m), dtype=float)
    for i in range(m):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, m):
            r1 = cv2.matchTemplate(tiles_img[j], templates[i], cv2.TM_CCOEFF_NORMED)
            v1 = cv2.minMaxLoc(r1)[1]
            r2 = cv2.matchTemplate(tiles_img[i], templates[j], cv2.TM_CCOEFF_NORMED)
            v2 = cv2.minMaxLoc(r2)[1]
            val = max(v1, v2)
            sim_mat[i, j] = val
            sim_mat[j, i] = val

    dist_mat = np.maximum(0, 1.0 - sim_mat)

    # Adaptive threshold sweep to find clustering satisfying Parity Constraint
    best_sol = None
    for linkage in ['complete', 'average']:
        for d in np.arange(0.04, 0.32, 0.01):
            ac = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage=linkage, distance_threshold=d)
            labels = ac.fit_predict(dist_mat)
            n_c = len(set(labels))
            counts = [int(np.sum(labels == k)) for k in range(n_c)]
            if all(c % 2 == 0 for c in counts) and all(c >= 2 for c in counts):
                best_sol = (labels, counts, dist_mat)
                break
        if best_sol:
            break

    if not best_sol:
        # Fallback to nearest threshold with minimum odd clusters
        best_d = 0.15
        ac = AgglomerativeClustering(n_clusters=None, metric='precomputed', linkage='complete', distance_threshold=best_d)
        labels = ac.fit_predict(dist_mat)
        n_c = len(set(labels))
        counts = [int(np.sum(labels == k)) for k in range(n_c)]
        best_sol = (labels, counts, dist_mat)
        all_parity_passed = False
    else:
        all_parity_passed = True

    labels, counts, dist_mat = best_sol
    clusters = []
    
    for k in range(len(counts)):
        indices = [i for i in range(m) if labels[i] == k]
        coords = [(matchable_tiles[i]["row"], matchable_tiles[i]["col"]) for i in indices]

        # Find medoid tile
        sub_dist = dist_mat[np.ix_(indices, indices)]
        medoid_sub_idx = int(np.argmin(np.sum(sub_dist, axis=1)))
        medoid_tile = matchable_tiles[indices[medoid_sub_idx]]

        tpl_img = cv2.resize(medoid_tile["image"], (45, 45))

        clusters.append({
            "cluster_id": k + 1,
            "color_family": "auto",
            "count": len(indices),
            "coords": coords,
            "medoid_tile": medoid_tile,
            "template_image": tpl_img
        })

    return clusters, all_parity_passed
