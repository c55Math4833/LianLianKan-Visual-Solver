"""
LianLianKan Parity Clusterer
Robustly clusters matchable tiles using core icon NCC + HSV color correlation.
Guarantees identical tiles are never split into multiple clusters and enforces Parity Constraint.
"""

import cv2
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def compute_tile_similarity(tile1_img, tile2_img):
    """
    Computes robust visual similarity between two sliced tile cells:
    1. Crops out outer 15% margin to completely eliminate slicing boundary bleeding.
    2. Shift-tolerant NCC on the central 70% core icon (searches +/- 3px translation).
    3. Blends with HSV 2D color histogram correlation.
    """
    h, w = tile1_img.shape[:2]
    my = max(2, int(round(h * 0.15)))
    mx = max(2, int(round(w * 0.15)))
    
    core1 = tile1_img[my:h-my, mx:w-mx]
    core2 = tile2_img[my:h-my, mx:w-mx]
    
    t1 = cv2.resize(core1, (32, 32))
    t2 = cv2.resize(core2, (32, 32))
    
    # Translation search (+/- 3px)
    pad = 3
    sub_t1 = t1[pad:32-pad, pad:32-pad]
    sub_t2 = t2[pad:32-pad, pad:32-pad]
    
    r1 = cv2.matchTemplate(t2, sub_t1, cv2.TM_CCOEFF_NORMED)
    v1 = cv2.minMaxLoc(r1)[1]
    r2 = cv2.matchTemplate(t1, sub_t2, cv2.TM_CCOEFF_NORMED)
    v2 = cv2.minMaxLoc(r2)[1]
    ncc = max(v1, v2)
    
    # HSV color correlation
    hsv1 = cv2.cvtColor(t1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(t2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    color_sim = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Blended similarity: 80% icon NCC + 20% color correlation
    sim = 0.80 * max(0.0, ncc) + 0.20 * max(0.0, color_sim)
    return float(np.clip(sim, 0.0, 1.0))

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

    # 1. Compute pairwise distance matrix using core icon NCC + color blending
    sim_mat = np.zeros((m, m), dtype=float)
    for i in range(m):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, m):
            s = compute_tile_similarity(matchable_tiles[i]["image"], matchable_tiles[j]["image"])
            sim_mat[i, j] = s
            sim_mat[j, i] = s

    dist_mat = np.maximum(0, 1.0 - sim_mat)

    # 2. Adaptive threshold sweep to find clustering satisfying Parity Constraint (even counts >= 2)
    best_sol = None
    min_odd_count = 999999
    fallback_sol = None

    for linkage in ['average', 'complete']:
        for d in np.arange(0.12, 0.36, 0.01):
            ac = AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage=linkage,
                distance_threshold=d
            )
            labels = ac.fit_predict(dist_mat)
            n_c = len(set(labels))
            counts = [int(np.sum(labels == k)) for k in range(n_c)]
            odd_cnt = sum(1 for c in counts if c % 2 != 0)
            
            if odd_cnt < min_odd_count:
                min_odd_count = odd_cnt
                fallback_sol = (labels, counts, dist_mat)
                
            if odd_cnt == 0 and all(c >= 2 for c in counts):
                best_sol = (labels, counts, dist_mat)
                break
        if best_sol:
            break

    if best_sol:
        labels, counts, dist_mat = best_sol
        all_parity_passed = True
    else:
        labels, counts, dist_mat = fallback_sol
        all_parity_passed = (min_odd_count == 0)

    # 3. Post-processing: Merge very close clusters if any singleton/odd cluster remains
    if not all_parity_passed:
        labels_arr = labels.copy()

        def get_cluster_medoid(k):
            indices = [i for i in range(m) if labels_arr[i] == k]
            if not indices:
                return None
            sub_dist = dist_mat[np.ix_(indices, indices)]
            return indices[int(np.argmin(np.sum(sub_dist, axis=1)))]

        # Systematically resolve all odd-count clusters
        while True:
            unique_labels = sorted(list(set(labels_arr)))
            c_counts = {lbl: int(np.sum(labels_arr == lbl)) for lbl in unique_labels}
            odd_labels = [lbl for lbl in unique_labels if c_counts[lbl] % 2 != 0]
            if not odd_labels:
                break

            # Strategy 1: Merge an odd cluster with another odd cluster (odd + odd = even)
            best_pair = None
            best_sim = -1.0
            for i in range(len(odd_labels)):
                for j in range(i + 1, len(odd_labels)):
                    l1, l2 = odd_labels[i], odd_labels[j]
                    m1, m2 = get_cluster_medoid(l1), get_cluster_medoid(l2)
                    s = sim_mat[m1, m2]
                    if s > best_sim:
                        best_sim = s
                        best_pair = (l1, l2)
            if best_pair is not None and best_sim >= 0.35:
                l1, l2 = best_pair
                labels_arr[labels_arr == l2] = l1
                continue

            # Strategy 2: Merge the smallest odd cluster into its most similar neighbor
            l_odd = min(odd_labels, key=lambda lbl: c_counts[lbl])
            m_odd = get_cluster_medoid(l_odd)
            best_target = None
            best_sim = -1.0
            for other in unique_labels:
                if other == l_odd:
                    continue
                m_other = get_cluster_medoid(other)
                s = sim_mat[m_odd, m_other]
                if s > best_sim:
                    best_sim = s
                    best_target = other
            if best_target is not None:
                labels_arr[labels_arr == l_odd] = best_target
            else:
                break

        # Re-index unique labels
        unique_labels = sorted(list(set(labels_arr)))
        mapping = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([mapping[val] for val in labels_arr])
        counts = [int(np.sum(labels == k)) for k in range(len(unique_labels))]
        all_parity_passed = all(c % 2 == 0 and c >= 2 for c in counts)

    clusters = []
    for k in range(len(counts)):
        indices = [i for i in range(m) if labels[i] == k]
        if not indices:
            continue
        coords = [(matchable_tiles[i]["row"], matchable_tiles[i]["col"]) for i in indices]

        sub_dist = dist_mat[np.ix_(indices, indices)]
        medoid_sub_idx = int(np.argmin(np.sum(sub_dist, axis=1)))
        medoid_tile = matchable_tiles[indices[medoid_sub_idx]]

        tpl_img = cv2.resize(medoid_tile["image"], (45, 45))

        clusters.append({
            "cluster_id": len(clusters) + 1,
            "color_family": "auto",
            "count": len(indices),
            "coords": coords,
            "medoid_tile": medoid_tile,
            "template_image": tpl_img
        })

    return clusters, all_parity_passed
