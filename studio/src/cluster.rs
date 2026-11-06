//! Tile similarity + parity-constrained clustering. Port of studio/clusterer.py.

use crate::imgproc::{pyround, resize, rgb_to_hsv, Image};

const SZ: usize = 32; // cores are normalised to 32x32
const PAD: usize = 3;
const SUB: usize = SZ - 2 * PAD; // 26

/// Everything about one tile that the pairwise similarity needs, computed once.
pub struct Prepared {
    /// 32x32 core, interleaved RGB as f64
    full: Vec<f64>,
    /// per-channel 2-D prefix sums of value and value^2, (SZ+1)^2 each, [channel]
    ps: [Vec<f64>; 3],
    ps2: [Vec<f64>; 3],
    /// centred 26x26 sub-template (interleaved) and its squared norm
    sub_c: Vec<f64>,
    sub_norm2: f64,
    hist: Vec<f64>,
}

pub fn prepare(t: &Image) -> Prepared {
    let my = 2i64.max(pyround(t.h as f64 * 0.15)) as usize;
    let mx = 2i64.max(pyround(t.w as f64 * 0.15)) as usize;
    let core = resize(&t.crop(mx, my, t.w.saturating_sub(2 * mx), t.h.saturating_sub(2 * my)), SZ, SZ);
    let full: Vec<f64> = core.data.iter().map(|&v| v as f64).collect();

    let mut ps: [Vec<f64>; 3] = [vec![0.0; (SZ + 1) * (SZ + 1)], vec![0.0; (SZ + 1) * (SZ + 1)], vec![0.0; (SZ + 1) * (SZ + 1)]];
    let mut ps2 = ps.clone();
    for c in 0..3 {
        for y in 0..SZ {
            for x in 0..SZ {
                let v = full[(y * SZ + x) * 3 + c];
                let i = (y + 1) * (SZ + 1) + x + 1;
                ps[c][i] = v + ps[c][i - 1] + ps[c][i - (SZ + 1)] - ps[c][i - (SZ + 1) - 1];
                ps2[c][i] = v * v + ps2[c][i - 1] + ps2[c][i - (SZ + 1)] - ps2[c][i - (SZ + 1) - 1];
            }
        }
    }
    // sub-template = core[PAD..SZ-PAD, PAD..SZ-PAD], mean removed per channel
    let mut mean = [0.0f64; 3];
    for y in 0..SUB {
        for x in 0..SUB {
            for c in 0..3 {
                mean[c] += full[((y + PAD) * SZ + x + PAD) * 3 + c];
            }
        }
    }
    mean.iter_mut().for_each(|m| *m /= (SUB * SUB) as f64);
    let mut sub_c = Vec::with_capacity(SUB * SUB * 3);
    let mut sub_norm2 = 0.0;
    for y in 0..SUB {
        for x in 0..SUB {
            for c in 0..3 {
                let v = full[((y + PAD) * SZ + x + PAD) * 3 + c] - mean[c];
                sub_c.push(v);
                sub_norm2 += v * v;
            }
        }
    }
    Prepared { full, ps, ps2, sub_c, sub_norm2, hist: hsv_hist(&core) }
}

/// max over positions of TM_CCOEFF_NORMED(image = `img`, templ = `other`'s sub-template).
fn match_max(img: &Prepared, other: &Prepared) -> f64 {
    let n = (SUB * SUB) as f64;
    let rect = |t: &Vec<f64>, ox: usize, oy: usize| {
        let w = SZ + 1;
        t[(oy + SUB) * w + ox + SUB] - t[oy * w + ox + SUB] - t[(oy + SUB) * w + ox] + t[oy * w + ox]
    };
    let mut best = f64::MIN;
    for oy in 0..=(SZ - SUB) {
        for ox in 0..=(SZ - SUB) {
            // numerator: sum(I * T_centred) (T_centred sums to 0 per channel)
            let mut num = 0.0;
            for y in 0..SUB {
                let irow = &img.full[((oy + y) * SZ + ox) * 3..((oy + y) * SZ + ox + SUB) * 3];
                let trow = &other.sub_c[y * SUB * 3..(y + 1) * SUB * 3];
                num += irow.iter().zip(trow).map(|(a, b)| a * b).sum::<f64>();
            }
            let mut inorm2 = 0.0;
            for c in 0..3 {
                let s = rect(&img.ps[c], ox, oy);
                inorm2 += rect(&img.ps2[c], ox, oy) - s * s / n;
            }
            let denom = (inorm2 * other.sub_norm2).sqrt();
            let r = if denom > 1e-12 { num / denom } else { 0.0 };
            if r > best {
                best = r;
            }
        }
    }
    best
}

fn hsv_hist(img: &Image) -> Vec<f64> {
    let mut hist = vec![0.0f64; 64];
    for p in img.data.chunks_exact(3) {
        let (h, s, _) = rgb_to_hsv(p[0], p[1], p[2]);
        let hb = ((h as f64 * 8.0 / 180.0) as usize).min(7);
        let sb = ((s as f64 * 8.0 / 256.0) as usize).min(7);
        hist[hb * 8 + sb] += 1.0;
    }
    let norm = hist.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        hist.iter_mut().for_each(|v| *v /= norm);
    }
    hist
}

/// cv2.compareHist(..., HISTCMP_CORREL)
fn hist_correl(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        num += (a[i] - ma) * (b[i] - mb);
        da += (a[i] - ma).powi(2);
        db += (b[i] - mb).powi(2);
    }
    if da.abs() > f64::EPSILON && db.abs() > f64::EPSILON {
        num / (da * db).sqrt()
    } else {
        1.0
    }
}

/// compute_tile_similarity(): 80 % shift-tolerant icon NCC + 20 % HSV histogram correlation.
pub fn similarity(a: &Prepared, b: &Prepared) -> f64 {
    let ncc = match_max(b, a).max(match_max(a, b));
    let color = hist_correl(&a.hist, &b.hist);
    (0.80 * ncc.max(0.0) + 0.20 * color.max(0.0)).clamp(0.0, 1.0)
}

#[allow(dead_code)]
pub fn tile_similarity(a: &Image, b: &Image) -> f64 {
    similarity(&prepare(a), &prepare(b))
}

#[derive(Clone, Copy, PartialEq)]
pub enum Linkage {
    Average,
    Complete,
}

/// Agglomerative clustering on a precomputed distance matrix (same tie-breaking as Python).
pub fn agglomerative(dist: &[Vec<f64>], threshold: f64, linkage: Linkage) -> Vec<usize> {
    let n = dist.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }
    let mut clusters: Vec<Option<Vec<usize>>> = (0..n).map(|i| Some(vec![i])).collect();
    let mut active: Vec<usize> = (0..n).collect();
    let mut d: Vec<Vec<f64>> = dist.to_vec();
    let mut sizes = vec![1usize; n];
    while active.len() > 1 {
        let mut min_d = f64::INFINITY;
        let mut best: Option<(usize, usize)> = None;
        for a in 0..active.len() {
            for b in (a + 1)..active.len() {
                let (i, j) = (active[a], active[b]);
                if d[i][j] < min_d {
                    min_d = d[i][j];
                    best = Some((i, j));
                }
            }
        }
        let (ci, cj) = match best {
            Some(p) if min_d <= threshold => p,
            _ => break,
        };
        let (si, sj) = (sizes[ci] as f64, sizes[cj] as f64);
        for &k in &active {
            if k == ci || k == cj {
                continue;
            }
            let nd = match linkage {
                Linkage::Complete => d[ci][k].max(d[cj][k]),
                Linkage::Average => (si * d[ci][k] + sj * d[cj][k]) / (si + sj),
            };
            d[ci][k] = nd;
            d[k][ci] = nd;
        }
        let moved = clusters[cj].take().unwrap();
        clusters[ci].as_mut().unwrap().extend(moved);
        sizes[ci] += sizes[cj];
        sizes[cj] = 0;
        active.retain(|&x| x != cj);
    }
    let mut labels = vec![0usize; n];
    for (lbl, &ci) in active.iter().enumerate() {
        for &e in clusters[ci].as_ref().unwrap() {
            labels[e] = lbl;
        }
    }
    labels
}

pub struct Cluster {
    pub id: usize,
    pub members: Vec<usize>, // indices into the input slice
    pub medoid: usize,
}

fn counts_of(labels: &[usize]) -> Vec<usize> {
    let nc = labels.iter().max().map(|m| m + 1).unwrap_or(0);
    (0..nc).map(|k| labels.iter().filter(|&&l| l == k).count()).collect()
}

/// cluster_matchable_tiles(): returns clusters (ids 1..) and whether every count is even.
pub fn cluster_tiles(images: &[&Image]) -> (Vec<Cluster>, bool) {
    let m = images.len();
    if m == 0 {
        return (vec![], true);
    }
    let prepared: Vec<Prepared> = images.iter().map(|im| prepare(im)).collect();
    // Pairwise similarities are independent: spread the rows over all cores (std only).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
    let mut sim = vec![vec![0.0f64; m]; m];
    let parts: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut local = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= m {
                            break;
                        }
                        let row: Vec<f64> = ((i + 1)..m).map(|j| similarity(&prepared[i], &prepared[j])).collect();
                        local.push((i, row));
                    }
                    local
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for i in 0..m {
        sim[i][i] = 1.0;
    }
    for (i, row) in parts.into_iter().flatten() {
        for (k, s) in row.into_iter().enumerate() {
            let j = i + 1 + k;
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }
    let dist: Vec<Vec<f64>> = sim.iter().map(|r| r.iter().map(|s| (1.0 - s).max(0.0)).collect()).collect();

    let mut best: Option<Vec<usize>> = None;
    let mut min_odd = usize::MAX;
    let mut fallback: Option<Vec<usize>> = None;
    'outer: for linkage in [Linkage::Average, Linkage::Complete] {
        for i in 0..24 {
            let d = 0.12 + i as f64 * 0.01;
            let labels = agglomerative(&dist, d, linkage);
            let counts = counts_of(&labels);
            let odd = counts.iter().filter(|c| *c % 2 != 0).count();
            if odd < min_odd {
                min_odd = odd;
                fallback = Some(labels.clone());
            }
            if odd == 0 && counts.iter().all(|&c| c >= 2) {
                best = Some(labels);
                break 'outer;
            }
        }
    }
    let (mut labels, mut ok) = match best {
        Some(l) => (l, true),
        None => (fallback.unwrap(), min_odd == 0),
    };

    if !ok {
        let medoid_of = |labels: &[usize], k: usize| -> Option<usize> {
            let idx: Vec<usize> = (0..m).filter(|&i| labels[i] == k).collect();
            if idx.is_empty() {
                return None;
            }
            let mut best = (f64::INFINITY, idx[0]);
            for &i in &idx {
                let s: f64 = idx.iter().map(|&j| dist[i][j]).sum();
                if s < best.0 {
                    best = (s, i);
                }
            }
            Some(best.1)
        };
        loop {
            let mut uniq: Vec<usize> = labels.clone();
            uniq.sort();
            uniq.dedup();
            let cnt = |l: usize| labels.iter().filter(|&&x| x == l).count();
            let odd: Vec<usize> = uniq.iter().cloned().filter(|&l| cnt(l) % 2 != 0).collect();
            if odd.is_empty() {
                break;
            }
            // Strategy 1: merge two odd clusters
            let mut best_pair: Option<(usize, usize)> = None;
            let mut best_sim = -1.0;
            for a in 0..odd.len() {
                for b in (a + 1)..odd.len() {
                    let (m1, m2) = (medoid_of(&labels, odd[a]).unwrap(), medoid_of(&labels, odd[b]).unwrap());
                    if sim[m1][m2] > best_sim {
                        best_sim = sim[m1][m2];
                        best_pair = Some((odd[a], odd[b]));
                    }
                }
            }
            if let Some((l1, l2)) = best_pair {
                if best_sim >= 0.35 {
                    for l in labels.iter_mut() {
                        if *l == l2 {
                            *l = l1;
                        }
                    }
                    continue;
                }
            }
            // Strategy 2: merge the smallest odd cluster into its most similar neighbour
            let l_odd = *odd.iter().min_by_key(|&&l| cnt(l)).unwrap();
            let m_odd = medoid_of(&labels, l_odd).unwrap();
            let mut target: Option<usize> = None;
            let mut bs = -1.0;
            for &other in &uniq {
                if other == l_odd {
                    continue;
                }
                let mo = medoid_of(&labels, other).unwrap();
                if sim[m_odd][mo] > bs {
                    bs = sim[m_odd][mo];
                    target = Some(other);
                }
            }
            match target {
                Some(t) => {
                    for l in labels.iter_mut() {
                        if *l == l_odd {
                            *l = t;
                        }
                    }
                }
                None => break,
            }
        }
        let mut uniq: Vec<usize> = labels.clone();
        uniq.sort();
        uniq.dedup();
        labels = labels.iter().map(|l| uniq.iter().position(|u| u == l).unwrap()).collect();
        ok = counts_of(&labels).iter().all(|&c| c % 2 == 0 && c >= 2);
    }

    let nc = counts_of(&labels).len();
    let mut clusters = Vec::new();
    for k in 0..nc {
        let idx: Vec<usize> = (0..m).filter(|&i| labels[i] == k).collect();
        if idx.is_empty() {
            continue;
        }
        let mut best = (f64::INFINITY, idx[0]);
        for &i in &idx {
            let s: f64 = idx.iter().map(|&j| dist[i][j]).sum();
            if s < best.0 {
                best = (s, i);
            }
        }
        clusters.push(Cluster { id: clusters.len() + 1, members: idx, medoid: best.1 });
    }
    (clusters, ok)
}
