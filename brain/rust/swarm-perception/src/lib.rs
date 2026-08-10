//! swarm-perception core: the pure detector→bearing logic, with no I/O.
//!
//! main.rs (the file-driven CLI) and the ROS 2 node (swarm_perception_node,
//! which subscribes to sensor_msgs/Image) both call these functions. Extracting
//! them keeps the ONNX pipeline single-sourced — the ROS wrapper is a thin
//! transport shim over the same math, not a fork.

/// Greedy peak extraction: strict 8-neighborhood local maxima above `thresh`,
/// strongest-first, suppressing anything within `nms_radius`. Row-major `resp`.
pub fn find_peaks(
    resp: &[f32],
    w: usize,
    h: usize,
    thresh: f32,
    nms_radius: f32,
) -> Vec<(usize, usize, f32)> {
    let mut cands: Vec<(usize, usize, f32)> = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let v = resp[y * w + x];
            if v <= thresh {
                continue;
            }
            let mut is_max = true;
            'nb: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nv = resp[(y as i64 + dy) as usize * w + (x as i64 + dx) as usize];
                    if nv > v {
                        is_max = false;
                        break 'nb;
                    }
                }
            }
            if is_max {
                cands.push((x, y, v));
            }
        }
    }
    cands.sort_by(|a, b| b.2.total_cmp(&a.2));
    let r2 = nms_radius * nms_radius;
    let mut kept: Vec<(usize, usize, f32)> = Vec::new();
    for c in cands {
        if kept.iter().all(|k| {
            let dx = c.0 as f32 - k.0 as f32;
            let dy = c.1 as f32 - k.1 as f32;
            dx * dx + dy * dy > r2
        }) {
            kept.push(c);
        }
    }
    kept
}

/// Sub-pixel refinement: response-weighted centroid over a 7x7 window.
/// One pixel of column error is ~0.36 deg of bearing; the centroid buys most back.
pub fn centroid(resp: &[f32], w: usize, h: usize, px: usize, py: usize) -> (f64, f64) {
    let (mut sw, mut su, mut sv) = (0.0f64, 0.0f64, 0.0f64);
    for dy in -3i64..=3 {
        for dx in -3i64..=3 {
            let x = px as i64 + dx;
            let y = py as i64 + dy;
            if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                continue;
            }
            let wt = resp[y as usize * w + x as usize].max(0.0) as f64;
            sw += wt;
            su += wt * x as f64;
            sv += wt * y as f64;
        }
    }
    if sw <= 0.0 {
        (px as f64, py as f64)
    } else {
        (su / sw, sv / sw)
    }
}

/// Wrap an angle to (-pi, pi].
pub fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI
}

/// Pixel column + intrinsics + observer heading → world-frame bearing.
/// bearing_world = wrap(heading + atan((u - cx) / fx)).
pub fn column_to_world_bearing(u: f64, cx: f64, fx: f64, heading: f64) -> f64 {
    wrap(heading + ((u - cx) / fx).atan())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_blob_peak_and_bearing() {
        let (w, h) = (32usize, 24usize);
        let mut resp = vec![0.0f32; w * h];
        resp[12 * w + 16] = 1.0; // peak at (16, 12)
        let peaks = find_peaks(&resp, w, h, 0.05, 6.0);
        assert_eq!(peaks.len(), 1);
        assert_eq!((peaks[0].0, peaks[0].1), (16, 12));
        // centered column with heading 0 → bearing 0
        let b = column_to_world_bearing(16.0, 16.0, 160.0, 0.0);
        assert!(b.abs() < 1e-9);
    }

    #[test]
    fn nms_suppresses_neighbors() {
        let (w, h) = (32usize, 24usize);
        let mut resp = vec![0.0f32; w * h];
        resp[12 * w + 16] = 1.0;
        resp[12 * w + 18] = 0.8; // within nms_radius 6 → suppressed
        assert_eq!(find_peaks(&resp, w, h, 0.05, 6.0).len(), 1);
    }
}
