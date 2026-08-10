//! swarm-perception: ONNX detector -> bearing observations.
//!
//! The edge-inference half of Layer 2. Consumes downlinked onboard camera
//! frames, runs the detector through ONNX Runtime (the `ort` crate), extracts
//! sub-pixel blob centroids, and converts pixel columns into WORLD-FRAME
//! bearing observations the estimator can absorb as factors.
//!
//! Deliberate boundaries:
//!   - This node knows the camera intrinsics and the observer's heading
//!     (legitimate onboard information: calibration + compass/IMU yaw).
//!     It NEVER reads the ground-truth target table in meta.jsonl.
//!   - Detections carry NO identity. Deciding *which* neighbor a blob is
//!     (data association) belongs to the estimator, which has the predicted
//!     swarm state to match against. Emitting raw bearings keeps that seam
//!     honest -- and keeps this node swappable for a real camera feed.
//!   - The .onnx file is the entire perception model. Swapping the DoG
//!     frontend for a trained detector changes zero lines here.
//!
//! Usage:
//!   swarm-perception --frames ../frames --model ../detector.onnx \
//!                    --out ../bearings.jsonl

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use swarm_perception::{centroid, column_to_world_bearing, find_peaks};

const W: usize = 320;
const H: usize = 240;

#[derive(Parser)]
struct Args {
    /// Directory containing the rendered frames + meta.jsonl
    #[arg(long)]
    frames: PathBuf,
    /// Path to the ONNX detector
    #[arg(long)]
    model: PathBuf,
    /// Output JSONL of bearing observations
    #[arg(long)]
    out: PathBuf,
    /// Detector response threshold
    #[arg(long, default_value_t = 0.05)]
    thresh: f32,
    /// Non-maximum-suppression radius in pixels
    #[arg(long, default_value_t = 6.0)]
    nms_radius: f32,
    /// Process at most this many images (0 = all)
    #[arg(long, default_value_t = 0)]
    limit: usize,
}

/// What the node is ALLOWED to know about each frame. serde skips the
/// ground-truth `truth` table in meta.jsonl by simply not declaring it.
#[derive(Deserialize)]
struct FrameMeta {
    file: String,
    t: u32,
    observer: u32,
    heading: f64,
    #[serde(rename = "K")]
    k: Intrinsics,
}

#[derive(Deserialize)]
struct Intrinsics {
    fx: f64,
    cx: f64,
}

#[derive(Serialize)]
struct BearingObs {
    t: u32,
    observer: u32,
    u: f64,
    v: f64,
    conf: f32,
    /// world-frame bearing = observer heading + atan((u - cx) / fx)
    bearing_world: f64,
}

/// ort's error type holds raw pointers (not Send+Sync), so it can't ride `?`
/// into anyhow directly - flatten it through Display.
macro_rules! ort_try {
    ($e:expr) => {
        $e.map_err(|e| anyhow::anyhow!("ort: {e}"))?
    };
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut session = ort_try!(ort_try!(ort_try!(Session::builder())
        .with_intra_threads(2))
    .commit_from_file(&args.model));

    let meta_path = args.frames.join("meta.jsonl");
    let reader = BufReader::new(File::open(&meta_path).context("opening meta.jsonl")?);
    let mut out = BufWriter::new(File::create(&args.out)?);

    let mut n_images = 0usize;
    let mut n_obs = 0usize;
    let mut infer_ms: Vec<f64> = Vec::new();
    let t_start = Instant::now();

    for line in reader.lines() {
        let meta: FrameMeta = serde_json::from_str(&line?)?;
        if args.limit > 0 && n_images >= args.limit {
            break;
        }

        // ---- load frame -> normalized f32 tensor -------------------------
        let img = image::open(args.frames.join(&meta.file))
            .with_context(|| format!("loading {}", meta.file))?
            .into_luma8();
        anyhow::ensure!(
            img.width() as usize == W && img.height() as usize == H,
            "unexpected image size in {}",
            meta.file
        );
        let data: Vec<f32> = img.as_raw().iter().map(|&p| p as f32 / 255.0).collect();

        // ---- ONNX inference ---------------------------------------------
        let t0 = Instant::now();
        let input = ort_try!(Tensor::from_array(([1usize, 1, H, W], data)));
        let outputs = ort_try!(session.run(ort::inputs!["image" => input]));
        let (_, resp) = ort_try!(outputs["response"].try_extract_tensor::<f32>());
        let resp: &[f32] = resp;

        // ---- peaks -> sub-pixel -> bearings (shared with the ROS node) ---
        for (px, py, conf) in find_peaks(resp, W, H, args.thresh, args.nms_radius) {
            let (u, v) = centroid(resp, W, H, px, py);
            let obs = BearingObs {
                t: meta.t,
                observer: meta.observer,
                u,
                v,
                conf,
                bearing_world: column_to_world_bearing(u, meta.k.cx, meta.k.fx, meta.heading),
            };
            serde_json::to_writer(&mut out, &obs)?;
            out.write_all(b"\n")?;
            n_obs += 1;
        }
        infer_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        n_images += 1;
    }
    out.flush()?;

    infer_ms.sort_by(|a, b| a.total_cmp(b));
    let pct = |p: f64| infer_ms[(p * (infer_ms.len() - 1) as f64) as usize];
    let wall = t_start.elapsed().as_secs_f64();
    eprintln!(
        "swarm-perception: {} images -> {} bearing observations in {:.1}s ({:.0} img/s)",
        n_images,
        n_obs,
        wall,
        n_images as f64 / wall
    );
    eprintln!(
        "inference+extract per image: p50 {:.2} ms  p99 {:.2} ms",
        pct(0.50),
        pct(0.99)
    );
    Ok(())
}
