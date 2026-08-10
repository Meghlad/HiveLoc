//! swarm-link: the transport half of close_the_loop.py, made real.
//!
//! The README wrote this program's spec before any Rust existed:
//!   "the real system splits estimation from transport - one estimator thread
//!    solving the swarm, and one lightweight ~20 Hz sender per vehicle
//!    re-streaming that vehicle's latest position, decoupled from solver speed."
//!
//! Architecture (one process, three kinds of task):
//!
//!   estimator (Python, any rate) --UDP JSON--> [ingest task]
//!                                                  |  watch::Sender<Snapshot>  (one per vehicle)
//!                                     +------------+------------+
//!                                     v            v            v
//!                               [sender 0]   [sender 1]  ...  [sender N-1]     20 Hz each
//!                                     |            |            |
//!                                MAVLink v2 VISION_POSITION_ESTIMATE
//!                                     v            v            v
//!                               udp:14551     udp:14561   ...  (one SITL per vehicle)
//!
//! Why the ownership story matters: the ingest task OWNS the estimate; each
//! sender can only `borrow()` an immutable snapshot from its watch channel.
//! That "estimator-owns-state / senders-borrow-snapshots" split is not a
//! convention here - the borrow checker rejects any code that violates it.
//! A sender that missed an update simply re-sends the last snapshot, so the
//! EKF never starves no matter how slow the solver is (the "nav heartbeat
//! that never dies", per vehicle, without a single sleep() in a loop that
//! also does math).
//!
//! Built-in evidence: every sender records estimate-to-wire latency (ingest
//! timestamp -> first transmission of that frame) in an HDR histogram; the
//! {"end":true} sentinel dumps p50/p99 and per-vehicle achieved rates. The
//! benchmark is not a harness bolted on - the binary measures itself.
//!
//! Run:  swarm-link --vehicles 12 --csv latencies.csv
//! Feed: {"seq":1,"pos":[[x,y],...12]}\n  as UDP datagrams to --ingest-port.

use std::net::UdpSocket;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use hdrhistogram::Histogram;
use mavlink::common::{MavMessage, VISION_POSITION_ESTIMATE_DATA};
use mavlink::{write_versioned_msg, MavHeader, MavlinkVersion};
use nalgebra::{Matrix2, Rotation2, Vector2};
use serde::Deserialize;
use tokio::sync::{mpsc, watch};

#[derive(Parser, Clone)]
struct Args {
    /// Number of vehicles (one 20 Hz sender each)
    #[arg(long, default_value_t = 12)]
    vehicles: usize,
    /// UDP port to receive estimator frames on
    #[arg(long, default_value_t = 47001)]
    ingest_port: u16,
    /// First vehicle's MAVLink UDP port (SITL -I0 pattern: 14551, 14561, ...)
    #[arg(long, default_value_t = 14551)]
    base_port: u16,
    /// Port stride between vehicles
    #[arg(long, default_value_t = 10)]
    port_stride: u16,
    /// Per-vehicle send rate (Hz)
    #[arg(long, default_value_t = 20.0)]
    rate: f64,
    /// meters of NED travel per estimator unit
    #[arg(long, default_value_t = 5.0)]
    scale: f64,
    /// map rotation (deg): estimator frame -> NED frame
    #[arg(long, default_value_t = 0.0)]
    map_rotation_deg: f64,
    /// held altitude (m) -> down = -alt
    #[arg(long, default_value_t = 2.0)]
    alt: f64,
    /// write per-send latency samples here
    #[arg(long, default_value = "swarm_link_latencies.csv")]
    csv: String,
}

/// One estimator frame off the wire. `cov` (per-vehicle marginal covariance
/// trace) is optional today; Layer 3's supervisor consumes it.
#[derive(Deserialize)]
struct EstimateFrame {
    #[serde(default)]
    seq: u64,
    #[serde(default)]
    end: bool,
    #[serde(default)]
    pos: Vec<[f64; 2]>,
    #[serde(default)]
    #[allow(dead_code)]
    cov: Vec<f64>,
}

/// What a sender is allowed to see: an immutable snapshot.
#[derive(Clone, Copy, Default)]
struct Snapshot {
    seq: u64,
    ned_north: f64,
    ned_east: f64,
    ned_down: f64,
    /// when the ingest task took ownership of this estimate
    recv: Option<InstantWrap>,
}

/// Instant isn't Default; tiny wrapper so Snapshot::default() exists.
#[derive(Clone, Copy)]
struct InstantWrap(Instant);

/// Estimator frame -> NED, as a proper rigid transform (nalgebra), not ad-hoc
/// arithmetic: p_ned = R(theta) * S * p_est ; down is constant (baro owns Z).
struct FrameMap {
    a: Matrix2<f64>,
    down: f64,
}

impl FrameMap {
    fn new(scale: f64, rot_deg: f64, alt: f64) -> Self {
        let r: Matrix2<f64> = *Rotation2::new(rot_deg.to_radians()).matrix();
        FrameMap { a: r * Matrix2::from_diagonal_element(scale), down: -alt }
    }
    fn apply(&self, p: [f64; 2]) -> (f64, f64, f64) {
        let v = self.a * Vector2::new(p[0], p[1]);
        (v.x, v.y, self.down)
    }
}

/// Per-vehicle report sent back to main at shutdown.
struct SenderReport {
    vehicle: usize,
    hist: Histogram<u64>,
    sends: u64,
    elapsed: Duration,
    samples: Vec<(u64, u64)>, // (seq, latency_us)
}

async fn sender_task(
    vehicle: usize,
    port: u16,
    rate_hz: f64,
    mut rx: watch::Receiver<Snapshot>,
    mut stop: watch::Receiver<bool>,
    report_tx: mpsc::Sender<SenderReport>,
    boot: Instant,
) -> Result<()> {
    // Plain std UdpSocket: sends never block meaningfully and we want the
    // cheapest possible wire path. Unconnected + sendto, same as pymavlink.
    let sock = UdpSocket::bind("127.0.0.1:0")?;
    let target = format!("127.0.0.1:{port}");

    let mut hist = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3)?;
    let mut samples = Vec::new();
    let mut last_seq_sent = u64::MAX;
    let mut seq_no: u8 = 0;
    let mut sends: u64 = 0;
    let t0 = Instant::now();

    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / rate_hz));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // NEWS vs HEARTBEAT: a fresh estimate is transmitted the moment it lands
    // (rx.changed() wakes us); the 20 Hz tick only re-sends the last snapshot
    // so the EKF never starves between estimator frames. This is also what
    // makes the latency histogram honest: estimate-to-wire measures
    // parse -> transform -> encode -> syscall, not the phase of a timer.
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            r = rx.changed() => { if r.is_err() { break } }
            _ = stop.changed() => break,
        }
        // Borrow the latest snapshot. This is the ONLY access senders have to
        // estimator state, and it is immutable by construction.
        let snap = *rx.borrow_and_update();
        let Some(recv) = snap.recv else { continue };  // nothing ingested yet

        let usec = boot.elapsed().as_micros() as u64;
        let msg = MavMessage::VISION_POSITION_ESTIMATE(VISION_POSITION_ESTIMATE_DATA {
            usec,
            x: snap.ned_north as f32,
            y: snap.ned_east as f32,
            z: snap.ned_down as f32,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
        });
        let header = MavHeader { system_id: 255, component_id: 191, sequence: seq_no };
        seq_no = seq_no.wrapping_add(1);

        let mut buf = Vec::with_capacity(64);
        write_versioned_msg(&mut buf, MavlinkVersion::V2, header, &msg)?;
        sock.send_to(&buf, &target)?;
        sends += 1;

        // estimate-to-wire: only the FIRST transmission of a fresh estimate
        // counts (re-sends are heartbeats, not news).
        if snap.seq != last_seq_sent {
            last_seq_sent = snap.seq;
            let lat_us = recv.0.elapsed().as_micros() as u64;
            hist.record(lat_us.max(1)).ok();
            samples.push((snap.seq, lat_us));
        }
    }

    report_tx
        .send(SenderReport { vehicle, hist, sends, elapsed: t0.elapsed(), samples })
        .await
        .ok();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let n = args.vehicles;
    let map = FrameMap::new(args.scale, args.map_rotation_deg, args.alt);
    let boot = Instant::now();

    // One watch channel per vehicle: ingest owns the Sender side, exclusively.
    let mut txs = Vec::with_capacity(n);
    let mut sender_handles = Vec::with_capacity(n);
    let (stop_tx, stop_rx) = watch::channel(false);
    let (report_tx, mut report_rx) = mpsc::channel(n);

    for v in 0..n {
        let (tx, rx) = watch::channel(Snapshot::default());
        txs.push(tx);
        let port = args.base_port + (v as u16) * args.port_stride;
        sender_handles.push(tokio::spawn(sender_task(
            v,
            port,
            args.rate,
            rx,
            stop_rx.clone(),
            report_tx.clone(),
            boot,
        )));
    }
    drop(report_tx);

    // Ingest loop: this task is the sole owner of estimator state.
    let ingest = tokio::net::UdpSocket::bind(("127.0.0.1", args.ingest_port)).await?;
    eprintln!(
        "swarm-link: {} vehicles at {:.0} Hz, ingest udp:{}, targets {}..{} stride {}",
        n,
        args.rate,
        args.ingest_port,
        args.base_port,
        args.base_port + ((n as u16).saturating_sub(1)) * args.port_stride,
        args.port_stride
    );

    let mut buf = vec![0u8; 65536];
    let mut frames: u64 = 0;
    loop {
        let (len, _) = ingest.recv_from(&mut buf).await?;
        let recv = Instant::now(); // estimate-to-wire clock starts HERE
        let frame: EstimateFrame = match serde_json::from_slice(&buf[..len]) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("swarm-link: bad frame ignored: {e}");
                continue;
            }
        };
        if frame.end {
            break;
        }
        if frame.pos.len() < n {
            eprintln!(
                "swarm-link: frame has {} vehicles, need {} - ignored",
                frame.pos.len(),
                n
            );
            continue;
        }
        for (v, tx) in txs.iter().enumerate() {
            let (north, east, down) = map.apply(frame.pos[v]);
            tx.send_replace(Snapshot {
                seq: frame.seq,
                ned_north: north,
                ned_east: east,
                ned_down: down,
                recv: Some(InstantWrap(recv)),
            });
        }
        frames += 1;
    }

    // Clean shutdown: senders drain, then report their histograms.
    stop_tx.send(true).ok();
    for h in sender_handles {
        h.await??;
    }

    let mut total = Histogram::<u64>::new_with_bounds(1, 10_000_000, 3)?;
    let mut csv = String::from("vehicle,seq,latency_us\n");
    let mut lines = Vec::new();
    while let Some(r) = report_rx.recv().await {
        total.add(&r.hist).ok();
        let rate = r.sends as f64 / r.elapsed.as_secs_f64();
        lines.push((
            r.vehicle,
            format!(
                "  vehicle {:2}: p50 {:>6} us  p99 {:>7} us  max {:>7} us  achieved {:>5.1} Hz",
                r.vehicle,
                r.hist.value_at_quantile(0.50),
                r.hist.value_at_quantile(0.99),
                r.hist.max(),
                rate
            ),
        ));
        for (seq, lat) in &r.samples {
            csv.push_str(&format!("{},{},{}\n", r.vehicle, seq, lat));
        }
    }
    lines.sort_by_key(|(v, _)| *v);
    eprintln!("swarm-link: {} frames ingested", frames);
    for (_, l) in &lines {
        eprintln!("{l}");
    }
    eprintln!(
        "TOTAL estimate-to-wire: p50 {} us  p90 {} us  p99 {} us  p99.9 {} us  (n={})",
        total.value_at_quantile(0.50),
        total.value_at_quantile(0.90),
        total.value_at_quantile(0.99),
        total.value_at_quantile(0.999),
        total.len()
    );
    std::fs::write(&args.csv, csv)?;
    eprintln!("wrote {}", args.csv);
    Ok(())
}
