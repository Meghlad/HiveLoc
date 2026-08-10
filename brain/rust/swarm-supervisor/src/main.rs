//! swarm-supervisor (binary): the only component that can emit a setpoint.
//!
//! Reads a plan (JSON, from the VLM planner via file or stdin) + the latest
//! estimator snapshot, validates with the pure library gate, prints the
//! Decision as JSON. Only on ACCEPT - and only with --emit - does it encode
//! SET_POSITION_TARGET_LOCAL_NED per assigned vehicle and send it toward that
//! vehicle's MAVLink port. A rejected plan produces exactly zero packets.
//!
//!   plan.json --> [validate: geofence, spacing, covariance, freshness] --+--> decision.json
//!                                                                       |
//!                                                          (accept + --emit only)
//!                                                                       v
//!                                                    SET_POSITION_TARGET_LOCAL_NED
//!
//! Usage:
//!   swarm-supervisor --plan plan.json --estimate estimate.json [--emit]
//!   cat plan.json | swarm-supervisor --estimate estimate.json

use std::io::Read;
use std::net::UdpSocket;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Parser;
use mavlink::common::{MavMessage, PositionTargetTypemask, SET_POSITION_TARGET_LOCAL_NED_DATA};
use mavlink::{write_versioned_msg, MavHeader, MavlinkVersion};
use swarm_supervisor::{validate, EstimateSnapshot, Plan, SupervisorConfig};

#[derive(Parser)]
struct Args {
    /// plan JSON file ("-" or omitted = stdin)
    #[arg(long, default_value = "-")]
    plan: String,
    /// estimator snapshot JSON file
    #[arg(long)]
    estimate: String,
    /// supervisor config JSON (defaults used if omitted)
    #[arg(long)]
    config: Option<String>,
    /// actually transmit setpoints on accept (otherwise: decide only)
    #[arg(long, default_value_t = false)]
    emit: bool,
    /// first vehicle's MAVLink port (SITL -I0 pattern)
    #[arg(long, default_value_t = 14551)]
    base_port: u16,
    #[arg(long, default_value_t = 10)]
    port_stride: u16,
    /// meters of NED per estimator unit (same as swarm-link)
    #[arg(long, default_value_t = 5.0)]
    scale: f64,
    /// commanded altitude (m) -> down = -alt
    #[arg(long, default_value_t = 2.0)]
    alt: f64,
    /// override "now" for testing (unix ms)
    #[arg(long)]
    now_ms: Option<u64>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let plan_raw = if args.plan == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        s
    } else {
        std::fs::read_to_string(&args.plan).context("reading plan")?
    };
    // Free text cannot reach the aircraft: if this parse fails, we're done.
    let plan: Plan = serde_json::from_str(&plan_raw).context("plan is not valid JSON")?;

    let est: EstimateSnapshot =
        serde_json::from_str(&std::fs::read_to_string(&args.estimate)?)
            .context("estimate is not valid JSON")?;

    let cfg: SupervisorConfig = match &args.config {
        Some(p) => serde_json::from_str(&std::fs::read_to_string(p)?)?,
        None => SupervisorConfig::default(),
    };

    let now = args.now_ms.unwrap_or_else(|| {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64
    });

    let decision = validate(&plan, &est, &cfg, now);
    println!("{}", serde_json::to_string_pretty(&decision)?);

    if !decision.accepted {
        eprintln!("supervisor: plan '{}' REJECTED - no setpoints emitted", plan.plan_id);
        std::process::exit(2);
    }

    if args.emit {
        let sock = UdpSocket::bind("127.0.0.1:0")?;
        // position-only typemask: ignore velocity/accel/yaw fields
        let mask = PositionTargetTypemask::POSITION_TARGET_TYPEMASK_VX_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_VY_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AX_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AY_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_AZ_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | PositionTargetTypemask::POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE;
        for (i, a) in plan.assignments.iter().enumerate() {
            let msg = MavMessage::SET_POSITION_TARGET_LOCAL_NED(
                SET_POSITION_TARGET_LOCAL_NED_DATA {
                    time_boot_ms: now as u32,
                    x: (a.waypoint_ne[0] * args.scale) as f32,
                    y: (a.waypoint_ne[1] * args.scale) as f32,
                    z: -args.alt as f32,
                    vx: 0.0, vy: 0.0, vz: 0.0,
                    afx: 0.0, afy: 0.0, afz: 0.0,
                    yaw: 0.0,
                    yaw_rate: 0.0,
                    type_mask: mask,
                    target_system: 1,
                    target_component: 1,
                    coordinate_frame: mavlink::common::MavFrame::MAV_FRAME_LOCAL_NED,
                },
            );
            let header = MavHeader { system_id: 255, component_id: 190, sequence: i as u8 };
            let mut buf = Vec::with_capacity(96);
            write_versioned_msg(&mut buf, MavlinkVersion::V2, header, &msg)?;
            let port = args.base_port + (a.vehicle as u16) * args.port_stride;
            sock.send_to(&buf, ("127.0.0.1", port))?;
            eprintln!(
                "supervisor: vehicle {} -> setpoint NED ({:.2}, {:.2}, {:.2}) via udp:{}",
                a.vehicle,
                a.waypoint_ne[0] * args.scale,
                a.waypoint_ne[1] * args.scale,
                -args.alt,
                port
            );
        }
    } else {
        eprintln!("supervisor: plan '{}' ACCEPTED (decide-only; use --emit to transmit)",
                  plan.plan_id);
    }
    Ok(())
}
