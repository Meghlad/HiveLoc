//! supervisor_node: the safety supervisor, as a ROS 2 node.
//!
//! This is the "ROS gap" fix that reinforces the Rust thread instead of forking
//! into a separate Python demo. The node is a thin rclrs shim over the SAME
//! `swarm_supervisor` crate the standalone binary and the 11 unit tests use —
//! validation logic lives in one place.
//!
//!   /swarm_estimate (SwarmEstimate) ─┐
//!                                     ├─▶ [supervisor_node] ─▶ /plan_decision (PlanDecision)
//!   /mission_plan   (MissionPlan)  ──┘        validate()
//!
//! The node holds the latest swarm estimate; each incoming plan is validated
//! against it and the config. Every plan produces a /plan_decision — the
//! accept/reject event is observable by the whole system. (Emitting setpoints
//! on accept is the deployment step; kept out of the node so the topic contract
//! stays pure and testable.)

use std::sync::{Arc, Mutex};

use rclrs::*;
use swarm_supervisor::{validate, Assignment, EstimateSnapshot, Plan, SupervisorConfig};

/// Shared latest estimate. The supervisor can only validate a plan against a
/// world state it currently believes; no estimate yet ⇒ reject on staleness.
type SharedEstimate = Arc<Mutex<Option<EstimateSnapshot>>>;

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn main() -> Result<(), RclrsError> {
    let context = Context::default_from_env()?;
    let mut executor = context.create_basic_executor();
    let node = executor.create_node("swarm_supervisor")?;

    let cfg = SupervisorConfig::default();
    let estimate: SharedEstimate = Arc::new(Mutex::new(None));

    // Cache the latest estimate as it streams in.
    let est_for_sub = Arc::clone(&estimate);
    let _estimate_sub = node.create_subscription::<swarm_msgs::msg::SwarmEstimate, _>(
        "swarm_estimate",
        move |msg: swarm_msgs::msg::SwarmEstimate| {
            let n = msg.pos_north.len();
            let pos: Vec<[f64; 2]> = (0..n)
                .map(|i| [msg.pos_north[i], msg.pos_east[i]])
                .collect();
            *est_for_sub.lock().unwrap() = Some(EstimateSnapshot {
                frame_index: msg.frame_index,
                stamp_unix_ms: now_unix_ms(),
                pos,
                cov_trace: msg.cov_trace,
            });
        },
    )?;

    // Publish a verdict for every plan.
    let decision_pub = node.create_publisher::<swarm_msgs::msg::PlanDecision>("plan_decision")?;

    let est_for_plan = Arc::clone(&estimate);
    let logger = node.logger().clone();
    let _plan_sub = node.create_subscription::<swarm_msgs::msg::MissionPlan, _>(
        "mission_plan",
        move |msg: swarm_msgs::msg::MissionPlan| {
            let plan = Plan {
                plan_id: msg.plan_id.clone(),
                issued_unix_ms: msg.issued_unix_ms,
                min_spacing_m: Some(msg.min_spacing_m),
                assignments: msg
                    .assignments
                    .iter()
                    .map(|a| Assignment {
                        vehicle: a.vehicle,
                        waypoint_ne: [a.waypoint_north, a.waypoint_east],
                    })
                    .collect(),
            };

            let mut out = swarm_msgs::msg::PlanDecision::default();
            out.plan_id = plan.plan_id.clone();

            match est_for_plan.lock().unwrap().as_ref() {
                None => {
                    // No estimate yet: nothing to trust, so nothing flies.
                    out.accepted = false;
                    out.violations = vec!["no swarm estimate received yet".into()];
                }
                Some(est) => {
                    let decision = validate(&plan, est, &cfg, now_unix_ms());
                    out.accepted = decision.accepted;
                    out.violations = decision
                        .violations
                        .iter()
                        .map(|v| format!("{v:?}"))
                        .collect();
                }
            }

            if out.accepted {
                log_info!(&logger, "plan '{}' ACCEPTED", out.plan_id);
            } else {
                log_warn!(
                    &logger,
                    "plan '{}' REJECTED ({} violations) — no setpoints emitted",
                    out.plan_id,
                    out.violations.len()
                );
            }
            let _ = decision_pub.publish(out);
        },
    )?;

    log_info!(
        node.logger(),
        "swarm_supervisor up: /mission_plan + /swarm_estimate -> /plan_decision"
    );
    executor.spin(SpinOptions::default()).first_error()
}
