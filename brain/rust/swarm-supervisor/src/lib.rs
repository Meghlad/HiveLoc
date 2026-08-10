//! swarm-supervisor: the deterministic gate between a language model and an
//! aircraft.
//!
//! Layer 3's VLM emits structured plans - waypoints, spacing, per-vehicle
//! assignments. A model WILL eventually hallucinate a waypoint into a
//! hillside. The answer is architectural: the plan never reaches MAVLink,
//! because this crate is the only component that can emit a setpoint, and it
//! contains no model. Every plan is rejected unless ALL of:
//!
//!   1. every waypoint falls inside the geofence polygon
//!   2. commanded inter-vehicle spacing stays above the minimum
//!   3. the estimator's marginal covariance for every ASSIGNED vehicle is
//!      below threshold  (Layer 2 is what makes this signal trustworthy: a
//!      floppy range graph makes covariance a liar; the bearing factors made
//!      it honest)
//!   4. both the plan and the estimate are fresh
//!
//! Rejection is total: one violation and NOTHING moves. There is no partial
//! acceptance, no "best effort", no model-in-the-loop retry. The supervisor
//! reports *why* so the operator (or the VLM, next round) can fix the plan.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

/// One vehicle's commanded destination, NED-north/east in estimator units.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Assignment {
    pub vehicle: u32,
    pub waypoint_ne: [f64; 2],
}

/// The full structured plan the VLM must produce. Anything that doesn't parse
/// into this shape is rejected before validation even starts - free text
/// cannot reach the aircraft because free text cannot construct this struct.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Plan {
    pub plan_id: String,
    /// when the plan was issued (unix ms) - stale plans are dead plans
    pub issued_unix_ms: u64,
    pub assignments: Vec<Assignment>,
    /// commanded minimum inter-vehicle spacing for this plan (m)
    #[serde(default)]
    pub min_spacing_m: Option<f64>,
}

/// What the estimator publishes (Layer 2's output, incl. the trust signal).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EstimateSnapshot {
    pub frame_index: u64,
    pub stamp_unix_ms: u64,
    pub pos: Vec<[f64; 2]>,
    /// per-vehicle marginal covariance trace from iSAM2
    pub cov_trace: Vec<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SupervisorConfig {
    /// geofence polygon (closed implicitly), estimator units
    pub geofence: Vec<[f64; 2]>,
    /// hard floor on inter-vehicle spacing (m, estimator units)
    pub min_spacing_m: f64,
    /// max acceptable marginal covariance trace for an assigned vehicle
    pub max_cov_trace: f64,
    /// plans older than this are rejected (ms)
    pub max_plan_age_ms: u64,
    /// estimates older than this are rejected (ms)
    pub max_estimate_age_ms: u64,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        SupervisorConfig {
            geofence: vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            min_spacing_m: 0.08,
            max_cov_trace: 0.004, // ~ sigma 4.5 cm/axis - Layer 2 numbers
            max_plan_age_ms: 5_000,
            max_estimate_age_ms: 1_000,
        }
    }
}

/// Every way a plan can die, with enough detail to fix it.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum Violation {
    PlanStale { age_ms: u64 },
    EstimateStale { age_ms: u64 },
    NoAssignments,
    UnknownVehicle { vehicle: u32 },
    DuplicateAssignment { vehicle: u32 },
    WaypointOutsideGeofence { vehicle: u32, waypoint_ne: [f64; 2] },
    SpacingTooClose { a: u32, b: u32, dist: f64, min: f64 },
    CovarianceTooHigh { vehicle: u32, trace: f64, max: f64 },
}

#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub plan_id: String,
    pub accepted: bool,
    pub violations: Vec<Violation>,
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/// Ray-casting point-in-polygon. Boundary counts as OUTSIDE: a waypoint ON the
/// fence is a waypoint the wind puts over it.
fn inside_polygon(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false; // a degenerate fence admits nothing
    }
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        // on-edge check: reject explicitly
        let cross = (xj - xi) * (p[1] - yi) - (yj - yi) * (p[0] - xi);
        if cross.abs() < 1e-12
            && p[0] >= xi.min(xj) - 1e-12
            && p[0] <= xi.max(xj) + 1e-12
            && p[1] >= yi.min(yj) - 1e-12
            && p[1] <= yi.max(yj) + 1e-12
        {
            return false;
        }
        if (yi > p[1]) != (yj > p[1]) {
            let x_at = (xj - xi) * (p[1] - yi) / (yj - yi) + xi;
            if p[0] < x_at {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// Validate a plan against the current estimate. Pure function: same inputs,
/// same verdict, testable without an aircraft, an estimator, or a model.
pub fn validate(
    plan: &Plan,
    est: &EstimateSnapshot,
    cfg: &SupervisorConfig,
    now_unix_ms: u64,
) -> Decision {
    let mut v = Vec::new();

    // 4. freshness first: stale inputs make every other check meaningless
    let plan_age = now_unix_ms.saturating_sub(plan.issued_unix_ms);
    if plan_age > cfg.max_plan_age_ms {
        v.push(Violation::PlanStale { age_ms: plan_age });
    }
    let est_age = now_unix_ms.saturating_sub(est.stamp_unix_ms);
    if est_age > cfg.max_estimate_age_ms {
        v.push(Violation::EstimateStale { age_ms: est_age });
    }

    if plan.assignments.is_empty() {
        v.push(Violation::NoAssignments);
    }

    // structural sanity: every assigned vehicle must exist, exactly once
    let n = est.pos.len() as u32;
    let mut seen = std::collections::HashSet::new();
    for a in &plan.assignments {
        if a.vehicle >= n {
            v.push(Violation::UnknownVehicle { vehicle: a.vehicle });
        }
        if !seen.insert(a.vehicle) {
            v.push(Violation::DuplicateAssignment { vehicle: a.vehicle });
        }
    }

    // 1. geofence: every waypoint strictly inside
    for a in &plan.assignments {
        if !inside_polygon(a.waypoint_ne, &cfg.geofence) {
            v.push(Violation::WaypointOutsideGeofence {
                vehicle: a.vehicle,
                waypoint_ne: a.waypoint_ne,
            });
        }
    }

    // 2. spacing: the commanded FORMATION must keep vehicles apart. The plan
    // may ask for wider spacing than the floor, never narrower.
    let min_spacing = plan
        .min_spacing_m
        .unwrap_or(cfg.min_spacing_m)
        .max(cfg.min_spacing_m);
    for (i, a) in plan.assignments.iter().enumerate() {
        for b in plan.assignments.iter().skip(i + 1) {
            let d = ((a.waypoint_ne[0] - b.waypoint_ne[0]).powi(2)
                + (a.waypoint_ne[1] - b.waypoint_ne[1]).powi(2))
            .sqrt();
            if d < min_spacing {
                v.push(Violation::SpacingTooClose {
                    a: a.vehicle,
                    b: b.vehicle,
                    dist: d,
                    min: min_spacing,
                });
            }
        }
    }

    // 3. trust: an assigned vehicle whose estimate is uncertain must not be
    // commanded - moving a drone you can't localize is how formations collide.
    for a in &plan.assignments {
        if let Some(&trace) = est.cov_trace.get(a.vehicle as usize) {
            if trace > cfg.max_cov_trace {
                v.push(Violation::CovarianceTooHigh {
                    vehicle: a.vehicle,
                    trace,
                    max: cfg.max_cov_trace,
                });
            }
        }
    }

    Decision {
        plan_id: plan.plan_id.clone(),
        accepted: v.is_empty(),
        violations: v,
    }
}

// ---------------------------------------------------------------------------
// Tests: every gate earns its keep, including the hillside
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn est() -> EstimateSnapshot {
        EstimateSnapshot {
            frame_index: 100,
            stamp_unix_ms: 1_000_000,
            pos: (0..12).map(|i| [0.1 + 0.06 * i as f64, 0.5]).collect(),
            cov_trace: vec![0.001; 12],
        }
    }

    fn plan(assignments: Vec<Assignment>) -> Plan {
        Plan {
            plan_id: "p1".into(),
            issued_unix_ms: 1_000_000,
            assignments,
            min_spacing_m: None,
        }
    }

    fn cfg() -> SupervisorConfig {
        SupervisorConfig::default()
    }

    const NOW: u64 = 1_000_100;

    #[test]
    fn valid_plan_passes() {
        let p = plan(vec![
            Assignment { vehicle: 0, waypoint_ne: [0.2, 0.2] },
            Assignment { vehicle: 1, waypoint_ne: [0.8, 0.8] },
        ]);
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(d.accepted, "{:?}", d.violations);
    }

    #[test]
    fn hillside_waypoint_rejected() {
        // the canonical failure: the model hallucinates a point outside the fence
        let p = plan(vec![Assignment { vehicle: 0, waypoint_ne: [1.7, 0.4] }]);
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(!d.accepted);
        assert!(matches!(
            d.violations[0],
            Violation::WaypointOutsideGeofence { vehicle: 0, .. }
        ));
    }

    #[test]
    fn fence_boundary_is_outside() {
        let p = plan(vec![Assignment { vehicle: 0, waypoint_ne: [1.0, 0.5] }]);
        assert!(!validate(&p, &est(), &cfg(), NOW).accepted);
    }

    #[test]
    fn collision_spacing_rejected() {
        let p = plan(vec![
            Assignment { vehicle: 0, waypoint_ne: [0.50, 0.50] },
            Assignment { vehicle: 1, waypoint_ne: [0.52, 0.50] },
        ]);
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(d.violations.iter().any(|v| matches!(v, Violation::SpacingTooClose { .. })));
    }

    #[test]
    fn plan_cannot_shrink_the_spacing_floor() {
        let mut p = plan(vec![
            Assignment { vehicle: 0, waypoint_ne: [0.50, 0.50] },
            Assignment { vehicle: 1, waypoint_ne: [0.55, 0.50] },
        ]);
        p.min_spacing_m = Some(0.001); // the model asks nicely; the floor holds
        // waypoints are 0.05 apart < cfg floor 0.08 -> rejected even though
        // the plan requested a 0.001 minimum
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(!d.accepted);
        assert!(d.violations.iter().any(|v| matches!(v, Violation::SpacingTooClose { .. })));
    }

    #[test]
    fn uncertain_vehicle_not_commanded() {
        let mut e = est();
        e.cov_trace[3] = 0.5; // estimator says: I have no idea where drone 3 is
        let p = plan(vec![Assignment { vehicle: 3, waypoint_ne: [0.5, 0.5] }]);
        let d = validate(&p, &e, &cfg(), NOW);
        assert!(d.violations.iter().any(|v| matches!(
            v,
            Violation::CovarianceTooHigh { vehicle: 3, .. }
        )));
    }

    #[test]
    fn stale_plan_rejected() {
        let p = plan(vec![Assignment { vehicle: 0, waypoint_ne: [0.5, 0.5] }]);
        let d = validate(&p, &est(), &cfg(), NOW + 60_000);
        assert!(d.violations.iter().any(|v| matches!(v, Violation::PlanStale { .. })));
    }

    #[test]
    fn stale_estimate_rejected() {
        let mut e = est();
        e.stamp_unix_ms = 1;
        let p = plan(vec![Assignment { vehicle: 0, waypoint_ne: [0.5, 0.5] }]);
        let d = validate(&p, &e, &cfg(), NOW);
        assert!(d.violations.iter().any(|v| matches!(v, Violation::EstimateStale { .. })));
    }

    #[test]
    fn unknown_and_duplicate_vehicles_rejected() {
        let p = plan(vec![
            Assignment { vehicle: 40, waypoint_ne: [0.5, 0.5] },
            Assignment { vehicle: 1, waypoint_ne: [0.3, 0.3] },
            Assignment { vehicle: 1, waypoint_ne: [0.7, 0.7] },
        ]);
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(d.violations.iter().any(|v| matches!(v, Violation::UnknownVehicle { .. })));
        assert!(d.violations.iter().any(|v| matches!(v, Violation::DuplicateAssignment { .. })));
    }

    #[test]
    fn empty_plan_rejected() {
        let d = validate(&plan(vec![]), &est(), &cfg(), NOW);
        assert!(d.violations.contains(&Violation::NoAssignments));
    }

    #[test]
    fn rejection_is_total() {
        // one bad assignment poisons the whole plan - no partial acceptance
        let p = plan(vec![
            Assignment { vehicle: 0, waypoint_ne: [0.2, 0.2] },  // fine
            Assignment { vehicle: 1, waypoint_ne: [9.9, 9.9] },  // hillside
        ]);
        let d = validate(&p, &est(), &cfg(), NOW);
        assert!(!d.accepted);
    }
}
