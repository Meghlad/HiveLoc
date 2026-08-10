# A structured plan from the language-conditioned planner (Layer 3).
# The supervisor node subscribes to this; free text never appears here because
# the plan was constrained-decoded into this shape upstream.
string plan_id
uint64 issued_unix_ms
float64 min_spacing_m
swarm_msgs/Assignment[] assignments
