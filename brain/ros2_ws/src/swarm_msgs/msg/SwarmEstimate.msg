# The estimator's output for the whole swarm at one frame: pose + trust.
# This is the Layer-2 iSAM2 result, published as a topic. The per-vehicle
# marginal covariance trace IS the trust signal the supervisor gates on.
uint64 frame_index
float64[] pos_north      # length n
float64[] pos_east       # length n
float64[] cov_trace      # length n: marginal covariance trace per vehicle
