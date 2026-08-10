# A single vision bearing: observer i saw a neighbor along a world-frame bearing.
# Identity (which neighbor) is deliberately NOT included - data association is
# the estimator's job, which has the predicted swarm state to match against.
uint32 observer          # which drone's camera produced this
float64 bearing_world    # radians, world frame (observer heading + pixel bearing)
float64 u                # detection pixel column (for audit)
float64 v                # detection pixel row
float32 confidence       # detector response strength
