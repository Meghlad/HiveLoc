import gtsam, numpy as np
from gtsam.symbol_shorthand import X
g = gtsam.NonlinearFactorGraph()
g.add(gtsam.PriorFactorPoint2(X(0), np.array([1.0, 2.0]), gtsam.noiseModel.Isotropic.Sigma(2, 0.1)))
print("gtsam OK")