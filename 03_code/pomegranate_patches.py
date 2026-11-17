"""
Workaround for pomegranate 1.1.2: IndependentComponents.log_probability()
slices the full input tensor (same dtype for every column, e.g. float32 for
z-scored continuous vars) and hands each slice to its component distribution.
Categorical.log_probability() then indexes with that slice directly
(`self._log_probs[i][X[:, i]]`), which requires a long/int tensor -- so a
float 0.0/1.0 binary column raises IndexError inside a mixed Normal+Categorical
IndependentComponents, even though Categorical.summarize() (used during
fitting/EM) has no such issue since it only does `X == i` comparisons.

CategoricalMixed casts to long right before indexing. Use it in place of
Categorical for every binary/categorical dimension inside an
IndependentComponents used for mixed-type HMM emissions.
"""
from pomegranate.distributions import Categorical


class CategoricalMixed(Categorical):
    def log_probability(self, X):
        return super().log_probability(X.long())
