import numpy as np

from intelligence.config import CORRELATION_THRESHOLD, CORRELATION_REDUCTION


class CorrelationEngine:

    def factor(self, returns_matrix, symbol_idx):

        if returns_matrix is None:
            return 1.0

        rm = np.asarray(returns_matrix, dtype=float)
        if rm.ndim < 2 or len(rm) == 0:
            return 1.0

        # Skip np.corrcoef for 0–2 return series (insufficient cross-section)
        if len(rm) <= 2:
            return 1.0

        corr_matrix = np.corrcoef(rm)
        if not np.isfinite(corr_matrix).all():
            return 1.0

        avg_corr = float(np.mean(corr_matrix[symbol_idx]))

        if avg_corr > CORRELATION_THRESHOLD:
            return CORRELATION_REDUCTION

        return 1.0
