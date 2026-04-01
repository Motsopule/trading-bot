class ExposureTracker:

    def strategy_exposure(self, portfolio, strategy_name):
        return sum(
            p.risk for p in portfolio.open_positions
            if p.strategy_name == strategy_name
        )
