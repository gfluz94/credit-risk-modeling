class NotFittedYet(Exception):
    """Exception raised when model has not been fitted yet."""


class ZerosNotPresentForZeroInflatedRegression(Exception):
    """Exception raised when there are no zeros, hence regular regression should be used in this case."""
