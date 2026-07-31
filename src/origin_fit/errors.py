class OriginFitError(Exception):
    """A safe, user-facing Origin Integration Application error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
