"""Usage counters accumulated during an agent run."""

from dataclasses import dataclass


@dataclass(slots=True)
class RunUsage:
    """Minimal usage information for the first learning stage."""

    requests: int = 0

    def copy(self) -> "RunUsage":
        """Return an independent snapshot for the public result."""
        return RunUsage(requests=self.requests)
