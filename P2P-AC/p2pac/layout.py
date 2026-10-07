"""Default arena layout of Sec. VI-A (kept free of third-party imports so that the configuration and the sweep
generator run with a plain Python 3 on an HPC login node)."""

# Cylindrical obstacles: (x, y, radius) in metres.
DEFAULT_OBSTACLES: tuple = (
    (0.00, 0.15, 0.23),   # entry
    (-1.30, 0.75, 0.23),
    (0.00, 1.45, 0.23),   # centre
    (1.30, 0.75, 0.23),
    (1.30, -0.45, 0.23),
    (-1.30, -0.45, 0.23),
)
