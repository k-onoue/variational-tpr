"""
This module contains common constants used throughout the stp package.
"""

# A small value to clamp inputs to log to avoid log(0).
EPSILON = 1e-12

# A small value added to the diagonal of kernels for numerical stability.
JITTER = 1e-6
