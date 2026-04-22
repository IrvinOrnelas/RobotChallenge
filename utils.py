"""
utils.py — Shared math utilities for RobotChallenge simulator
"""
import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def norm2(x: float, y: float) -> float:
    return float(np.sqrt(x**2 + y**2))


def deg2rad(d: float) -> float:
    return d * np.pi / 180.0


def rad2deg(r: float) -> float:
    return r * 180.0 / np.pi
