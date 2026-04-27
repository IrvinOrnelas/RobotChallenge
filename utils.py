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

def check_aabb_collision(boundsA, boundsB):
    """
    Returns True if bounding box A overlaps with bounding box B.
    Bounds format: (x_min, x_max, y_min, y_max)
    """
    a_xmin, a_xmax, a_ymin, a_ymax = boundsA
    b_xmin, b_xmax, b_ymin, b_ymax = boundsB
    
    # If one rectangle is on left side of other
    if a_xmax < b_xmin or a_xmin > b_xmax:
        return False
    # If one rectangle is above other
    if a_ymax < b_ymin or a_ymin > b_ymax:
        return False
    return True


def get_aabb_distance(boundsA, boundsB):
    """
    Returns the minimum distance between two AABBs (0 if overlapping).
    Bounds format: (x_min, x_max, y_min, y_max)
    """
    a_xmin, a_xmax, a_ymin, a_ymax = boundsA
    b_xmin, b_xmax, b_ymin, b_ymax = boundsB
    
    dx = max(a_xmin - b_xmax, b_xmin - a_xmax, 0)
    dy = max(a_ymin - b_ymax, b_ymin - a_ymax, 0)
    
    return np.sqrt(dx**2 + dy**2)


def detect_collision_groups(mover_bounds, objects, contact_margin=0.05):
    """
    Detects collision chains: objects that are in contact with the moving object
    or transitively in contact with other colliding objects.
    
    Returns a list of objects that should move as a group.
    """
    colliding = set()
    to_check = []
    
    # Find all objects directly touching the mover
    for obj in objects:
        dist = get_aabb_distance(mover_bounds, obj.get_bounds())
        if dist <= contact_margin:
            colliding.add(id(obj))
            to_check.append(obj)
    
    # Propagate through chains: find objects touching colliding objects
    checked = set()
    while to_check:
        current = to_check.pop(0)
        obj_id = id(current)
        
        if obj_id in checked:
            continue
        checked.add(obj_id)
        
        for other in objects:
            other_id = id(other)
            if other_id not in colliding and other_id not in checked:
                dist = get_aabb_distance(current.get_bounds(), other.get_bounds())
                if dist <= contact_margin:
                    colliding.add(other_id)
                    to_check.append(other)
    
    # Return actual objects (not just IDs)
    return [obj for obj in objects if id(obj) in colliding]


def propagate_push_force(mover_bounds, mover_velocity, objects, dt, contact_margin=0.05):
    """
    Applies push forces to all objects in contact with the mover, including chains.
    Returns updated object positions.
    
    Args:
        mover_bounds: (x_min, x_max, y_min, y_max) of the pushing object
        mover_velocity: (vx, vy) velocity of the pusher
        objects: list of movable objects with .x, .y, .w, .h, get_bounds()
        dt: time delta
        contact_margin: tolerance for contact detection
    """
    collision_group = detect_collision_groups(mover_bounds, objects, contact_margin)
    
    for obj in collision_group:
        # Push along mover's velocity direction
        obj.x += mover_velocity[0] * dt
        obj.y += mover_velocity[1] * dt
    
    return collision_group
