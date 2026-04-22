"""
coordinator.py
Multi-robot task coordinator — correct mission logic per TE3002B spec.

CORRECT MISSION FLOW:
─────────────────────────────────────────────────────────────────────────────
PHASE 1 — CLEARING (Husky only)
  · Husky detects 3 large obstacle boxes (B1,B2,B3) in corridor via LiDAR
  · Husky pushes EACH box out of the 6×2 m corridor, one at a time
  · PuzzleBots are RIDING on ANYmal's back — they do NOT move yet
  · ANYmal waits at the start zone
  · Success: all 3 obstacle boxes are outside corridor rectangle

PHASE 2 — TRANSPORTING (ANYmal carrying PuzzleBots)
  · ANYmal walks through the now-clear corridor
  · 3 PuzzleBots are ON ANYmal's back (passive cargo)
  · ANYmal navigates to pdest = (11.0, 3.6) with error < 0.15 m
  · Singularity monitoring on every leg every step (|det J| > 1e-3)
  · Success: ANYmal arrives at work zone

PHASE 3 — DEPLOYING
  · ANYmal is stationary at pdest
  · PuzzleBots are placed one by one at deploy positions (xArm action)

PHASE 4 — STACKING (PuzzleBots, ONE AT A TIME — time-slotting)
  · Stack order MANDATORY: C first (bottom), then B (middle), then A (top)
  · PuzzleBot 0 picks box C → carries it → places it at stack target
  · ONLY AFTER C is placed: PuzzleBot 1 picks box B → stacks on C
  · ONLY AFTER B is placed: PuzzleBot 2 picks box A → stacks on B
  · Each pick/place uses arm FK/IK + τ = Jᵀf force control
  · Success: A-B-C stacked in correct order, stable
─────────────────────────────────────────────────────────────────────────────
"""
from enum import Enum, auto
import numpy as np
from utils import norm2, wrap_angle, clamp


# ── ENUMS ────────────────────────────────────────────────────────────────────

class Phase(Enum):
    STARTING     = auto()
    CLEARING     = auto()
    TRANSPORTING = auto()
    DEPLOYING    = auto()
    STACKING     = auto()
    DONE         = auto()
    ERROR        = auto()


class HuskyFSM(Enum):
    APPROACH = auto()
    ALIGN    = auto()
    PUSH     = auto()
    RETREAT  = auto()
    IDLE     = auto()


class PuzzleBotFSM(Enum):
    RIDING   = auto()   # on ANYmal's back
    DEPLOYED = auto()   # placed on ground, waiting turn
    APPROACH = auto()   # driving toward target box
    GRASPING = auto()   # arm moving to grasp pose
    CARRYING = auto()   # driving to stack while holding box
    PLACING  = auto()   # arm lowering box onto stack
    DONE     = auto()


# ── BOX ──────────────────────────────────────────────────────────────────────

class Box:
    """A box in the world. obstacle_box=True → Husky must push it out."""
    def __init__(self, box_id, x, y, w=0.4, h=0.4,
                 color='red', obstacle_box=False):
        self.id           = box_id
        self.x            = x
        self.y            = y
        self.w            = w
        self.h            = h
        self.color        = color
        self.obstacle_box = obstacle_box
        self.stacked      = False
        self.stack_layer  = -1
        self.carried_by   = None

    def __repr__(self):
        return (f"Box(id={self.id}, pos=({self.x:.2f},{self.y:.2f}), "
                f"obstacle={self.obstacle_box}, stacked={self.stacked})")


# ── COORDINATOR ───────────────────────────────────────────────────────────────

class Coordinator:
    """
    Orchestrates Husky, ANYmal and 3 PuzzleBots through the full mission.

    Key invariants:
    · PuzzleBots stay on ANYmal during CLEARING and TRANSPORTING phases.
    · Stacking is strictly sequential C→B→A (time-slotting, no parallelism).
    · ANYmal monitors |det J| every step and counts violations.
    """

    # Where each PuzzleBot is placed after ANYmal arrives (relative to pdest)
    DEPLOY_OFFSETS = [
        np.array([ 0.0,  0.5]),
        np.array([ 0.0,  0.0]),
        np.array([ 0.0, -0.5]),
    ]

    # Riding offsets on ANYmal's back (body frame x-forward, y-left)
    RIDING_OFFSETS = [
        np.array([ 0.15,  0.12]),
        np.array([ 0.00,  0.00]),
        np.array([-0.15, -0.12]),
    ]

    def __init__(self, husky, anymal, puzzlebots: list,
                 obstacle_boxes: list, stack_boxes: list,
                 anymal_dest=(11.0, 3.6),
                 stack_target=(12.0, 3.0)):
        
        self.husky      = husky
        self.anymal     = anymal
        self.puzzlebots = puzzlebots

        self.anymal_dest  = np.array(anymal_dest,  dtype=float)
        self.stack_target = np.array(stack_target, dtype=float)

        # Store separate lists for logic
        self.obstacle_boxes = obstacle_boxes
        self.stack_boxes = stack_boxes
        
        # Combined list for the sim.py renderer
        self.boxes = self.obstacle_boxes + self.stack_boxes

        # ── Phase ────────────────────────────────────────────────────────────
        self.phase = Phase.STARTING

        # ── Husky state ──────────────────────────────────────────────────────
        self.husky_fsm     = HuskyFSM.APPROACH
        self.husky_box_idx = 0
        self.husky_origins = [(b.x, b.y) for b in self.obstacle_boxes]

        # ── ANYmal state ─────────────────────────────────────────────────────
        self.det_J_violations = 0

        # ── PuzzleBot state ───────────────────────────────────────────────────
        # All bots start RIDING on ANYmal
        self.pb_fsm          = [PuzzleBotFSM.RIDING] * 3
        self.deploy_idx      = 0
        self.active_stacker  = 0       # time-slot index (0→1→2)
        self.grasp_timer     = [0.0] * 3
        # Assignment: PB0→C (bottom), PB1→B (middle), PB2→A (top)
        self.pb_assignment   = {0: 'C', 1: 'B', 2: 'A'}
        self.stack_layer_h   = 0.35    # height added per stacked box

        self.t   = 0.0
        self.log = []

    # ── LOGGING ──────────────────────────────────────────────────────────────

    def _log(self, msg):
        entry = f"[{self.t:7.2f}s] {msg}"
        self.log.append(entry)
        print(entry)

    # ── MAIN STEP ────────────────────────────────────────────────────────────

    def step(self, dt):
        self.t += dt
        if   self.phase == Phase.STARTING:     self._do_starting()
        elif self.phase == Phase.CLEARING:     self._do_clearing(dt)
        elif self.phase == Phase.TRANSPORTING: self._do_transporting(dt)
        elif self.phase == Phase.DEPLOYING:    self._do_deploying(dt)
        elif self.phase == Phase.STACKING:     self._do_stacking(dt)

    def task_completed(self):
        return self.phase == Phase.DONE

    # ── PHASE: STARTING ──────────────────────────────────────────────────────

    def _do_starting(self):
        """Put all PuzzleBots on ANYmal and start clearing."""
        for i in range(3):
            self._snap_bot_to_anymal(i)
            self.pb_fsm[i] = PuzzleBotFSM.RIDING
        self._log("═══ PHASE 1: CLEARING — Husky pushes obstacle boxes out of corridor ═══")
        self.phase = Phase.CLEARING

    # ── PHASE: CLEARING ──────────────────────────────────────────────────────

    def _do_clearing(self, dt):
        """
        Husky clears corridor of B1, B2, B3 one by one.
        ANYmal stays still. PuzzleBots stay on ANYmal's back.
        """
        # Keep bots riding on stationary ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)

        if self.husky_box_idx >= len(self.obstacle_boxes):
            self._log("✅ CLEARING done — corridor clear")
            self._log("═══ PHASE 2: TRANSPORTING — ANYmal walks to work zone ═══")
            self.phase = Phase.TRANSPORTING
            return

        box    = self.obstacle_boxes[self.husky_box_idx]
        origin = self.husky_origins[self.husky_box_idx]

        if self.husky_fsm == HuskyFSM.APPROACH:
            # Drive to a point behind the box (push in +X direction)
            target = np.array([box.x - 0.65, box.y])
            dist = norm2(target[0] - self.husky.x, target[1] - self.husky.y)
            v, w = self.husky.model.compute_velocity_command(self.husky.pose, target)
            wr, wl = self.husky.model.inverse_kinematics(v, w)
            self.husky.step(dt, wr, wl)
            self.husky.v_cmd, self.husky.w_cmd = v, w
            if dist < 0.25:
                self.husky_fsm = HuskyFSM.ALIGN
                self._log(f"  Husky behind {box.id} — aligning")

        elif self.husky_fsm == HuskyFSM.ALIGN:
            ang_err = wrap_angle(0.0 - self.husky.pose[2])
            w = clamp(2.0 * ang_err, -1.5, 1.5)
            wr, wl = self.husky.model.inverse_kinematics(0.0, w)
            self.husky.step(dt, wr, wl)
            self.husky.v_cmd, self.husky.w_cmd = 0.0, w
            if abs(ang_err) < 0.08:
                self.husky_fsm = HuskyFSM.PUSH
                self._log(f"  Husky pushing {box.id}")

        elif self.husky_fsm == HuskyFSM.PUSH:
            # Drive forward; box is glued in front of Husky
            v, w = self.husky.model.compute_velocity_command(
                self.husky.pose, [box.x + 4.0, box.y])
            wr, wl = self.husky.model.inverse_kinematics(clamp(v, 0, 0.5), w * 0.2)
            self.husky.step(dt, wr, wl)
            self.husky.v_cmd, self.husky.w_cmd = v, 0.0
            # Box rides in front of Husky
            box.x = self.husky.x + 0.46
            box.y = self.husky.y
            displaced = norm2(box.x - origin[0], box.y - origin[1])
            if displaced > 1.8 or box.x > 6.3:
                self._log(f"  ✓ {box.id} cleared (displaced {displaced:.2f} m)")
                self.husky_fsm = HuskyFSM.RETREAT

        elif self.husky_fsm == HuskyFSM.RETREAT:
            # Return to a fixed staging point near start, ready for next box
            next_box_idx = min(self.husky_box_idx + 1, len(self.obstacle_boxes) - 1)
            origin_next  = self.husky_origins[next_box_idx]
            retreat_goal = np.array([origin_next[0] - 2.0, 0.0])
            dist = norm2(retreat_goal[0] - self.husky.x,
                         retreat_goal[1] - self.husky.y)
            v, w = self.husky.model.compute_velocity_command(
                self.husky.pose, retreat_goal)
            wr, wl = self.husky.model.inverse_kinematics(v * 0.6, w)
            self.husky.step(dt, wr, wl)
            self.husky.v_cmd, self.husky.w_cmd = v * 0.6, w
            if dist < 0.4:
                self.husky_box_idx += 1
                self.husky_fsm = HuskyFSM.APPROACH
                if self.husky_box_idx < len(self.obstacle_boxes):
                    nxt = self.obstacle_boxes[self.husky_box_idx].id
                    self._log(f"  Husky retreated — next target: {nxt}")

    # ── PHASE: TRANSPORTING ───────────────────────────────────────────────────

    def _do_transporting(self, dt):
        """
        ANYmal walks to pdest carrying PuzzleBots on its back.
        PuzzleBots move rigidly with ANYmal.
        """
        # PuzzleBots ride on ANYmal
        for i in range(3):
            self._snap_bot_to_anymal(i)

        # Singularity monitoring every step
        for name, leg in self.anymal.legs.items():
            d = abs(leg.det_J())
            if d < 1e-3:
                self.det_J_violations += 1

        arrived = self.anymal.navigate_to(self.anymal_dest, dt)

        if arrived:
            err = norm2(self.anymal_dest[0] - self.anymal.x,
                        self.anymal_dest[1] - self.anymal.y)
            self._log(f"✅ ANYmal at pdest, err={err:.4f} m | "
                      f"det_J violations={self.det_J_violations}")
            self._log("═══ PHASE 3: DEPLOYING — placing PuzzleBots at work surface ═══")
            self.phase     = Phase.DEPLOYING
            self.deploy_idx = 0

    # ── PHASE: DEPLOYING ─────────────────────────────────────────────────────

    def _do_deploying(self, dt):
        """
        ANYmal is still. PuzzleBots are placed one by one at deploy positions.
        (In real life the xArm does this; here we snap them instantly.)
        """
        if self.deploy_idx >= 3:
            self._log("✅ All 3 PuzzleBots deployed on work surface")
            self._log("═══ PHASE 4: STACKING — C → B → A (time-slotting) ═══")
            self.active_stacker = 0
            self.phase = Phase.STACKING
            return

        i      = self.deploy_idx
        bot    = self.puzzlebots[i]
        offset = self.DEPLOY_OFFSETS[i]
        pos    = self.anymal_dest + offset

        bot.pose[0]  = pos[0]
        bot.pose[1]  = pos[1]
        bot.pose[2]  = 0.0
        self.pb_fsm[i] = PuzzleBotFSM.DEPLOYED

        self._log(f"  PuzzleBot {i} (→box {self.pb_assignment[i]}) deployed at "
                  f"({pos[0]:.2f}, {pos[1]:.2f})")
        self.deploy_idx += 1

    # ── PHASE: STACKING ──────────────────────────────────────────────────────

    def _do_stacking(self, dt):
        """
        TIME-SLOTTING: only one PuzzleBot active at a time.
        Order enforced: PB0 stacks C (layer 0), then PB1 stacks B (layer 1),
        then PB2 stacks A (layer 2).
        """
        if self.active_stacker >= 3:
            self._log("✅ STACKING complete — stack C→B→A is stable. MISSION DONE.")
            self.phase = Phase.DONE
            return

        i      = self.active_stacker
        bot    = self.puzzlebots[i]
        fsm    = self.pb_fsm[i]
        box_id = self.pb_assignment[i]
        box    = next(b for b in self.stack_boxes if b.id == box_id)
        layer  = i   # 0=C(bottom), 1=B(middle), 2=A(top)

        # ── Activate idle bot ────────────────────────────────────────────────
        if fsm == PuzzleBotFSM.DEPLOYED:
            self.pb_fsm[i] = PuzzleBotFSM.APPROACH
            self._log(f"  [Slot {i}] PuzzleBot {i} → picking box {box_id}")
            return

        # ── APPROACH: drive toward the box ───────────────────────────────────
        if fsm == PuzzleBotFSM.APPROACH:
            arrived = bot.navigate_to([box.x, box.y], dt)
            bot.step(dt)
            if arrived:
                self._log(f"  [Slot {i}] PuzzleBot {i} reached box {box_id}")
                self.pb_fsm[i]    = PuzzleBotFSM.GRASPING
                self.grasp_timer[i] = 0.0

        # ── GRASPING: arm descends and grips ─────────────────────────────────
        elif fsm == PuzzleBotFSM.GRASPING:
            bot.arm.set_ee_target(0.12, 0.0, 0.10)
            bot.step(dt)
            self.grasp_timer[i] += dt

            # Force control: τ = Jᵀ · f_grip
            f_grip = np.array([0.0, 0.0, -5.0])
            tau    = bot.arm.force_to_torque(f_grip)

            if self.grasp_timer[i] > 0.6:
                box.carried_by = i
                self._log(f"  [Slot {i}] Grasped {box_id} | τ={np.round(tau,3)}")
                self.pb_fsm[i]    = PuzzleBotFSM.CARRYING
                self.grasp_timer[i] = 0.0

        # ── CARRYING: navigate to stack zone ─────────────────────────────────
        elif fsm == PuzzleBotFSM.CARRYING:
            arrived = bot.navigate_to(self.stack_target, dt)
            bot.step(dt)
            # Box follows arm EE in world frame
            ee = bot.arm.ee_position()
            box.x = bot.x + ee[0]*np.cos(bot.theta) - ee[1]*np.sin(bot.theta)
            box.y = bot.y + ee[0]*np.sin(bot.theta) + ee[1]*np.cos(bot.theta)

            if arrived:
                self._log(f"  [Slot {i}] PuzzleBot {i} at stack zone — placing {box_id}")
                self.pb_fsm[i]    = PuzzleBotFSM.PLACING
                self.grasp_timer[i] = 0.0

        # ── PLACING: lower onto stack with force control ──────────────────────
        elif fsm == PuzzleBotFSM.PLACING:
            z_rest = layer * self.stack_layer_h + 0.04
            bot.arm.set_ee_target(0.12, 0.0, max(0.04, z_rest))
            bot.step(dt)
            self.grasp_timer[i] += dt

            f_place = np.array([0.0, 0.0, -2.0])
            tau     = bot.arm.force_to_torque(f_place)

            if self.grasp_timer[i] > 0.8:
                # Finalise box position
                box.x          = self.stack_target[0]
                box.y          = self.stack_target[1]
                box.stack_layer = layer
                box.stacked    = True
                box.carried_by = None
                bot.arm.home()
                self.pb_fsm[i] = PuzzleBotFSM.DONE
                self._log(f"  [Slot {i}] ✓ Box {box_id} placed at layer {layer} | "
                          f"τ={np.round(tau,3)}")
                # Unlock next time-slot
                self.active_stacker += 1
                if self.active_stacker < 3:
                    nxt = self.pb_assignment[self.active_stacker]
                    self._log(f"  ── Slot released → PuzzleBot {self.active_stacker} "
                              f"now active (box {nxt}) ──")

    # ── HELPER ───────────────────────────────────────────────────────────────

    def _snap_bot_to_anymal(self, i):
        """Rigidly attach PuzzleBot i to ANYmal's back."""
        off = self.RIDING_OFFSETS[i]
        th  = self.anymal.pose[2]
        self.puzzlebots[i].pose[0] = (self.anymal.x
                                      + off[0]*np.cos(th) - off[1]*np.sin(th))
        self.puzzlebots[i].pose[1] = (self.anymal.y
                                      + off[0]*np.sin(th) + off[1]*np.cos(th))
        self.puzzlebots[i].pose[2] = th

    # ── STATUS ────────────────────────────────────────────────────────────────

    def status(self):
        return {
            'phase'  : self.phase.name,
            't'      : round(self.t, 2),
            'husky'  : {'pos': self.husky.pose[:2].tolist(),
                        'v_cmd': round(self.husky.v_cmd, 4),
                        'v_meas': round(self.husky.v_meas, 4)},
            'anymal' : {'pos': self.anymal.pose[:2].tolist(),
                        'dist_goal': round(norm2(
                            self.anymal_dest[0]-self.anymal.x,
                            self.anymal_dest[1]-self.anymal.y), 4),
                        'det_J_violations': self.det_J_violations},
            'puzzlebots': [{'id': i, 'state': self.pb_fsm[i].name,
                            'pos': self.puzzlebots[i].pose[:2].tolist(),
                            'box': self.pb_assignment.get(i)}
                           for i in range(3)],
            'stack_boxes': [{'id': b.id, 'stacked': b.stacked,
                             'layer': b.stack_layer}
                            for b in self.stack_boxes],
        }