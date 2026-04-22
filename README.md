# RobotChallenge — Almacén Robótico Colaborativo
**TE3002B · Robots Móviles · Tecnológico de Monterrey**

## Quick Start

```bash
pip install -r requirements.txt
python sim.py                  # live animation at 1× speed
python sim.py --speed 5        # 5× real-time
python sim.py --no-lidar       # hide LiDAR rays
python sim.py --save           # export robot_sim.mp4 (requires ffmpeg)
```

## Project Structure

```
robot_sim/
├── sim.py                      # 2D matplotlib simulator (entry point)
├── coordinator.py              # 3-phase mission state machine
├── utils.py                    # shared math helpers
├── requirements.txt
└── classes/
    ├── husky/
    │   ├── husky.py            # Husky skid-steer model + entity
    │   └── lidar.py            # 2D LiDAR ray-casting sensor
    ├── anymal/
    │   └── anymal.py           # ANYmal leg FK/IK/Jacobian + trot gait
    └── puzzlebot/
        ├── puzzlebot.py        # PuzzleBot diff-drive base
        └── puzzlebot_arm.py    # 3-DOF arm FK/IK/Jacobian/τ=Jᵀf
```

## Architecture

```
Layer 1 — Kinematics Models (pure math, no state)
  HuskyModel         fk / ik / compute_velocity_command / integrate
  PuzzleBotArmModel  fk / ik / jacobian / force_to_torque / trajectory
  AnymalLegModel     fk / ik / jacobian / singularity check

Layer 2 — Robot Entities (state + model)
  Husky              pose, trail, v_cmd/v_meas, step(dt, wr, wl)
  PuzzleBot          pose, arm joints, FSM state, navigate_to()
  Anymal             pose, 4 legs, gait phase, navigate_to()

Layer 3 — Coordinator (mission logic)
  Phase.CLEARING     Husky FSM: APPROACH → PUSH → RETREAT (×3 boxes)
  Phase.TRANSPORTING ANYmal walks to p_dest=(11.0, 3.6)
  Phase.STACKING     3 PuzzleBots stack boxes C→B→A

Layer 4 — Sim2D (renderer)
  FuncAnimation      50ms frame interval
  draw2D             trails, LiDAR, boxes, robots, HUD, telemetry panel
```

## Mission Phases

| Phase | Robot | Task |
|-------|-------|------|
| CLEARING | Husky | Push boxes A, B, C out of the 6×2m corridor |
| TRANSPORTING | ANYmal | Walk from origin to (11.0, 3.6), carrying 3 PuzzleBots |
| STACKING | PuzzleBots 0-2 | Pick and stack boxes in order C→B→A |

## Key Equations

```
Husky FK:      v = (r/4)(wr1+wr2+wl1+wl2)·s,  ω = (r/2B)(r_avg − l_avg)
PuzzleBot IK:  q3 = atan2(−√(1−D²), D),  q2 = atan2(zr,r) − atan2(l3·s3, l2+l3·c3)
Arm Jacobian:  ẋ = J·q̇  →  J is 3×3 analytic
Force→Torque:  τ = Jᵀ·f
Singularity:   |det J| < 1e-3  →  warning printed
```

## Dependencies
- Python 3.8+
- numpy ≥ 1.21
- matplotlib ≥ 3.5
- ffmpeg (optional, only for `--save`)
