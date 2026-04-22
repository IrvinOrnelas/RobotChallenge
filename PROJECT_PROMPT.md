# RobotChallenge Project - Comprehensive Description

## Project Overview

This is a **Multi-Robot Coordination and Manipulation System** written in Python that simulates and controls multiple autonomous robots working together to solve complex tasks. The project focuses on robotics simulation, kinematics modeling, autonomous navigation, manipulation, and sensor integration.

### Key Robots in the System

#### 1. **PuzzleBot** - Mobile Manipulator Robot
- **Type**: Differential-drive mobile base with a 3-DOF robotic arm
- **Base Model**: 
  - Wheel radius: 0.05m
  - Wheel separation: 0.19m
  - Mass: 2kg
  - Maximum wheel speed: 0.8 rad/s
- **Arm Specifications**:
  - 3 revolute joints (q1, q2, q3)
  - Link lengths: l1=0.10m, l2=0.08m, l3=0.06m
  - Base height: l1
  - Full 360° rotation capability for q1
  - Joint 2 and 3: ±90° range
- **Capabilities**:
  - Differential kinematics for base navigation
  - Arm forward and inverse kinematics (3D Cartesian positioning)
  - Cartesian trajectory planning
  - End-effector force-torque calculations
  - Singularity detection
  - Joint velocity and acceleration limits

#### 2. **Husky** - Large Autonomous Platform
- **Type**: Four-wheel differential-drive robot (skid-steer configuration)
- **Base Model**:
  - Wheel radius: 0.1651m
  - Track width (distance between left/right wheels): 0.555m
  - Mass: 50kg
  - Maximum speed: 1 m/s (configurable)
- **Sensing**:
  - LiDAR sensor (360-degree, up to 8.0m range)
  - 360 rays with configurable noise (0.01m standard deviation)
  - Full 360° field of view
- **Kinematics**:
  - Four-wheel motor control (wr1, wr2, wl1, wl2)
  - Slip modeling with scaling factor 's'
  - Inverse kinematics with speed saturation
  - Velocity command computation using proportional control
  - Telemetry monitoring (commanded vs measured velocities)

#### 3. **Coordinator** - Multi-Robot State Machine
- **Purpose**: Orchestrates tasks between multiple robots (Husky, AnyMAL, PuzzleBots)
- **State Machine**:
  - STARTING: Initial state
  - CLEARING: Object/obstacle clearance phase
  - TRANSPORTING: Movement and relocation phase
  - STACKING: Object stacking/arrangement phase
  - DONE: Task completion
  - ERROR: Error handling state

---

## Architecture & Components

### Core Kinematics Models

#### PuzzleBot Arm (`PuzzleBotArmModel`)
```
Forward Kinematics:
- Takes joint angles (q1, q2, q3)
- Returns 3D end-effector position (x, y, z)
- q1 controls horizontal rotation
- q2, q3 control vertical reach and angle

Inverse Kinematics:
- Takes 3D target position (x, y, z)
- Returns required joint angles
- Uses geometric method with singularity checking
- Clips values to prevent numerical errors

Jacobian Matrix:
- 3×3 matrix relating joint velocities to end-effector velocity
- Used for force-torque calculations via: τ = J^T × f

Trajectory Planning:
- Linear interpolation in Cartesian space
- Inverse kinematics computed at each waypoint
- Automatic singularity detection and warnings
```

#### PuzzleBot Base (`PuzzleBotModel`)
```
Forward Kinematics:
- Input: wheel angular velocities (ωr, ωl)
- Output: linear velocity (v) and angular velocity (ω)
- v = (r/2) × (ωr + ωl)
- ω = (r/L) × (ωr - ωl)

Inverse Kinematics:
- Input: desired (v, ω)
- Output: required wheel speeds
- Clamping to max wheel speed (0.8 rad/s)

Integration:
- Euler method for pose updates
- Accounts for robot orientation during movement
- Wraps angles to [-π, π] range

Jacobian:
- Relates base velocities to arm manipulation
- Accounts for end-effector offset from base
```

#### Husky Base (`HuskyModel`)
```
Forward Kinematics:
- Input: four wheel speeds (wr1, wr2, wl1, wl2) + slip factor (s)
- Right wheels average: (wr1 + wr2) / 2
- Left wheels average: (wl1 + wl2) / 2
- v = (r/4) × (sum of all wheels) × s
- ω = (r/(2B)) × (right - left)

Inverse Kinematics:
- Computes required wheel speeds for target (v, ω)
- Automatic speed saturation (no wheel exceeds maxspeed)
- Redundant actuation: both right wheels get same speed, both left wheels get same speed

Integration (Mid-point Method):
- More accurate than Euler for rotation
- Evaluates rotation halfway through time step
- Slip modeling for terrain interaction

Velocity Commands:
- Proportional control to goal
- k_v: linear velocity gain (default 0.5)
- k_w: angular velocity gain (default 2.0)
```

---

## Simulation Modules

### `puzzle_sim.py` - PuzzleBot Visualization & Demonstration
- **Purpose**: Visualize PuzzleBot base and arm motion in 2D/3D
- **Features**:
  - Frame-by-frame animation building
  - Robot corner polygon computation (for collision detection visualization)
  - Arm joint position calculation in world coordinates
  - Color-coded visualization (links, joints, base)
  - Scenario demonstrations:
    - Forward straight-line motion
    - Arm homing (moving to predefined configuration)
    - Cartesian manipulation tasks
    - Combined base and arm motion

### `push_sim.py` - Husky Box-Pushing Task
- **Scenario**: Autonomous box pushing into a designated zone
- **Environment**:
  - Box: 0.5m × 0.5m square
  - Target zone: center at (4.0, 0.0), radius 1.8m
  - Start position: Husky at (0.5, -1.8)
  - Push direction: +X axis
- **Physics**:
  - Simple penetration resolution
  - Robot-box collision detection
  - Box pushing only when robot is behind and in contact
  - Robot bouncing when not in push mode
- **Autonomy Features**:
  - State machine: SCAN → APPROACH → PUSH → RETREAT
  - LiDAR-based box detection and centroid computation
  - Navigation with proportional control
  - Collision avoidance logic
- **Data Logging**:
  - Time series for velocities, positions, zone distance
  - Used for visualization and performance analysis

### `coordinator.py` - Multi-Robot Task Coordinator
- **Purpose**: Manage complex multi-robot workflows
- **Robots Managed**:
  - Husky (primary mover)
  - AnyMAL (alternative platform, not yet fully implemented)
  - Multiple PuzzleBots (manipulation units)
- **Task Decomposition**:
  - CLEARING: Remove obstacles
  - TRANSPORTING: Move objects using PuzzleBots and Husky
  - STACKING: Arrange objects in desired configuration
- **Coordination Strategies**:
  - State machine ensures sequential task phases
  - Synchronization between robots
  - Error detection and recovery

### `sim.py` - Base Simulation Framework
- Currently a placeholder for core simulation engine
- Intended to provide unified simulation interface

---

## Sensor Integration

### LiDAR Sensor (`classes/husky/lidar.py`)
- **Configuration**:
  - Number of rays: 360 (configurable)
  - Maximum range: 8.0m
  - Field of view: Full 360° (or configurable)
  - Noise model: Gaussian with std = 0.01m
- **Capabilities**:
  - Ray casting in 2D environment
  - Obstacle detection
  - Centroid extraction from point clouds
  - Object boundary detection
- **Usage**: Box detection and localization in `push_sim.py`

---

## Key Mathematical Concepts

### Kinematics
- **Forward kinematics**: Joint angles → End-effector position
- **Inverse kinematics**: Target position → Joint angles
- **Differential kinematics**: Joint velocities → End-effector velocity

### Control
- **Proportional control**: Error proportional to control command
- **Velocity command generation**: From pose error to (v, ω)
- **Speed saturation**: Prevents wheel speed exceed hardware limits

### Pose Representation
- **State vector**: [x, y, θ]
  - x, y: 2D position in world frame
  - θ: Orientation angle in world frame
- **Angle wrapping**: Keeps angles in [-π, π] for numerical stability

### Collision/Contact Physics
- **Penetration detection**: Distance from point to box
- **Separation force**: Proportional to overlap depth
- **Push mode logic**: Only pushes from behind, bounces otherwise

---

## File Structure

```
RobotChallenge/
├── coordinator.py          # Multi-robot task coordinator
├── push_sim.py            # Husky box-pushing simulation
├── puzzle_sim.py          # PuzzleBot visualization
├── sim.py                 # Core simulation framework (placeholder)
├── classes/
│   ├── husky/
│   │   ├── husky.py       # Husky robot class & kinematics model
│   │   └── lidar.py       # LiDAR sensor implementation
│   └── puzzlebot/
│       ├── puzzlebot.py   # PuzzleBot class & base kinematics
│       └── puzzlebot_arm.py # 3-DOF arm model & kinematics
└── PROJECT_PROMPT.md      # This file
```

---

## Typical Workflow

### 1. Robot Initialization
```python
# Create Husky with LiDAR
husky = Husky(pose=(0.5, -1.8, 0.0))
lidar = LiDAR(n_rays=360, max_range=8.0)
husky.attach_lidar(lidar)

# Create PuzzleBot with arm
arm_model = PuzzleBotArmModel(l1=0.10, l2=0.08, l3=0.06)
arm = PuzzleBotArm(model=arm_model, q_home=[0, π/6, -π/4])
bot_model = PuzzleBotModel(r=0.05, L=0.19)
bot = PuzzleBot(model=bot_model, arm=arm)

# Create coordinator
coordinator = Coordinator(husky, anymal, [bot])
```

### 2. Motion Control
```python
# Husky: Set velocity command
v, w = husky.model.compute_velocity_command(pose, goal)
omega = husky.model.inverse_kinematics(v, w)
husky.set_wheel_speeds(omega)

# PuzzleBot base: Set twist
bot.set_twist(v_linear, v_angular)

# PuzzleBot arm: Set joint target
arm.set_q_target(q_target)
```

### 3. Simulation Loop
```python
dt = 0.05  # Time step
for _ in range(num_steps):
    # Update robot states
    husky.step(dt)
    bot.step(dt)
    
    # Run coordinator logic
    coordinator.step()
    
    # Log data
    t_data.append(current_time)
    trajectory.append(pose)
```

### 4. Visualization
- Matplotlib animation with custom drawing functions
- Real-time pose updates
- Sensor data overlay (LiDAR rays, detected objects)
- Trajectory history

---

## Advanced Features

### Singularity Detection
- Monitors Jacobian determinant during arm motion
- Warns when approaching singular configurations
- Prevents joint velocity inversion errors

### Slip Modeling (Husky)
- Scaling factor 's' accounts for terrain friction
- Affects linear velocity: v' = v × s
- Models different terrain: s=1 (ideal), s<1 (slippery)

### Trajectory Planning
- Linear interpolation in Cartesian space
- Automatic inverse kinematics at each waypoint
- Singularity checking throughout trajectory

### Force-Torque Calculation
- Converts end-effector force to joint torques: τ = J^T × f
- Used for compliance control and force feedback

---

## Potential Extensions & Future Work

1. **3D Simulation**: Currently 2D-focused; extend to full 3D
2. **Dynamics Simulation**: Add masses, inertias, friction models
3. **Advanced Path Planning**: RRT, RRT*, Dijkstra's algorithm
4. **Grasping & Manipulation**: Contact mechanics, grasp planning
5. **AnyMAL Integration**: Full implementation of third robot type
6. **Real Hardware Integration**: ROS bridge for physical robots
7. **Visual Servoing**: Camera-based feedback control
8. **Machine Learning**: Learned controllers for complex tasks

---

## Dependencies & Requirements

- **Python 3.6+**
- **NumPy**: Numerical computations, linear algebra
- **Matplotlib**: Visualization and animation
- **Standard Library**: `enum`, `numpy`, `time`, `os`, `sys`

---

## Notes for AI Assistance

When asking about this project:
- **Kinematics questions**: Refer to specific model (PuzzleBot arm, PuzzleBot base, or Husky)
- **Control/motion**: Discuss twist commands, wheel speeds, joint targets
- **Simulation**: Consider physics approximations (no dynamics, penetration resolution only)
- **Tasks**: Box pushing, manipulation, coordination scenarios
- **Performance metrics**: Trajectory tracking error, task completion time, singularity avoidance

This is a robotics simulation and control project with emphasis on kinematics, autonomous navigation, and multi-robot coordination for complex manipulation tasks.
