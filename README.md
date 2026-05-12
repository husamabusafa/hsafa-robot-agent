# hsafa-robot

Voice + vision + body brain for the Reachy Mini, powered by Gemini Live and
Haseef (Hsafa Core).

## Project layout

```
.
├── main.py              # The one entry point: Gemini Live + Haseef brain + robot
├── setup_haseef.py      # One-shot: create/update the Haseef entity on Hsafa Core
├── hsafa_robot/         # Robot package (gemini_live, camera, controller, scheduler, ...)
├── hsafa_voice_vision.py# Camera + RobotController helpers
├── requirements.txt
├── scripts/
│   └── daemon.sh        # start/stop the Reachy Mini daemon (motors + media)
├── docs/                # Design & architecture notes
├── archive/             # Old experiments and legacy scripts (not used at runtime)
└── .venv/               # Python virtualenv with reachy-mini installed
```

## Quick start

```bash
# 1. Activate the venv (already created)
source .venv/bin/activate

# 2. (Once) Create the Haseef entity on Hsafa Core
python setup_haseef.py

# 3. Start the Reachy Mini daemon (motors + media on :8000)
./scripts/daemon.sh start

# 4. Start the brain (Gemini Live + Haseef + robot controller)
python main.py
```

Env vars expected in `.env` (see `.env.example`):
`GEMINI_API_KEY`, `HSAFA_CORE_URL`, `HSAFA_CORE_KEY`, `HASEEF_ID`.

---

# Reachy Mini hardware notes

Getting started with a **Reachy Mini (wired / USB-C version)** from macOS.

## How this robot actually connects

The wired Reachy Mini is a **pure USB peripheral**, not a networked device:

| USB function  | Purpose                                                |
|---------------|--------------------------------------------------------|
| Serial (CDC)  | Motor bus — 9 motors (body yaw, 6-DOF Stewart platform, 2 antennas) |
| USB Audio     | Speaker + microphones                                  |
| UVC Camera    | `Reachy Mini Camera` webcam                            |

There is **no `reachy.local`**, no Ethernet-over-USB, no onboard Pi you SSH
into. Instead, a Python daemon runs **on your Mac**, opens the USB serial
port, and exposes an HTTP/WebSocket API on `localhost:8000`. Your scripts
use the `reachy-mini` SDK which talks to that local daemon.

```
  Mac  ┌────────────────────┐   USB-C    ┌────────────────┐
       │  your Python code  │            │  Reachy Mini   │
       │        │           │            │  motors/cam/   │
       │        ▼           │   serial   │  audio/mics    │
       │  reachy-mini SDK   │◀──────────▶│                │
       │        │ HTTP :8000│    UVC     │                │
       │        ▼           │    UAC     │                │
       │  reachy-mini-daemon│            │                │
       └────────────────────┘            └────────────────┘
```

## 1. Install

Requires Python 3.10+ (tested with 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Start the daemon (once per session)

```bash
./scripts/daemon.sh start      # launches daemon in background
./scripts/daemon.sh status
./scripts/daemon.sh logs       # follow logs (Ctrl-C to stop tailing)
./scripts/daemon.sh stop       # stops daemon; robot goes to sleep
```

On first start the daemon will:
1. Auto-detect the USB serial port
2. Initialise all 9 motors (you'll see `[OK]` lines per motor)
3. Wake the robot up (head lifts into its neutral pose)
4. Listen on <http://localhost:8000>

If motor init fails (stuck at "Waiting for voltage..."), unplug / replug the
USB-C cable and try again. Make sure you are using a **data-capable** USB-C
cable.

## 3. Run the examples

With the daemon running:

```bash
python examples/01_hello.py              # read current pose + joint state
python examples/02_head_motion.py        # nod yes, shake no, tilt
python examples/03_antennas_and_body.py  # body rotation + antenna flap
python examples/04_look_around.py        # "curious" look_at_world behaviour
```

## 4. Minimal code snippet

```python
import math, numpy as np
from scipy.spatial.transform import Rotation as R
from reachy_mini import ReachyMini

def pose(roll=0, pitch=0, yaw=0):
    M = np.eye(4)
    M[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    return M

with ReachyMini() as reachy:
    reachy.goto_target(head=pose(pitch=math.radians(15)), duration=0.5)
    reachy.goto_target(body_yaw=math.radians(30), duration=0.8)
    reachy.goto_target(antennas=[math.radians(45), math.radians(-45)],
                       duration=0.3, body_yaw=None)
    reachy.look_at_world(0.5, 0.2, 0.1, duration=1.0)  # aim head at (x,y,z) m
```

## Key SDK methods

- `reachy.wake_up()` / `reachy.goto_sleep()`
- `reachy.goto_target(head=<4x4>, antennas=[l, r], body_yaw=f, duration=s)` — smooth (min-jerk) motion
- `reachy.set_target(...)` — instant target (same args)
- `reachy.look_at_world(x, y, z, duration)` — aim head at 3D point (meters, robot frame: +X forward, +Y left, +Z up)
- `reachy.get_current_head_pose()` → 4x4 numpy matrix
- `reachy.get_current_joint_positions()` → `(head_joints, antenna_joints)`
- `reachy.enable_motors()` / `reachy.disable_motors()`

> When calling `goto_target` with only `head=...`, pass `body_yaw=None` to
> avoid unintentionally resetting the body (the default value is `0.0`).

## Troubleshooting

- **Daemon exits immediately** — the USB serial port wasn't found. Check
  `ls /dev/cu.usbmodem*` and pass it explicitly:
  `reachy-mini-daemon -p /dev/cu.usbmodemXXXXXXX ...`
- **"Address already in use" on port 8000** — another daemon is still
  running: `./scripts/daemon.sh stop`.
- **Head doesn't move when calling SDK** — ensure the daemon log shows
  `Daemon started successfully.` and `Motor control mode: …` (not
  `Disabled`). The SDK call `reachy.wake_up()` re-enables motors if needed.
- **macOS permissions** — first run will prompt for camera/mic access
  (only when you enable media; we use `--no-media` by default here).
# hsafa-robot-testing
