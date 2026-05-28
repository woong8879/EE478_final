EE478 final-project drone workspace.

Stack:
- VINS-Fusion (mono + IMU, loop closure) — vendored in src/VINS-Fusion/
- EGO-Planner-V2 (kinodynamic B-spline planner) — vendored in src/ego_planner_v2/
- ee478_localization     — vio_bridge with sparse-landmark anchor
- ee478_planner          — ego_bridge (agent goal <-> EGO trajectory <-> offboard)
- ee478_perception       — YOLOv11n storefront recogniser (TensorRT on Jetson)
- ee478_agent            — LLM mission state machine
- ee478_msgs             — custom message contracts
- ee478_bringup          — launch files
- gpt_llm_client         — course-provided OpenAI service

Build:
    cd ~/EE478/final_project_ws
    catkin_make
    source devel/setup.bash

Mission (per project_overview_drone):
    1. Read Command       (LLM text -> target store)
    2. Pass Quiz Gate     (math/quiz signboard -> choose gate)
    3. Navigate Course    (EGO-Planner + VIO + sparse-landmark anchor)
    4. Arrive at store    (signature move, no drop)
    5. Return             (same gate as outbound, no quiz)

Target platform:
    Jetson Orin Nano 8GB, PX4 over UART, RGB-D + IMU.
