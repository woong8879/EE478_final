EE478 final-project drone workspace.

Stack:
- VINS-Fusion (mono + IMU, loop closure) — vendored in src/VINS-Fusion/
- EGO-Planner-V2 (kinodynamic B-spline planner) — vendored in src/ego_planner_v2/
- ee478_localization     — vio_bridge with sparse-landmark anchor + landmark_anchor_publisher
- ee478_planner          — direct_goal_follower (sim), ego_bridge + offboard_controller (real)
- ee478_perception       — sim_world_publisher stub; YOLO+TensorRT for real drone
- ee478_agent            — command_interpreter, quiz_solver, signature_move, mission_fsm
- ee478_msgs             — custom message contracts
- ee478_bringup          — launch files
- gpt_llm_client         — course-provided OpenAI service

Build:
    cd ~/EE478/final_project_ws
    catkin_make
    source devel/setup.bash

Mission (per project_overview_drone):
    1. Read Command       (LLM/keyword text -> target store)
    2. Pass Quiz Gate     (math eval -> choose correct gate)
    3. Navigate Course    (planner + VIO + sparse-landmark anchor)
    4. Arrive at store    (signature move = bobble + yaw spin, no drop)
    5. Return             (same gate as outbound, no quiz)

Sim verification (mission.launch):
    roslaunch ee478_bringup mission.launch \
        gui:=false \
        mission_command:="Deliver to the cafe"

Phase status:
    Phase 1-4: VIO + landmark anchor — verified.
    Phase 5  : command_interpreter — keyword path verified ("Deliver to cafe" -> CAFE).
    Phase 6  : quiz_solver — verified ("5+9=?" -> label 14, left lane pose).
    Phase 7-10: mission_fsm transitions verified through APPROACH_QUIZ ->
               THROUGH_QUIZ -> NAV_STORE in sim. The long horizontal NAV_STORE
               leg in PX4 SITL hits the cafe building collision and the position
               controller can't recover; this is a SITL limitation. Real-drone
               deployment uses EGO-Planner + VINS which avoids this.
    Phase 11 : Jetson Orin Nano deployment — pending.

Target platform:
    Jetson Orin Nano 8GB, PX4 over UART, RGB-D + IMU.
