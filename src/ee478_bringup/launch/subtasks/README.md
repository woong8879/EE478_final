# Sub-task launches

Each sub-task brings up the minimum stack needed to test ONE mission
slice in isolation. They all share `s1_takeoff.launch` as the base
(PX4 SITL + VIO + offboard_controller), so the actual node under
test runs on top of an already-airborne drone.

| # | Launch                          | Tests                                  | New node                               |
|---|---------------------------------|----------------------------------------|----------------------------------------|
| 1 | `s1_takeoff.launch`             | Arm + ascend to `hover_z` and hold     | —                                      |
| 2 | `s2_command_read.launch`        | "Deliver to X" -> `/mission_target`    | command_screen_reader (TODO real)      |
| 3 | `s3_quiz_solve.launch`          | Quiz arithmetic -> `/quiz/chosen_pose` | quiz_screen_reader (TODO real)         |
| 4 | `s4_nav_course.launch`          | Single goal, EGO avoids obstacles      | —                                      |
| 5 | `s5_store_select.launch`        | All 3 stores -> `/target_store_pose`   | `store_identifier_node`                |
| 6 | `s6_signature.launch`           | Yaw spin in place                      | —                                      |
| 7 | `s7_return_gate.launch`         | cafe -> gate -> pickup                 | —                                      |
| 8 | `s8_land.launch`                | Descend + disarm                       | `land_node`                            |

## Test commands per sub-task

### 1. Takeoff
    roslaunch ee478_bringup s1_takeoff.launch
    # pass: gazebo iris z > 0.6 within 30 s, OFFBOARD + armed for >5 s.

### 2. Command read
    roslaunch ee478_bringup s2_command_read.launch \
        mission_command:="Deliver to the cafe"
    rostopic echo /mission_target   # expect CAFE

### 3. Quiz solve
    roslaunch ee478_bringup s3_quiz_solve.launch \
        question:="5 + 9 = ?" gate_left_label:=14 \
        gate_center_label:=7 gate_right_label:=9
    rostopic echo /quiz/chosen_label  # expect 14
    rostopic echo /quiz/chosen_pose   # expect y near 1.8

### 4. Obstacle nav
    roslaunch ee478_bringup s4_nav_course.launch \
        goal_x:=15.0 goal_y:=0.0
    # pass: gazebo iris within 1.0 m of (15, 0) without crashing.

### 5. Store select
    roslaunch ee478_bringup s5_store_select.launch \
        mission_command:="Deliver to the pharmacy"
    rostopic echo /target_store_pose  # expect y near -2.5

### 6. Signature
    roslaunch ee478_bringup s6_signature.launch
    # wait until takeoff complete, then:
    rostopic pub -1 /mission/signature_trigger std_msgs/Int32 "data: 7"
    rostopic echo /mission/signature_done   # expect True

### 7. Return via outbound gate
    roslaunch ee478_bringup s7_return_gate.launch \
        gate_y:=1.8 pickup_x:=0.0 pickup_y:=0.0
    # pass: gazebo iris within 0.6 m of (0, 0).

### 8. Land
    roslaunch ee478_bringup s8_land.launch
    # wait until hover, then:
    rostopic pub -1 /mission/land_trigger std_msgs/Bool "data: true"
    rostopic echo /mission/land_done   # expect True

## VIO source selection

Every sub-task accepts `vio_source:=gt` (default) or
`vio_source:=rtabmap`. `gt` makes the tests reproducible by removing
the VIO failure modes; `rtabmap` is what the real Jetson + RealSense
deployment will actually run.

## Replacing sim stubs for real-drone deployment

- `sim_world_publisher_node.py` → real `command_screen_reader_node` +
  `quiz_screen_reader_node` + `storefront_recogniser_node`
  (YOLO/AprilTag on the camera feed).
- `gt_vision_bridge_node.py` → already disabled when
  `vio_source:=rtabmap`.

The downstream consumers (`command_interpreter`, `quiz_solver`,
`store_identifier`, `signature_move`, `land_node`,
`mission_orchestrator`) are camera-source-agnostic.
