# Sub-task launches — sim + real-drone

Every launch takes `platform:={sim,real}`.

- `platform:=sim` (default) brings up PX4 SITL + Gazebo + GT vision
  bridge + sim perception stubs. Reproducible regression test.
- `platform:=real` brings up MAVROS connecting to the Pixhawk over
  UART (`fcu_url:=serial:///dev/ttyTHS1:921600` by default) + RTAB-Map
  on the depth camera + vio_bridge. **`auto_arm` is forced FALSE**;
  the operator arms via QGroundControl or the RC after verifying
  battery + RC failsafe + clear test area.

## Real-drone safety checklist (do EVERY test)

Before launching anything with `platform:=real`:

1. **RC failsafe**: failsafe mode set to LAND or HOLD; channel
   trimmed; receiver bound.
2. **Battery**: full pack, undamaged. Low-battery failsafe enabled.
3. **Geofence**: enabled in PX4 with sensible XY/Z limits for the
   test area.
4. **Tether or net** for the first hover (especially s1 + s4).
5. **Kill switch (operator hand on RC)** at all times. Disarm
   switch tested before takeoff.
6. **OFFBOARD only after takeoff confirmation**: launch the stack,
   wait for setpoint stream to be visible in QGC, THEN arm and
   switch to OFFBOARD.

## Per-task tests

### 1. Takeoff (`s1_takeoff.launch`)
    # Sim
    roslaunch ee478_bringup s1_takeoff.launch platform:=sim
    # Real
    roslaunch ee478_bringup s1_takeoff.launch platform:=real \
        fcu_url:=serial:///dev/ttyTHS1:921600 hover_z:=0.5
PASS: drone climbs to hover_z within 30 s and stays armed + OFFBOARD.

### 2. Command read (`s2_command_read.launch`)
Pure software, no flight motion. Drone hovers; LLM parses the command
string into a category.
    roslaunch ee478_bringup s2_command_read.launch platform:=real \
        mission_command:="Deliver to the pharmacy"
    rostopic echo /mission_target   # PHARMACY

### 3. Quiz solve (`s3_quiz_solve.launch`)
Sim publishes /quiz/gates from GT. Real-drone: manually inject (or
wire up your quiz_screen_reader). Drone keeps hovering.
    rostopic pub -1 /quiz/gates ee478_msgs/QuizGateArray \
      "{header: {frame_id: 'map'}, question: '5+9=?',
        gates: [{label: 14, center_world: {x: 3.0, y: 1.8, z: 0.7}},
                {label: 7,  center_world: {x: 3.0, y: 0.0, z: 0.7}},
                {label: 9,  center_world: {x: 3.0, y:-1.8, z: 0.7}}]}"
    rostopic echo /quiz/chosen_label   # 14

### 4. Obstacle nav (`s4_nav_course.launch`)
First real-drone run: small step.
    roslaunch ee478_bringup s4_nav_course.launch platform:=real \
        goal_x:=2.0 goal_y:=0.0 max_vel:=0.4
PASS: drone arrives near (2, 0) without grazing obstacles.

### 5. Store select (`s5_store_select.launch`)
Sim publishes /semantic_map from GT. Real-drone: manually inject
once you have at least one store position (template below) until
the real storefront_recogniser is wired.
    rostopic pub -1 /semantic_map ee478_msgs/SemanticMap \
      "{header: {frame_id: 'map'},
        stores: [{store_id: 1, category: 'CAFE',
                  position_world: {x: 3.0, y: 1.0, z: 1.0},
                  category_confidence: 1.0, visited: false}],
        pickup_point: {x: 0, y: 0, z: 0}}"

### 6. Signature (`s6_signature.launch`)
In-place yaw spin. Real-drone uses 90° first.
    roslaunch ee478_bringup s6_signature.launch platform:=real \
        spin_dyaw_deg:=90.0
    # after takeoff:
    rostopic pub -1 /mission/signature_trigger std_msgs/Int32 "data: 7"

### 7. Return via outbound gate (`s7_return_gate.launch`)
Use SHORT legs first on real drone. `gate_x:=2.0 gate_y:=0` means
"go 1 m forward of pickup, fly through, return to pickup."
    roslaunch ee478_bringup s7_return_gate.launch platform:=real \
        gate_x:=2.0 gate_y:=0.0 max_vel:=0.4

### 8. Land (`s8_land.launch`)
    roslaunch ee478_bringup s8_land.launch platform:=real
    rostopic pub -1 /mission/land_trigger std_msgs/Bool "data: true"
PASS: drone descends smoothly; disarms when close to ground.

## What still needs to be wired for the real drone

| Component                                    | Sim                                | Real (TODO)                                  |
|----------------------------------------------|------------------------------------|----------------------------------------------|
| `/mission_command` source                    | rostopic pub                       | command_screen_reader (camera + OCR/YOLO)    |
| `/quiz/gates` source                         | sim_world_publisher (from GT)      | quiz_screen_reader (camera + OCR/YOLO)       |
| `/semantic_map` source                       | sim_world_publisher (from GT)      | storefront_recogniser (camera + YOLO)        |
| Depth cloud for EGO grid_map (s4 / s7)       | iris_depth_camera/depth/points     | RealSense /camera/depth/color/points         |
| RGB + depth + camera_info for RTAB-Map       | iris_depth_camera_vio/* topics     | RealSense /camera/{rgb,depth}/*              |
| Static base_link -> camera_link TF           | px4_sitl.launch's tf_assist        | px4_real.launch (measure your mount + edit)  |
