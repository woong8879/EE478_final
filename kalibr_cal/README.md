# Pixhawk-IMU ↔ D435i IR stereo extrinsic calibration (Kalibr)

Goal: get an accurate `T_cam_imu` for the **Pixhawk IMU** (`/mavros/imu/data_raw`)
and the IR stereo pair, then drop it into
`src/VINS-Fusion/config/realsense_d435i/realsense_stereo_pixhawk_config.yaml`
(replacing the hand-guessed `body_T_cam0/1`).

## 0. Build Kalibr (one-time)
Deps (run yourself — needs sudo):
```bash
sudo apt-get update
sudo apt-get install -y python3-igraph libv4l-dev
```
Build (no sudo):
```bash
cd ~/kalibr_ws && catkin build -DCMAKE_BUILD_TYPE=Release
source ~/kalibr_ws/devel/setup.bash
```

## 1. Print the AprilGrid
Kalibr ships `aprilgrid` PDFs; print one, tape it flat to a rigid board.
**Measure a tag side and put it in `aprilgrid.yaml` (`tagSize`, metres).**

## 2. Record a bag (camera + IMU together)
Bring up the camera + MAVROS (so `/mavros/imu/data_raw` + IR streams exist),
then record. IMPORTANT: move slowly+smoothly, excite ALL 6 DoF (3 rotations,
3 translations), keep the whole grid in both IR views, ~60-90 s.
```bash
rosbag record -O ~/kalibr_cal/cam_imu.bag \
  /camera/infra1/image_rect_raw /camera/infra2/image_rect_raw \
  /mavros/imu/data_raw
```
Kalibr likes ~20 Hz images for cam-imu; the IR run at 30 Hz is fine.

## 3. Calibrate cameras (intrinsics + stereo) -> camchain.yaml
```bash
kalibr_calibrate_cameras \
  --bag ~/kalibr_cal/cam_imu.bag \
  --topics /camera/infra1/image_rect_raw /camera/infra2/image_rect_raw \
  --models pinhole-radtan pinhole-radtan \
  --target ~/kalibr_cal/aprilgrid.yaml
# -> cam_imu-camchain.yaml
```

## 4. Calibrate camera-IMU -> camchain-imucam.yaml (THE extrinsic)
```bash
kalibr_calibrate_imu_camera \
  --bag ~/kalibr_cal/cam_imu.bag \
  --cam ~/kalibr_cal/cam_imu-camchain.yaml \
  --imu ~/kalibr_cal/imu_pixhawk.yaml \
  --target ~/kalibr_cal/aprilgrid.yaml
# -> cam_imu-camchain-imucam.yaml with T_cam_imu (4x4) per camera + timeshift
```

## 5. Convert to VINS
- Kalibr gives `T_cam_imu` (imu->cam... actually cam<-imu). VINS wants
  `body_T_cam` = `T_imu_cam` = inverse(`T_cam_imu`). I (Claude) will do the
  inverse + write it into the pixhawk config, and set `estimate_extrinsic: 0`
  (trust the calibrated value) and `td` from Kalibr's timeshift.

Hand the two output yaml files back and I'll wire them into VINS.
