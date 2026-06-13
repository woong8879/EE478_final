#!/usr/bin/env bash
# reset_camera.sh — software USB reset for the RealSense D435i(s) when they
# stick ("control_transfer Resource temporarily unavailable" / no IR streams,
# which happens after repeated launches + abrupt kills). No sudo, no physical
# replug. Run this, then relaunch the camera(s).
#
# TWO D435i are connected now (forward VIO cam + downward delivery cam):
# resets EVERY 8086:0b3a device, one per line.
pkill -9 -f realsense2 2>/dev/null
sleep 2
found=0
lsusb | grep -i "8086:0b3a" | sed -E 's/Bus ([0-9]+) Device ([0-9]+).*/\/dev\/bus\/usb\/\1\/\2/' \
| while read -r busdev; do
  [ -z "$busdev" ] && continue
  python3 - "$busdev" <<'PY'
import sys, fcntl
USBDEVFS_RESET = ord('U') << 8 | 20   # _IO('U', 20)
with open(sys.argv[1], 'wb') as f:
    fcntl.ioctl(f, USBDEVFS_RESET, 0)
print("RealSense USB reset OK:", sys.argv[1])
PY
done
lsusb | grep -qi "8086:0b3a" || { echo "RealSense (8086:0b3a) not found on USB"; exit 1; }
