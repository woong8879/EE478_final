#!/usr/bin/env bash
# install_jetson_tuning.sh
# Run ONCE with sudo:  sudo bash ~/EE478_final/install_jetson_tuning.sh
#
# Makes the RealSense-stability settings persist across reboot via a
# systemd oneshot service (no bootloader edits):
#   * usbcore.usbfs_memory_mb = 1000  (default 16 MB -> D435i streams die)
#   * USB autosuspend disabled        (camera randomly suspends otherwise)
#   * jetson_clocks                   (lock max clocks so processing keeps up)
set -e

if [ "$(id -u)" -ne 0 ]; then
  echo "이 스크립트는 sudo로 실행해야 합니다:  sudo bash $0"; exit 1
fi

# 1) Boot-time tuning script ------------------------------------------------
cat > /usr/local/bin/ee478-jetson-tuning.sh <<'SH'
#!/usr/bin/env bash
# USB bulk-transfer buffer for RealSense (16 MB default is too small).
echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb 2>/dev/null || true
# Disable USB autosuspend as the default for newly-bound devices.
echo -1   > /sys/module/usbcore/parameters/autosuspend     2>/dev/null || true
# Force 'on' (no autosuspend) for any already-present Intel RealSense device.
for d in /sys/bus/usb/devices/*/; do
  v=$(cat "$d/idVendor" 2>/dev/null || true)
  if [ "$v" = "8086" ]; then echo on > "${d}power/control" 2>/dev/null || true; fi
done
# Lock Jetson clocks to max (disable DVFS) so SLAM/planner keep real time.
if command -v jetson_clocks >/dev/null 2>&1; then jetson_clocks || true; fi
SH
chmod +x /usr/local/bin/ee478-jetson-tuning.sh

# 2) udev rule: any RealSense plugged in AFTER boot also gets autosuspend off.
cat > /etc/udev/rules.d/99-realsense-no-autosuspend.rules <<'UDEV'
# Intel RealSense: disable USB autosuspend (prevents stream watchdog death)
ACTION=="add", SUBSYSTEM=="usb", ATTR{idVendor}=="8086", TEST=="power/control", ATTR{power/control}="on"
UDEV

# 3) systemd service to run the tuning script at every boot ------------------
cat > /etc/systemd/system/ee478-jetson-tuning.service <<'SVC'
[Unit]
Description=EE478 Jetson USB + clock tuning (RealSense stability)
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/ee478-jetson-tuning.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
SVC

udevadm control --reload-rules 2>/dev/null || true
systemctl daemon-reload
systemctl enable ee478-jetson-tuning.service
systemctl start  ee478-jetson-tuning.service

echo "--------------------------------------------------------------"
echo "설치 완료. 재부팅해도 자동 적용됩니다."
echo "  usbfs_memory_mb = $(cat /sys/module/usbcore/parameters/usbfs_memory_mb)  (1000 이어야 함)"
echo "  autosuspend     = $(cat /sys/module/usbcore/parameters/autosuspend)  (-1 이어야 함)"
echo "서비스 상태:  systemctl status ee478-jetson-tuning.service"
echo "--------------------------------------------------------------"
