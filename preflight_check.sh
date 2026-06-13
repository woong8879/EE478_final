#!/usr/bin/env bash
# preflight_check.sh — EE478 final real-drone go/no-go check
#
# Usage:
#   source ~/EE478_final/setup_env.sh
#   roslaunch ee478_bringup s1_takeoff.launch platform:=real &
#   sleep 20
#   ~/EE478_final/preflight_check.sh
#
# Or run standalone (launches s1 internally, checks, leaves it running):
#   ~/EE478_final/preflight_check.sh --launch

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
BOLD='\033[1m'; NC='\033[0m'
pass(){ echo -e "  ${GREEN}✅ PASS${NC}  $1"; }
fail(){ echo -e "  ${RED}❌ FAIL${NC}  $1"; FAILURES=$((FAILURES+1)); }
warn(){ echo -e "  ${YELLOW}⚠  WARN${NC}  $1"; }
header(){ echo -e "\n${BOLD}── $1 ──────────────────────────────────────────${NC}"; }

FAILURES=0
LAUNCH_MODE=false
[[ "${1:-}" == "--launch" ]] && LAUNCH_MODE=true

# ── workspace source ──────────────────────────────────────────────────────────
WS="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/noetic/setup.bash
[ -f "$HOME/catkin_ws/devel/setup.bash" ]  && source "$HOME/catkin_ws/devel/setup.bash"
[ -f "$WS/devel/setup.bash" ]              && source "$WS/devel/setup.bash" --extend
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# ── helper: get topic hz (returns empty string on timeout) ────────────────────
topic_hz(){           # $1=topic $2=window_s
  timeout "$2" rostopic hz "$1" 2>/dev/null | grep -m1 "average rate" | awk '{print $3}'
}
param_int(){          # $1=param_name → integer value or empty
  timeout 5 rosservice call /mavros/param/get "$1" 2>/dev/null \
    | grep -oE "integer: [0-9]+" | head -1 | awk '{print $2}'
}
pose_z(){             # $1=topic → z value
  timeout 4 rostopic echo -n1 "$1" 2>/dev/null \
    | python3 -c "
import sys, re
data = sys.stdin.read()
m = re.search(r'position:.*?z:\s*([-0-9.eE+]+)', data, re.DOTALL)
if m: print(m.group(1))
" 2>/dev/null
}

echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║         EE478 Pre-flight Check                  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"

# ─────────────────────────────────────────────────────────────────────────────
header "1. Hardware"

# RealSense USB3
RS_SPEED=$(for d in /sys/bus/usb/devices/*/; do
  v=$(cat "$d/idVendor" 2>/dev/null)
  [ "$v" = "8086" ] && cat "$d/speed" 2>/dev/null && break
done)
if [ "${RS_SPEED:-0}" -ge 5000 ] 2>/dev/null; then
  pass "RealSense D435i: USB${RS_SPEED} Mbit/s (USB3 ✓)"
else
  fail "RealSense D435i: speed=${RS_SPEED:-NOT FOUND} — must be 5000 Mbit/s (USB3 직결 필요)"
fi

# ttyTHS0 (Pixhawk)
if ls /dev/ttyTHS0 >/dev/null 2>&1; then
  pass "FCU serial: /dev/ttyTHS0 exists"
else
  fail "FCU serial: /dev/ttyTHS0 not found — Pixhawk USB 미연결?"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "2. rosmaster"

if rostopic list >/dev/null 2>&1; then
  pass "rosmaster reachable at $ROS_MASTER_URI"
else
  fail "rosmaster not reachable — roscore 실행 필요"
  echo -e "\n${RED}rosmaster 없이 나머지 체크 불가 — 종료${NC}"
  exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
header "3. Stack launch (--launch 모드)"

if $LAUNCH_MODE; then
  echo "  s1_takeoff.launch platform:=real 시작..."
  roslaunch ee478_bringup s1_takeoff.launch platform:=real \
    > /tmp/s1_preflight.log 2>&1 &
  LAUNCH_PID=$!
  echo "  PID=$LAUNCH_PID, 초기화 대기 (25s)..."
  sleep 25
  pass "Stack launched PID=$LAUNCH_PID"
else
  warn "스택이 이미 실행 중이라고 가정 (--launch 아님)"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "4. MAVROS / FCU"

STATE=$(timeout 5 rostopic echo -n1 /mavros/state 2>/dev/null)
if echo "$STATE" | grep -q "connected: True"; then
  pass "FCU connected: True"
else
  fail "FCU connected: False — MAVROS↔Pixhawk 링크 없음"
fi

ARMED=$(echo "$STATE" | grep "armed:" | awk '{print $2}')
if [ "${ARMED}" = "False" ]; then
  pass "FCU armed: False (pre-flight 정상)"
else
  fail "FCU armed: True — 비행 전에 비무장 상태여야 함"
fi

MODE=$(echo "$STATE" | grep "mode:" | awk '{print $2}' | tr -d '"')
pass "FCU mode: ${MODE:-unknown}"

# ─────────────────────────────────────────────────────────────────────────────
header "5. EKF2 Parameters"

AID=$(param_int "EKF2_AID_MASK")
if [ "${AID:-0}" -ge 8 ] && (( (AID & 8) )); then
  pass "EKF2_AID_MASK = $AID (vision position bit ✓)"
else
  fail "EKF2_AID_MASK = ${AID:-?} — vision bit(8) 꺼짐. 24로 설정 필요"
fi

HGT=$(param_int "EKF2_HGT_REF")
if [ "${HGT:-0}" -eq 3 ]; then
  pass "EKF2_HGT_REF = 3 (vision height ✓)"
else
  warn "EKF2_HGT_REF = ${HGT:-?} — vision(3) 권장"
fi

# ─────────────────────────────────────────────────────────────────────────────
header "6. Topic Rates"

# 모든 토픽을 병렬로 8초간 측정 → /tmp/hz_*.txt
TOPICS=(
  "/mavros/imu/data:40:Pixhawk IMU"
  "/camera/color/image_raw:20:RealSense color"
  "/camera/aligned_depth_to_color/image_raw:20:RealSense depth(aligned)"
  "/rtabmap/odom:4:RTAB-Map VIO odom"
  "/mavros/vision_pose/pose:20:vision_pose → PX4"
  "/mavros/local_position/pose:20:PX4 local position"
  "/mavros/setpoint_position/local:15:offboard setpoint stream"
)
rm -f /tmp/preflight_hz_*.txt
PIDS=()
for entry in "${TOPICS[@]}"; do
  TOPIC=$(echo "$entry" | cut -d: -f1)
  SAFE=$(echo "$TOPIC" | tr '/' '_')
  (timeout 8 rostopic hz "$TOPIC" 2>/dev/null \
    | grep -m1 "average rate" | awk '{print $3}' \
    > "/tmp/preflight_hz_${SAFE}.txt") &
  PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

for entry in "${TOPICS[@]}"; do
  TOPIC=$(echo "$entry" | cut -d: -f1)
  MIN=$(echo "$entry" | cut -d: -f2)
  LABEL=$(echo "$entry" | cut -d: -f3)
  SAFE=$(echo "$TOPIC" | tr '/' '_')
  HZ=$(cat "/tmp/preflight_hz_${SAFE}.txt" 2>/dev/null | tr -d '[:space:]')
  if [ -z "$HZ" ]; then
    fail "$LABEL: NO DATA"
  elif python3 -c "exit(0 if float('$HZ') >= $MIN else 1)" 2>/dev/null; then
    pass "$LABEL: ${HZ} Hz (≥${MIN} ✓)"
  else
    fail "$LABEL: ${HZ} Hz (최소 ${MIN} Hz 필요)"
  fi
done
rm -f /tmp/preflight_hz_*.txt

# ─────────────────────────────────────────────────────────────────────────────
header "7. EKF Convergence"

EKF_Z=$(pose_z /mavros/local_position/pose)
VIO_Z=$(pose_z /mavros/vision_pose/pose)

if [ -n "$EKF_Z" ]; then
  ABS=$(python3 -c "print(abs(float('$EKF_Z')))" 2>/dev/null)
  if python3 -c "exit(0 if abs(float('$EKF_Z')) < 0.30 else 1)" 2>/dev/null; then
    pass "EKF z = ${EKF_Z} m (|z| < 0.30 m ✓)"
  else
    fail "EKF z = ${EKF_Z} m — 발산 의심 (|z| ≥ 0.30 m)"
  fi
else
  fail "EKF z: 읽기 실패"
fi

if [ -n "$VIO_Z" ]; then
  pass "VIO z = ${VIO_Z} m"
else
  warn "VIO z: 읽기 실패 (rtabmap 초기화 중?)"
fi

# ─────────────────────────────────────────────────────────────────────────────
echo
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${NC}"
if [ "$FAILURES" -eq 0 ]; then
  echo -e "${BOLD}║  ${GREEN}GO  ✅  모든 체크 통과 — 비행 가능${NC}${BOLD}            ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
  echo
  echo "  다음 단계:"
  echo "    1. 프로펠러 + 배터리 연결"
  echo "    2. 외부 컨트롤러로 OFFBOARD 전환"
  echo "    3. 코드가 자동 ARM → 0.5m 호버링"
  exit 0
else
  echo -e "${BOLD}║  ${RED}NO-GO ❌  ${FAILURES}개 실패 — 비행 불가${NC}${BOLD}               ║${NC}"
  echo -e "${BOLD}╚══════════════════════════════════════════════════╝${NC}"
  exit 1
fi
