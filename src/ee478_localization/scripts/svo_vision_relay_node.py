#!/usr/bin/env python3
"""svo_vision_relay_node.py

Relay SVO Pro's pose (/svo/pose_imu, PoseWithCovarianceStamped) to PX4's
external-vision input, with a jump gate so a VO glitch can't yank the EKF.

Two output modes (~ev_mode):

  "pose"  (legacy)  -> /mavros/vision_pose/pose       (PoseStamped, position)
                    +  /mavros/vision_speed/speed_twist_cov (velocity)
                       NOTE: with EKF2_EV_NOISE_MD=1 the PoseStamped has no
                       covariance, so PX4 under-weights EV position; vertically
                       the accel-Z bias then wins and the drone sinks.

  "odom"  (default) -> /mavros/odometry/out           (nav_msgs/Odometry)
                       ONE message carrying pose + velocity + EXPLICIT
                       covariances. This is PX4 1.15's canonical EV interface:
                       EKF2_EV_CTRL=15 fuses position+velocity+yaw from it. The
                       tight pose covariance makes the EKF actually track VIO
                       (instead of drifting on the biased accel), and the
                       velocity (finite-difference of the gated pose) cancels
                       the phantom vertical-velocity bias -- holding altitude
                       WITHOUT baro. ROS Odometry convention: pose in the world
                       (parent) frame, twist in the BODY (child) frame, so we
                       rotate the world-frame VIO velocity into body frame.

SVO's body frame = the Pixhawk IMU we calibrated against (= base_link, via the
Kalibr T_B_C), and SVO's world frame is gravity-aligned ENU, so pose goes
almost straight through (mavros handles ENU->NED).
"""
import math
import rospy
from geometry_msgs.msg import (PoseWithCovarianceStamped, PoseStamped,
                               TwistWithCovarianceStamped)
from nav_msgs.msg import Odometry


def quat_rotate_inverse(q, v):
    """Rotate world-frame vector v into body frame (apply q^-1).

    q = (w,x,y,z) is the body orientation in world (rotates body->world).
    """
    w, x, y, z = q
    # conjugate (world->body)
    x, y, z = -x, -y, -z
    vx, vy, vz = v
    # t = 2 * cross(qv, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    # v' = v + w*t + cross(qv, t)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return (rx, ry, rz)


class SvoVisionRelay(object):
    def __init__(self):
        self.in_topic = rospy.get_param("~in_topic", "/svo/pose_imu")
        self.max_jump = float(rospy.get_param("~max_jump_m", 0.5))
        # implied-speed outlier gate (m/s). DISABLED by default (100): in
        # flight SVO is noisy enough that a low gate rejects many GOOD frames
        # -> EV position dropout -> EKF dead-reckons on the biased accel ->
        # drift. The 0.5 m absolute jump gate already catches real glitches.
        # Lower this only if single-frame spikes are a problem and the 0.5 m
        # gate misses them.
        self.max_track_vel = float(rospy.get_param("~max_track_vel", 100.0))
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.ev_mode = rospy.get_param("~ev_mode", "odom").lower()

        # legacy "pose" mode topics
        self.out_topic = rospy.get_param("~vision_pose_topic",
                                         "/mavros/vision_pose/pose")
        self.speed_topic = rospy.get_param(
            "~speed_topic", "/mavros/vision_speed/speed_twist_cov")
        # "odom" mode topic + frames
        self.odom_topic = rospy.get_param("~odom_topic",
                                          "/mavros/odometry/out")
        self.odom_parent = rospy.get_param("~odom_parent_frame", "odom")
        self.odom_child = rospy.get_param("~odom_child_frame", "base_link")

        # velocity differentiation
        self.speed_lp = float(rospy.get_param("~speed_lp_alpha", 0.5))
        self.dt_min = float(rospy.get_param("~speed_dt_min", 0.004))
        self.dt_max = float(rospy.get_param("~speed_dt_max", 0.2))
        # EKF2_EV_NOISE_MD=1 => PX4 reads noise FROM the message covariance.
        self.pos_var = float(rospy.get_param("~pos_var", 0.01))    # xy: 0.1 m std
        # z is trusted ~100% to VIO: tiny variance so the EKF snaps height to
        # VIO and the accel-Z bias can't drift it (the whole sink problem).
        self.pos_var_z = float(rospy.get_param("~pos_var_z", 0.0004))  # 2 cm std
        self.vel_var = float(rospy.get_param("~speed_var", 0.04))  # 0.2 m/s std

        self.last = None          # last accepted RAW position (x,y,z)
        self.last_t = None        # timestamp of last accepted pose (s)
        self.vel_f = [0.0, 0.0, 0.0]
        self.jumps = 0

        # --- landmark / AprilTag anchor blend (VIO drift correction) ---
        # /landmark_anchor_pose says "the drone IS at this world pose". We hold
        # a world-frame offset = anchor - raw_VO and slide the APPLIED offset
        # toward it at anchor_blend_rate (m/s), then add it to the published
        # pose, so the EV stream is corrected smoothly and never teleports.
        # No anchor msgs -> offset stays 0 -> identical to the old behaviour.
        self.anchor_topic = rospy.get_param("~anchor_topic",
                                             "/landmark_anchor_pose")
        self.anchor_blend_rate = float(
            rospy.get_param("~anchor_blend_rate", 0.2))
        self.anchor_max_shift = float(
            rospy.get_param("~anchor_max_shift_m", 5.0))
        self.applied_offset = [0.0, 0.0, 0.0]
        self.target_offset = [0.0, 0.0, 0.0]
        self._last_blend_t = None

        if self.ev_mode == "odom":
            self.pub_odom = rospy.Publisher(self.odom_topic, Odometry,
                                            queue_size=10)
            self.pub = self.pub_v = None
        else:
            self.pub = rospy.Publisher(self.out_topic, PoseStamped,
                                       queue_size=10)
            self.pub_v = rospy.Publisher(self.speed_topic,
                                         TwistWithCovarianceStamped,
                                         queue_size=10)
            self.pub_odom = None
        rospy.Subscriber(self.in_topic, PoseWithCovarianceStamped, self.cb,
                         queue_size=50)
        rospy.Subscriber(self.anchor_topic, PoseStamped, self.on_anchor,
                         queue_size=5)
        rospy.loginfo("[svo_relay] mode=%s  %s -> %s (max_jump %.2fm)",
                      self.ev_mode, self.in_topic,
                      self.odom_topic if self.ev_mode == "odom"
                      else self.out_topic, self.max_jump)

    def _velocity(self, p, t):
        """Filtered world-frame velocity between the last accepted pose and now;
        None if not computable (no anchor / bad dt)."""
        if self.last is None or self.last_t is None:
            return None
        dt = t - self.last_t
        if not (self.dt_min < dt < self.dt_max):
            return None
        a = self.speed_lp
        self.vel_f[0] = a * (p.x - self.last[0]) / dt + (1.0 - a) * self.vel_f[0]
        self.vel_f[1] = a * (p.y - self.last[1]) / dt + (1.0 - a) * self.vel_f[1]
        self.vel_f[2] = a * (p.z - self.last[2]) / dt + (1.0 - a) * self.vel_f[2]
        return (self.vel_f[0], self.vel_f[1], self.vel_f[2])

    def on_anchor(self, msg):
        """Landmark/AprilTag fix: 'the drone IS at msg.pose'. Set the target
        world offset = anchor - raw_VO; cb() slides applied_offset toward it at
        anchor_blend_rate so the published EV pose is corrected smoothly."""
        if self.last is None:
            return
        tx = float(msg.pose.position.x) - self.last[0]
        ty = float(msg.pose.position.y) - self.last[1]
        # z is NOT corrected by the anchor: the VIO z + the offboard z-PID hold
        # altitude well, and the tag's solvePnP DEPTH (z) is the noisiest axis.
        # Injecting it oscillated the altitude -> the drone bobbed -> SVO lost
        # features -> VIO diverged -> crash. The anchor is for HORIZONTAL drift.
        tz = 0.0
        # Reject a fix that would shift the output far beyond the offset we are
        # already applying -- almost certainly a tag-id mismatch, not drift.
        d = math.sqrt((tx - self.applied_offset[0]) ** 2
                      + (ty - self.applied_offset[1]) ** 2
                      + (tz - self.applied_offset[2]) ** 2)
        if d > self.anchor_max_shift:
            rospy.logwarn_throttle(
                1.0, "[svo_relay] ignoring anchor (would shift %.2f m)", d)
            return
        self.target_offset = [tx, ty, tz]

    def _blend_offset(self, t):
        """Slide applied_offset toward target_offset at anchor_blend_rate."""
        if self._last_blend_t is None:
            self._last_blend_t = t
            return
        dt = t - self._last_blend_t
        self._last_blend_t = t
        if dt <= 0.0 or dt > 1.0:
            return
        step = self.anchor_blend_rate * dt
        for i in range(3):
            d = self.target_offset[i] - self.applied_offset[i]
            if abs(d) <= step:
                self.applied_offset[i] = self.target_offset[i]
            else:
                self.applied_offset[i] += step if d > 0 else -step

    def cb(self, msg):
        p = msg.pose.pose.position
        t = msg.header.stamp.to_sec()
        if self.last is not None and self.last_t is not None:
            d = math.sqrt((p.x - self.last[0]) ** 2 +
                          (p.y - self.last[1]) ** 2 +
                          (p.z - self.last[2]) ** 2)
            dt = t - self.last_t
            # Outlier = either a big absolute jump OR an implied speed no real
            # (slow indoor) drone can do. SVO feature-loss spikes ("Lost N
            # features") are single-frame; rejecting the frame (without moving
            # the anchor) means the next good frame diffs from the correct
            # pre-spike pose, so no pose OR velocity spike reaches the EKF.
            spd = d / dt if dt > 1e-4 else 0.0
            if d > self.max_jump or spd > self.max_track_vel:
                self.jumps += 1
                rospy.logwarn_throttle(
                    1.0, "[svo_relay] outlier d=%.2fm spd=%.1fm/s, hold (#%d)",
                    d, spd, self.jumps)
                if self.jumps < 10:
                    return                       # gated: no pose, no velocity
                rospy.logwarn("[svo_relay] 10 outliers; resetting baseline")
                self.jumps = 0
                self.last = None                 # drop anchor: no velocity spike
                self.last_t = None
            else:
                self.jumps = 0

        vel = self._velocity(p, t)               # world-frame, or None
        self._blend_offset(t)                    # advance the anchor correction

        if self.ev_mode == "odom":
            self._pub_odom(msg, vel)
        else:
            self._pub_pose_speed(msg, vel)

        self.last = (p.x, p.y, p.z)
        self.last_t = t

    def _pub_odom(self, msg, vel):
        od = Odometry()
        od.header.stamp = msg.header.stamp
        od.header.frame_id = self.odom_parent     # world (ENU); mavros -> NED
        od.child_frame_id = self.odom_child       # body (FLU)
        # pose = raw VO + anchor offset. Read from msg, write into the fresh od
        # Pose so msg is NOT mutated (self.last must stay RAW for velocity +
        # anchor math). applied_offset is 0 unless an anchor is correcting.
        od.pose.pose.position.x = msg.pose.pose.position.x + self.applied_offset[0]
        od.pose.pose.position.y = msg.pose.pose.position.y + self.applied_offset[1]
        od.pose.pose.position.z = msg.pose.pose.position.z + self.applied_offset[2]
        od.pose.pose.orientation = msg.pose.pose.orientation
        pc = [0.0] * 36
        pc[0] = pc[7] = self.pos_var              # x, y
        pc[14] = self.pos_var_z                   # z: ~100% VIO trust (tight)
        pc[21] = pc[28] = pc[35] = 0.01           # orientation var (rad^2)
        od.pose.covariance = pc
        if vel is not None:
            q = (msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,
                 msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
            bvx, bvy, bvz = quat_rotate_inverse(q, vel)  # world -> body twist
            od.twist.twist.linear.x = bvx
            od.twist.twist.linear.y = bvy
            od.twist.twist.linear.z = bvz
            tc = [0.0] * 36
            tc[0] = tc[7] = tc[14] = self.vel_var
            tc[21] = tc[28] = tc[35] = 1e6        # angular: no info
            od.twist.covariance = tc
        else:
            od.twist.covariance = [1e6 if i in (0, 7, 14, 21, 28, 35) else 0.0
                                   for i in range(36)]
        self.pub_odom.publish(od)

    def _pub_pose_speed(self, msg, vel):
        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        out.pose.position.x = msg.pose.pose.position.x + self.applied_offset[0]
        out.pose.position.y = msg.pose.pose.position.y + self.applied_offset[1]
        out.pose.position.z = msg.pose.pose.position.z + self.applied_offset[2]
        out.pose.orientation = msg.pose.pose.orientation
        self.pub.publish(out)
        if vel is not None:
            tw = TwistWithCovarianceStamped()
            tw.header.stamp = msg.header.stamp
            tw.header.frame_id = self.frame_id
            tw.twist.twist.linear.x = vel[0]
            tw.twist.twist.linear.y = vel[1]
            tw.twist.twist.linear.z = vel[2]
            cov = [0.0] * 36
            cov[0] = cov[7] = cov[14] = self.vel_var
            cov[21] = cov[28] = cov[35] = 1e6
            tw.twist.covariance = cov
            self.pub_v.publish(tw)


if __name__ == "__main__":
    rospy.init_node("svo_vision_relay")
    SvoVisionRelay()
    rospy.spin()
