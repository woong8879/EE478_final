#!/usr/bin/env python3
"""waypoint_markers.py

Publish the course waypoints as a LATCHED MarkerArray (/course_waypoints) in the
map frame, so RViz shows the INTENDED course (absolute / map coords) next to the
drone's ACTUAL trajectory -> you can eyeball how far the path drifts from each
target. The spheres are drawn at the arrival radius (0.5 m), so if the trajectory
passes inside a sphere the drone "reached" that waypoint.

Standalone (ground review):  python3 waypoint_markers.py
With a different course:      python3 waypoint_markers.py _waypoints:="2,0; 4,0"
In a launch (live flight):    <node ... type="waypoint_markers.py">
                                 <param name="waypoints" value="$(arg waypoints)"/>
"""
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# Default = the current s4 course (keep in sync with s4_nav_course.launch).
DEFAULT_WPS = ("1.5,0; 3.5,1; 6.5,0; 9.9,0; 9.9,1.5; 8.9,3.5; "
               "9.9,6.5; 9.9,7.2; 5.5,7.2; 3.5,6.2; 1,7.2; 0.25,8.95")


def parse_wps(s):
    pts = []
    for seg in s.split(';'):
        seg = seg.strip()
        if not seg:
            continue
        v = [float(x) for x in seg.split(',')]
        pts.append((v[0], v[1]))
    return pts


def main():
    rospy.init_node('waypoint_markers')
    wps = parse_wps(rospy.get_param('~waypoints', DEFAULT_WPS))
    z = float(rospy.get_param('~z', 0.7))
    frame = rospy.get_param('~frame_id', 'map')
    radius = float(rospy.get_param('~arrival_radius', 0.5))  # = ego arrival radius

    pub = rospy.Publisher('/course_waypoints', MarkerArray,
                          queue_size=1, latch=True)
    ma = MarkerArray()

    # line connecting the waypoints (the intended path)
    line = Marker()
    line.header.frame_id = frame
    line.ns, line.id = 'wp_line', 0
    line.type, line.action = Marker.LINE_STRIP, Marker.ADD
    line.scale.x = 0.04
    line.color.r = line.color.g = line.color.b = 1.0
    line.color.a = 0.6
    line.pose.orientation.w = 1.0
    for x, y in wps:
        p = Point()
        p.x, p.y, p.z = x, y, z
        line.points.append(p)
    ma.markers.append(line)

    # arrival sphere + index label at each waypoint
    for i, (x, y) in enumerate(wps):
        s = Marker()
        s.header.frame_id = frame
        s.ns, s.id = 'wp', i + 1
        s.type, s.action = Marker.SPHERE, Marker.ADD
        s.pose.position.x, s.pose.position.y, s.pose.position.z = x, y, z
        s.pose.orientation.w = 1.0
        s.scale.x = s.scale.y = s.scale.z = 2.0 * radius  # diameter
        s.color.r, s.color.g, s.color.b, s.color.a = 1.0, 1.0, 0.0, 0.30
        ma.markers.append(s)

        t = Marker()
        t.header.frame_id = frame
        t.ns, t.id = 'wp_txt', 100 + i
        t.type, t.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z + 0.35
        t.pose.orientation.w = 1.0
        t.scale.z = 0.25
        t.color.r = t.color.g = t.color.b = t.color.a = 1.0
        t.text = "%d (%.1f,%.1f)" % (i, x, y)
        ma.markers.append(t)

    pub.publish(ma)
    rospy.loginfo("[waypoint_markers] %d waypoints on /course_waypoints "
                  "(z=%.2f, frame=%s, r=%.2f)", len(wps), z, frame, radius)
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
