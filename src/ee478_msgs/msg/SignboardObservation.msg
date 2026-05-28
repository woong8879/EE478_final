std_msgs/Header header
# Bundle id of the AprilTag triplet that anchored this signboard, if any (-1 if unknown).
int32 bundle_id
# Detected icon labels in order, e.g. ["PHARMACY", "CAFE"] (or empty if none).
string[] icon_labels
# Direction arrow per icon (parallel to icon_labels): "LEFT" / "RIGHT" / "STRAIGHT" / "NONE".
string[] arrow_labels
# Pixel center of the entire signboard frame
int32 cx
int32 cy
# 3D position of the signboard center in base_link frame (meters). NaN if unavailable.
geometry_msgs/Point position_base
