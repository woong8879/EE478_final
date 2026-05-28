std_msgs/Header header
string class_name
float32 confidence
# 2D bbox in image (pixels)
int32 x_min
int32 y_min
int32 x_max
int32 y_max
# 3D position of the object centroid in base_link frame (meters)
geometry_msgs/Point position_base
# 3D position in camera optical frame (raw, before transform)
geometry_msgs/Point position_camera
